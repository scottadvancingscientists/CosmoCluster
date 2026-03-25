"""Minimal hybrid prototype for CosmoCluster.

Implements:
1) Expansion phase (global repulsion + local graph cohesion)
2) Gravity phase (graph-local attraction driven by density-derived mass)
3) Basin extraction (gradient ascent on graph potential)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HybridParams:
    """Hyperparameters for the minimal hybrid clustering routine."""

    k: int = 10
    expansion_steps: int = 40
    gravity_steps: int = 40
    expansion_step_size: float = 0.03
    gravity_step_size: float = 0.04
    repulsion_strength: float = 1.0
    cohesion_strength: float = 0.2
    gravity_strength: float = 1.2
    damping: float = 0.85
    mass_power: float = 1.0
    max_repulsion_neighbors: int = 64
    random_state: int | None = None


class CosmoClusterMinimal:
    """Minimal hybrid force-based clustering prototype.

    This prototype intentionally favors clarity and easy iteration over speed.
    """

    def __init__(self, params: HybridParams | None = None) -> None:
        self.params = params or HybridParams()

        self.labels_: np.ndarray | None = None
        self.positions_: np.ndarray | None = None
        self.masses_: np.ndarray | None = None
        self.attractors_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "CosmoClusterMinimal":
        """Fit the minimal hybrid pipeline and populate labels_."""
        x = _validate_array(x)
        n = x.shape[0]
        if n < 2:
            raise ValueError("Need at least two samples to cluster.")

        k = min(max(1, self.params.k), n - 1)
        graph_idx, graph_dist = _knn_graph(x, k=k)

        # Start from normalized input coordinates.
        z = _standardize(x)

        z = _expansion_phase(z, graph_idx, graph_dist, self.params)

        density = 1.0 / (graph_dist.mean(axis=1) + 1e-9)
        masses = np.power(density + 1e-9, self.params.mass_power)
        z = _gravity_phase(z, graph_idx, masses, self.params)

        attractors, labels = _extract_basins(graph_idx, masses)

        self.positions_ = z
        self.masses_ = masses
        self.attractors_ = attractors
        self.labels_ = labels
        return self

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        assert self.labels_ is not None
        return self.labels_


def _validate_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Input contains non-finite values.")
    return x


def _standardize(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True) + 1e-9
    return (x - mu) / sigma


def _pairwise_distances(x: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - x[None, :, :]
    return np.linalg.norm(diff, axis=2)


def _knn_graph(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    dmat = _pairwise_distances(x)
    np.fill_diagonal(dmat, np.inf)
    idx = np.argpartition(dmat, kth=k - 1, axis=1)[:, :k]
    row = np.arange(x.shape[0])[:, None]
    dist = dmat[row, idx]
    order = np.argsort(dist, axis=1)
    idx = idx[row, order]
    dist = dist[row, order]
    return idx, dist


def _expansion_phase(
    z: np.ndarray,
    graph_idx: np.ndarray,
    graph_dist: np.ndarray,
    params: HybridParams,
) -> np.ndarray:
    n = z.shape[0]
    vel = np.zeros_like(z)

    # local equilibrium length scale from graph
    target = np.median(graph_dist)
    rep_cap = min(params.max_repulsion_neighbors, n - 1)

    for t in range(params.expansion_steps):
        progress = t / max(1, params.expansion_steps - 1)
        w_rep = (1.0 - progress) ** 1.2
        w_coh = 0.2 + 0.8 * progress

        force = np.zeros_like(z)
        dmat = _pairwise_distances(z)
        np.fill_diagonal(dmat, np.inf)

        # global repulsion (capped nearest-neighbor subset for stability/cost)
        nn_idx = np.argpartition(dmat, kth=rep_cap - 1, axis=1)[:, :rep_cap]
        for i in range(n):
            neigh = nn_idx[i]
            delta = z[i] - z[neigh]
            dist = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-6
            rep = delta / (dist**3)
            force[i] += params.repulsion_strength * w_rep * rep.sum(axis=0)

        # graph cohesion spring toward local neighborhood centroid
        for i in range(n):
            neigh = graph_idx[i]
            centroid = z[neigh].mean(axis=0)
            spring = centroid - z[i]
            # softly preserve neighborhood scale
            scale = np.clip(target / (graph_dist[i].mean() + 1e-9), 0.5, 2.0)
            force[i] += params.cohesion_strength * w_coh * scale * spring

        vel = params.damping * vel + params.expansion_step_size * force
        z = z + vel

    return z


def _gravity_phase(
    z: np.ndarray,
    graph_idx: np.ndarray,
    masses: np.ndarray,
    params: HybridParams,
) -> np.ndarray:
    vel = np.zeros_like(z)

    for t in range(params.gravity_steps):
        progress = t / max(1, params.gravity_steps - 1)
        w_grav = 0.3 + 0.7 * progress

        force = np.zeros_like(z)
        for i in range(z.shape[0]):
            neigh = graph_idx[i]
            delta = z[neigh] - z[i]
            dist = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-6
            unit = delta / dist
            strength = (masses[neigh, None] / (dist**2))
            g = (unit * strength).sum(axis=0)
            force[i] += params.gravity_strength * w_grav * g

        vel = params.damping * vel + params.gravity_step_size * force
        z = z + vel

    return z


def _extract_basins(graph_idx: np.ndarray, masses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = graph_idx.shape[0]
    parent = np.arange(n)

    # one-step flow: move to highest-mass neighbor if strictly better
    for i in range(n):
        neigh = graph_idx[i]
        best = neigh[np.argmax(masses[neigh])]
        if masses[best] > masses[i]:
            parent[i] = best

    # path compression to final attractor
    def find(u: int) -> int:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    attractor = np.array([find(i) for i in range(n)], dtype=int)
    uniq, labels = np.unique(attractor, return_inverse=True)
    # map attractor ids to compact ids then back to node indices for transparency
    compact_to_node = uniq
    attractor_node = compact_to_node[labels]
    return attractor_node, labels
