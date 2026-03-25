"""Synthetic benchmark suite for force-based clustering.

The suite intentionally creates geometric structures that are often difficult for
single-bias density clustering (e.g. HDBSCAN default settings) and are better
matched by expansion->gravity style dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticCase:
    """Container for one synthetic clustering benchmark case."""

    name: str
    x: np.ndarray
    y: np.ndarray
    notes: str


def create_hdbscan_challenge_suite(
    n_samples: int = 2000,
    noise: float = 0.025,
    random_state: int = 42,
) -> list[SyntheticCase]:
    """Create a suite of 10 synthetic datasets.

    Parameters
    ----------
    n_samples:
        Approximate total number of samples per dataset.
    noise:
        Base isotropic Gaussian noise level added to coordinates.
    random_state:
        Seed for reproducibility.
    """
    if n_samples < 400:
        raise ValueError("n_samples must be >= 400 for balanced structure generation.")

    rng = np.random.default_rng(random_state)

    cases = [
        _case_density_bridge(n_samples, noise, rng),
        _case_crossing_filaments(n_samples, noise, rng),
        _case_nested_rings_with_gaps(n_samples, noise, rng),
        _case_spiral_with_satellites(n_samples, noise, rng),
        _case_variable_width_moons(n_samples, noise, rng),
        _case_hub_and_spokes(n_samples, noise, rng),
        _case_ladder_manifold(n_samples, noise, rng),
        _case_cluster_chain_with_bottlenecks(n_samples, noise, rng),
        _case_shell_and_core(n_samples, noise, rng),
        _case_hierarchical_subclusters(n_samples, noise, rng),
    ]
    return cases


def _sample_split(total: int, weights: list[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = np.maximum(w, 1e-9)
    w = w / w.sum()
    counts = np.floor(total * w).astype(int)
    counts[: total - counts.sum()] += 1
    return counts


def _add_noise(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return x + rng.normal(scale=sigma, size=x.shape)


def _case_density_bridge(n: int, sigma: float, rng: np.random.Generator) -> SyntheticCase:
    c = _sample_split(n, [0.42, 0.42, 0.16])
    left = rng.normal(loc=(-2.6, 0.0), scale=(0.25, 0.35), size=(c[0], 2))
    right = rng.normal(loc=(2.6, 0.0), scale=(0.25, 0.35), size=(c[1], 2))
    t = rng.uniform(-1.0, 1.0, size=c[2])
    bridge = np.column_stack((2.3 * t, 0.2 * np.sin(5 * t)))
    x = _add_noise(np.vstack((left, right, bridge)), sigma, rng)
    y = np.concatenate((np.zeros(c[0], int), np.ones(c[1], int), np.full(c[2], 2, int)))
    return SyntheticCase(
        name="density_bridge",
        x=x,
        y=y,
        notes="Two dense lobes connected by a sparse bridge; tests over-merging behavior.",
    )


def _case_crossing_filaments(n: int, sigma: float, rng: np.random.Generator) -> SyntheticCase:
    c = _sample_split(n, [0.34, 0.33, 0.33])
    t1 = rng.uniform(-2.5, 2.5, c[0])
    f1 = np.column_stack((t1, 0.6 * np.sin(1.5 * t1)))
    t2 = rng.uniform(-2.5, 2.5, c[1])
    f2 = np.column_stack((0.8 * np.sin(1.7 * t2), t2))
    t3 = rng.uniform(-2.5, 2.5, c[2])
    f3 = np.column_stack((t3, -0.4 * np.sin(2.1 * t3) - 0.8))
    x = _add_noise(np.vstack((f1, f2, f3)), sigma, rng)
    y = np.concatenate((np.zeros(c[0], int), np.ones(c[1], int), np.full(c[2], 2, int)))
    return SyntheticCase(
        name="crossing_filaments",
        x=x,
        y=y,
        notes="Intersections of non-convex filaments; tests manifold-following and crossing disambiguation.",
    )


def _case_nested_rings_with_gaps(n: int, sigma: float, rng: np.random.Generator) -> SyntheticCase:
    c = _sample_split(n, [0.36, 0.34, 0.30])

    def ring(count: int, r: float, cut: tuple[float, float]) -> np.ndarray:
        theta = rng.uniform(0.0, 2 * np.pi, count * 2)
        mask = (theta < cut[0]) | (theta > cut[1])
        theta = theta[mask][:count]
        return np.column_stack((r * np.cos(theta), r * np.sin(theta)))

    r1 = ring(c[0], 1.3, (0.2 * np.pi, 0.45 * np.pi))
    r2 = ring(c[1], 2.2, (1.1 * np.pi, 1.35 * np.pi))
    r3 = ring(c[2], 3.1, (1.75 * np.pi, 1.95 * np.pi))
    x = _add_noise(np.vstack((r1, r2, r3)), sigma, rng)
    y = np.concatenate((np.zeros(c[0], int), np.ones(c[1], int), np.full(c[2], 2, int)))
    return SyntheticCase(
        name="nested_rings_gapped",
        x=x,
        y=y,
        notes="Nested partial rings with missing arcs; tests robustness to local density discontinuities.",
    )


def _case_spiral_with_satellites(n: int, sigma: float, rng: np.random.Generator) -> SyntheticCase:
    c = _sample_split(n, [0.55, 0.30, 0.15])

    def spiral(count: int, phase: float) -> np.ndarray:
        t = rng.uniform(0.2, 4.8 * np.pi, count)
        r = 0.18 * t
        return np.column_stack((r * np.cos(t + phase), r * np.sin(t + phase)))

    s1 = spiral(c[0], 0.0)
    s2 = spiral(c[1], np.pi)
    sat = rng.normal(loc=(3.3, -2.6), scale=(0.22, 0.22), size=(c[2], 2))
    x = _add_noise(np.vstack((s1, s2, sat)), sigma, rng)
    y = np.concatenate((np.zeros(c[0], int), np.ones(c[1], int), np.full(c[2], 2, int)))
    return SyntheticCase(
        name="double_spiral_satellite",
        x=x,
        y=y,
        notes="Two intertwined spirals plus a detached micro-cluster; tests multi-scale attractor basins.",
    )


def _case_variable_width_moons(n: int, sigma: float, rng: np.random.Generator) -> SyntheticCase:
    c = _sample_split(n, [0.5, 0.5])
    t1 = rng.uniform(0.0, np.pi, c[0])
    t2 = rng.uniform(0.0, np.pi, c[1])
    moon1 = np.column_stack((np.cos(t1), np.sin(t1)))
    moon2 = np.column_stack((1.0 - np.cos(t2), 0.35 - np.sin(t2)))
    moon1 += rng.normal(scale=np.column_stack((0.025 + 0.08 * (t1 / np.pi), np.full(c[0], 0.03))))
    moon2 += rng.normal(scale=np.column_stack((np.full(c[1], 0.03), 0.02 + 0.09 * (t2 / np.pi))))
    x = _add_noise(np.vstack((2.8 * moon1, 2.8 * moon2)), sigma, rng)
    y = np.concatenate((np.zeros(c[0], int), np.ones(c[1], int)))
    return SyntheticCase(
        name="variable_width_moons",
        x=x,
        y=y,
        notes="Crescent pairs with heteroscedastic width; tests sensitivity to varying local scale.",
    )


def _case_hub_and_spokes(n: int, sigma: float, rng: np.random.Generator) -> SyntheticCase:
    c = _sample_split(n, [0.2, 0.2, 0.2, 0.2, 0.2])
    hub = rng.normal(loc=(0.0, 0.0), scale=(0.26, 0.26), size=(c[0], 2))
    arms = []
    labels = [np.zeros(c[0], int)]
    angles = [0.0, 0.6 * np.pi, 1.15 * np.pi, 1.7 * np.pi]
    for i, (count, ang) in enumerate(zip(c[1:], angles, strict=True), start=1):
        t = rng.uniform(0.2, 3.2, count)
        arm = np.column_stack((t * np.cos(ang), t * np.sin(ang)))
        arm += rng.normal(scale=(0.07, 0.07), size=arm.shape)
        arms.append(arm)
        labels.append(np.full(count, i, int))
    x = _add_noise(np.vstack([hub, *arms]), sigma, rng)
    y = np.concatenate(labels)
    return SyntheticCase(
        name="hub_spokes",
        x=x,
        y=y,
        notes="Radial topology with dense center and elongated arms; tests center-collapse tendencies.",
    )


def _case_ladder_manifold(n: int, sigma: float, rng: np.random.Generator) -> SyntheticCase:
    c = _sample_split(n, [0.4, 0.4, 0.2])
    t_top = rng.uniform(-3.0, 3.0, c[0])
    t_bot = rng.uniform(-3.0, 3.0, c[1])
    top = np.column_stack((t_top, 1.1 + 0.2 * np.sin(1.3 * t_top)))
    bot = np.column_stack((t_bot, -1.1 + 0.2 * np.sin(1.5 * t_bot + 0.7)))
    rung_x = rng.uniform(-2.8, 2.8, c[2])
    rungs = np.column_stack((rung_x, rng.uniform(-0.95, 0.95, c[2])))
    x = _add_noise(np.vstack((top, bot, rungs)), sigma, rng)
    y = np.concatenate((np.zeros(c[0], int), np.ones(c[1], int), np.full(c[2], 2, int)))
    return SyntheticCase(
        name="ladder_manifold",
        x=x,
        y=y,
        notes="Two noisy rails joined by sparse rungs; tests bridge handling in graph-based density methods.",
    )


def _case_cluster_chain_with_bottlenecks(n: int, sigma: float, rng: np.random.Generator) -> SyntheticCase:
    c = _sample_split(n, [0.22, 0.22, 0.22, 0.22, 0.12])
    centers = [(-3.2, -1.2), (-1.3, -0.2), (0.9, 0.6), (3.0, 1.4)]
    blobs = [rng.normal(loc=ctr, scale=(0.32, 0.22), size=(count, 2)) for ctr, count in zip(centers, c[:4], strict=True)]
    t = rng.uniform(-1.0, 1.0, c[4])
    bottleneck = np.column_stack((2.8 * t, 0.15 * np.sin(10 * t) + 0.25 * t))
    x = _add_noise(np.vstack([*blobs, bottleneck]), sigma, rng)
    y = np.concatenate(
        [
            np.zeros(c[0], int),
            np.ones(c[1], int),
            np.full(c[2], 2, int),
            np.full(c[3], 3, int),
            np.full(c[4], 4, int),
        ]
    )
    return SyntheticCase(
        name="cluster_chain_bottlenecks",
        x=x,
        y=y,
        notes="Blob chain connected by narrow bottlenecks; tests persistence-based splitting decisions.",
    )


def _case_shell_and_core(n: int, sigma: float, rng: np.random.Generator) -> SyntheticCase:
    c = _sample_split(n, [0.45, 0.35, 0.2])
    core = rng.normal(loc=(0.0, 0.0), scale=(0.23, 0.23), size=(c[0], 2))
    theta = rng.uniform(0.0, 2 * np.pi, c[1])
    shell = np.column_stack((2.4 * np.cos(theta), 1.8 * np.sin(theta)))
    shell += rng.normal(scale=(0.09, 0.09), size=shell.shape)
    arc_theta = rng.uniform(0.2 * np.pi, 0.9 * np.pi, c[2])
    arc = np.column_stack((3.0 * np.cos(arc_theta), 2.5 * np.sin(arc_theta) - 0.3))
    x = _add_noise(np.vstack((core, shell, arc)), sigma, rng)
    y = np.concatenate((np.zeros(c[0], int), np.ones(c[1], int), np.full(c[2], 2, int)))
    return SyntheticCase(
        name="shell_core_arc",
        x=x,
        y=y,
        notes="Concentric density inversion (dense core + sparse shell + arc); tests scale-adaptive assignment.",
    )


def _case_hierarchical_subclusters(n: int, sigma: float, rng: np.random.Generator) -> SyntheticCase:
    parent_centers = [(-2.8, 2.3), (2.4, 2.1), (-0.1, -2.2)]
    parent_weights = [0.36, 0.34, 0.30]
    parent_counts = _sample_split(n, parent_weights)

    clouds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    label = 0
    for parent_idx, (center, pcount) in enumerate(zip(parent_centers, parent_counts, strict=True)):
        child_counts = _sample_split(pcount, [0.46, 0.34, 0.20])
        offsets = np.array([[0.0, 0.0], [0.5, -0.35], [-0.45, 0.4]])
        for child_idx, ccount in enumerate(child_counts):
            loc = np.array(center) + offsets[child_idx]
            scale = (0.17 + 0.04 * child_idx, 0.14 + 0.03 * parent_idx)
            part = rng.normal(loc=loc, scale=scale, size=(ccount, 2))
            clouds.append(part)
            labels.append(np.full(ccount, label, int))
            label += 1

    x = _add_noise(np.vstack(clouds), sigma, rng)
    y = np.concatenate(labels)
    return SyntheticCase(
        name="hierarchical_subclusters",
        x=x,
        y=y,
        notes="Three macro-groups each containing micro-clusters; tests hierarchy recovery and stable merges.",
    )


if __name__ == "__main__":
    suite = create_hdbscan_challenge_suite()
    for case in suite:
        print(f"{case.name:28s} shape={case.x.shape} labels={len(np.unique(case.y))} :: {case.notes}")
