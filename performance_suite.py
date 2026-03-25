"""Performance benchmark suite for CosmoCluster vs common clustering baselines.

Runs a synthetic benchmark battery and reports quality/runtime metrics plus plots.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from cosmocluster import CosmoClusterMinimal, HybridParams
from synthetic_suite import SyntheticCase, create_hdbscan_challenge_suite

try:
    import hdbscan
except Exception:  # pragma: no cover - optional dependency
    hdbscan = None


@dataclass(frozen=True)
class AlgoSpec:
    name: str
    factory: Callable[[int, int], object]


@dataclass(frozen=True)
class ClusterSnapshot:
    case: str
    x: np.ndarray
    y_true: np.ndarray
    y_pred_by_algo: dict[str, np.ndarray]


def _safe_cluster_metrics(x: np.ndarray, pred_labels: np.ndarray, true_labels: np.ndarray) -> dict[str, float]:
    non_noise = pred_labels[pred_labels != -1]
    n_pred_clusters = len(np.unique(non_noise))

    metrics: dict[str, float] = {
        "ari": adjusted_rand_score(true_labels, pred_labels),
        "nmi": normalized_mutual_info_score(true_labels, pred_labels),
        "n_clusters": float(n_pred_clusters),
        "noise_frac": float(np.mean(pred_labels == -1)),
    }

    if n_pred_clusters > 1 and np.unique(pred_labels).size > 1:
        metrics["silhouette"] = float(silhouette_score(x, pred_labels))
        metrics["davies_bouldin"] = float(davies_bouldin_score(x, pred_labels))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(x, pred_labels))
    else:
        metrics["silhouette"] = np.nan
        metrics["davies_bouldin"] = np.nan
        metrics["calinski_harabasz"] = np.nan

    return metrics


def _algo_specs() -> list[AlgoSpec]:
    specs = [
        AlgoSpec(
            name="cosmocluster_minimal",
            factory=lambda n_clusters, seed: CosmoClusterMinimal(
                HybridParams(
                    k=12,
                    expansion_steps=35,
                    gravity_steps=35,
                    random_state=seed,
                )
            ),
        ),
        AlgoSpec(
            name="kmeans",
            factory=lambda n_clusters, seed: KMeans(n_clusters=n_clusters, random_state=seed, n_init=10),
        ),
        AlgoSpec(name="dbscan", factory=lambda n_clusters, seed: DBSCAN(eps=0.27, min_samples=8)),
        AlgoSpec(
            name="agglomerative",
            factory=lambda n_clusters, seed: AgglomerativeClustering(n_clusters=n_clusters, linkage="ward"),
        ),
    ]

    if hdbscan is not None:
        specs.append(
            AlgoSpec(
                name="hdbscan",
                factory=lambda n_clusters, seed: hdbscan.HDBSCAN(
                    min_cluster_size=25,
                    min_samples=8,
                    cluster_selection_method="eom",
                ),
            )
        )

    return specs


def _fit_predict(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "fit_predict"):
        return np.asarray(model.fit_predict(x))
    fitted = model.fit(x)
    labels = getattr(fitted, "labels_", None)
    if labels is None:
        raise ValueError(f"Model {type(model).__name__} did not expose labels_.")
    return np.asarray(labels)


def run_benchmark(
    n_samples: int,
    noise: float,
    random_state: int,
    repeats: int,
) -> tuple[pd.DataFrame, list[ClusterSnapshot]]:
    suite = create_hdbscan_challenge_suite(n_samples=n_samples, noise=noise, random_state=random_state)
    algos = _algo_specs()

    rows: list[dict[str, float | int | str]] = []
    snapshots: list[ClusterSnapshot] = []
    for case_idx, case in enumerate(suite):
        n_true_clusters = int(np.unique(case.y).size)
        y_pred_by_algo: dict[str, np.ndarray] = {}
        for rep in range(repeats):
            seed = random_state + case_idx * 100 + rep
            for algo in algos:
                model = algo.factory(n_true_clusters, seed)
                start = time.perf_counter()
                pred = _fit_predict(model, case.x)
                fit_seconds = time.perf_counter() - start
                if rep == 0:
                    y_pred_by_algo[algo.name] = pred.copy()

                vals = _safe_cluster_metrics(case.x, pred, case.y)
                row: dict[str, float | int | str] = {
                    "case": case.name,
                    "algorithm": algo.name,
                    "repeat": rep,
                    "fit_seconds": fit_seconds,
                    "n_true_clusters": n_true_clusters,
                    "notes": case.notes,
                }
                row.update(vals)
                rows.append(row)

        snapshots.append(
            ClusterSnapshot(
                case=case.name,
                x=case.x.copy(),
                y_true=case.y.copy(),
                y_pred_by_algo=y_pred_by_algo,
            )
        )

    return pd.DataFrame(rows), snapshots


def render_visualizations(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    summary = (
        df.groupby("algorithm", as_index=False)
        .agg(
            ari_mean=("ari", "mean"),
            ari_std=("ari", "std"),
            nmi_mean=("nmi", "mean"),
            fit_seconds_median=("fit_seconds", "median"),
        )
        .sort_values("ari_mean", ascending=False)
    )

    plt.figure(figsize=(10, 5))
    plt.bar(summary["algorithm"], summary["ari_mean"], yerr=summary["ari_std"].fillna(0.0), capsize=4)
    plt.ylabel("ARI (mean ± std)")
    plt.title("Clustering quality across synthetic benchmark")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "quality_bar.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    data = [df.loc[df["algorithm"] == a, "fit_seconds"].to_numpy() for a in summary["algorithm"]]
    plt.boxplot(data, tick_labels=summary["algorithm"], showfliers=False)
    plt.ylabel("Fit time (seconds)")
    plt.title("Runtime distribution by algorithm")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "runtime_boxplot.png", dpi=180)
    plt.close()

    pivot = (
        df.groupby(["case", "algorithm"], as_index=False)["ari"]
        .mean()
        .pivot(index="case", columns="algorithm", values="ari")
        .sort_index()
    )

    plt.figure(figsize=(11, 6))
    im = plt.imshow(pivot.to_numpy(), aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    plt.colorbar(im, label="Mean ARI")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.title("Per-case clustering quality (ARI)")
    plt.tight_layout()
    plt.savefig(outdir / "ari_heatmap.png", dpi=180)
    plt.close()


def _label_to_color(labels: np.ndarray) -> np.ndarray:
    mapped = labels.copy()
    noise_mask = mapped == -1
    mapped = mapped.astype(int)
    if noise_mask.any():
        mapped = mapped + 1
        mapped[noise_mask] = 0
    return mapped


def render_cluster_htmls(snapshots: list[ClusterSnapshot], outdir: Path) -> None:
    html_dir = outdir / "clusters_html"
    html_dir.mkdir(parents=True, exist_ok=True)

    for snap in snapshots:
        algo_names = sorted(snap.y_pred_by_algo.keys())
        cols = len(algo_names) + 1
        fig = make_subplots(
            rows=1,
            cols=cols,
            subplot_titles=["ground_truth", *algo_names],
            horizontal_spacing=0.02,
        )
        plot_sets = [("ground_truth", snap.y_true), *[(a, snap.y_pred_by_algo[a]) for a in algo_names]]

        for col_idx, (name, labels) in enumerate(plot_sets, start=1):
            fig.add_trace(
                go.Scattergl(
                    x=snap.x[:, 0],
                    y=snap.x[:, 1],
                    mode="markers",
                    marker={
                        "size": 4,
                        "color": _label_to_color(labels),
                        "colorscale": "Turbo",
                        "opacity": 0.78,
                        "showscale": col_idx == cols,
                    },
                    name=name,
                    showlegend=False,
                    text=[f"label={int(v)}" for v in labels],
                    hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>%{text}<extra></extra>",
                ),
                row=1,
                col=col_idx,
            )
            fig.update_xaxes(title_text="x1", row=1, col=col_idx)
            fig.update_yaxes(title_text="x2", row=1, col=col_idx)

        fig.update_layout(
            title=f"Cluster geometry by algorithm — {snap.case}",
            width=max(1300, 340 * cols),
            height=420,
        )
        fig.write_html(html_dir / f"{snap.case}_clusters.html", include_plotlyjs="cdn")


def render_geometry_html(snapshots: list[ClusterSnapshot], outdir: Path, k: int = 12) -> None:
    records: list[dict[str, float | str]] = []
    for snap in snapshots:
        x = snap.x
        dmat = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
        np.fill_diagonal(dmat, np.inf)
        knn = np.partition(dmat, kth=k - 1, axis=1)[:, :k]
        local_scale = knn.mean(axis=1)
        radial = np.linalg.norm(x - x.mean(axis=0, keepdims=True), axis=1)
        for i in range(x.shape[0]):
            records.append(
                {
                    "case": snap.case,
                    "local_scale": float(local_scale[i]),
                    "radius_from_center": float(radial[i]),
                }
            )
    geom = pd.DataFrame(records)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Local neighborhood scale", "Radial spread"])
    for case in sorted(geom["case"].unique()):
        part = geom[geom["case"] == case]
        fig.add_trace(
            go.Box(y=part["local_scale"], name=case, boxpoints=False, showlegend=False),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Box(y=part["radius_from_center"], name=case, boxpoints=False, showlegend=False),
            row=1,
            col=2,
        )
    fig.update_layout(
        title="Geometry diagnostics across synthetic cases",
        width=1400,
        height=520,
    )
    fig.write_html(outdir / "geometry_diagnostics.html", include_plotlyjs="cdn")


def _markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *rows])


def write_report(df: pd.DataFrame, snapshots: list[ClusterSnapshot], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "benchmark_results.csv", index=False)

    summary = (
        df.groupby("algorithm", as_index=False)
        .agg(
            ari_mean=("ari", "mean"),
            nmi_mean=("nmi", "mean"),
            silhouette_mean=("silhouette", "mean"),
            fit_seconds_median=("fit_seconds", "median"),
            fit_seconds_p90=("fit_seconds", lambda x: np.percentile(x, 90)),
            noise_frac_mean=("noise_frac", "mean"),
        )
        .sort_values(["ari_mean", "nmi_mean"], ascending=False)
    )
    summary.to_csv(outdir / "benchmark_summary.csv", index=False)

    top_by_case = (
        df.groupby(["case", "algorithm"], as_index=False)["ari"]
        .mean()
        .sort_values(["case", "ari"], ascending=[True, False])
        .groupby("case", as_index=False)
        .head(1)
    )

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": int(len(df)),
        "algorithms": sorted(df["algorithm"].unique().tolist()),
        "cases": sorted(df["case"].unique().tolist()),
        "top_algorithm_per_case": top_by_case.to_dict(orient="records"),
        "cluster_html_dir": "clusters_html",
        "geometry_html": "geometry_diagnostics.html",
    }
    (outdir / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# CosmoCluster Performance Benchmark Report",
        "",
        "## Aggregate results",
        "",
        _markdown_table(summary),
        "",
        "## Best ARI by synthetic case",
        "",
        _markdown_table(top_by_case),
        "",
        "## Visualizations",
        "",
        "![Quality bar](quality_bar.png)",
        "",
        "![Runtime boxplot](runtime_boxplot.png)",
        "",
        "![ARI heatmap](ari_heatmap.png)",
        "",
        "## Interactive geometry",
        "",
        "- [Geometry diagnostics](geometry_diagnostics.html)",
        "- Per-case cluster HTMLs (ground truth + each algorithm):",
        *[f"  - [clusters_html/{snap.case}_clusters.html](clusters_html/{snap.case}_clusters.html)" for snap in snapshots],
    ]
    (outdir / "REPORT.md").write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clustering benchmark suite with visualizations.")
    parser.add_argument("--n-samples", type=int, default=1500)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--outdir", type=Path, default=Path("artifacts/performance"))
    args = parser.parse_args()

    df, snapshots = run_benchmark(
        n_samples=args.n_samples,
        noise=args.noise,
        random_state=args.random_state,
        repeats=args.repeats,
    )
    write_report(df, snapshots, args.outdir)
    render_visualizations(df, args.outdir)
    render_cluster_htmls(snapshots, args.outdir)
    render_geometry_html(snapshots, args.outdir)
    print(f"Benchmark complete. Wrote results to: {args.outdir}")


if __name__ == "__main__":
    main()
