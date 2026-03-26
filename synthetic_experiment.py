"""Small synthetic experiment harness for CosmoCluster parameter tuning.

This script performs a lightweight parameter search for:
1) cosmology field schedule
2) pulsed field schedule

It then compares the best configurations to common legacy algorithms
(HDBSCAN when installed, plus k-means / DBSCAN / agglomerative clustering).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from cosmocluster import CosmoClusterMinimal, FieldPulse, HybridParams
from synthetic_suite import create_hdbscan_challenge_suite

try:
    import hdbscan
except Exception:  # pragma: no cover - optional dependency
    hdbscan = None


@dataclass(frozen=True)
class Candidate:
    name: str
    params: HybridParams


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "n_clusters": float(np.unique(y_pred[y_pred != -1]).size),
        "noise_frac": float(np.mean(y_pred == -1)),
    }


def _score(row: pd.Series) -> float:
    # ARI is primary, NMI secondary, mild penalty for excessive noise.
    return float(row["ari_mean"] + 0.35 * row["nmi_mean"] - 0.10 * row["noise_frac_mean"])


def cosmology_candidates(seed: int) -> list[Candidate]:
    base = dict(
        random_state=seed,
        max_repulsion_neighbors=64,
        damping=0.85,
        mass_power=1.0,
        field_mode="cosmology",
    )
    return [
        Candidate(
            "cosmo_balanced",
            HybridParams(
                **base,
                k=12,
                expansion_steps=40,
                gravity_steps=40,
                expansion_step_size=0.030,
                gravity_step_size=0.040,
                repulsion_strength=1.00,
                cohesion_strength=0.20,
                gravity_strength=1.20,
            ),
        ),
        Candidate(
            "cosmo_separation_heavy",
            HybridParams(
                **base,
                k=14,
                expansion_steps=48,
                gravity_steps=36,
                expansion_step_size=0.034,
                gravity_step_size=0.038,
                repulsion_strength=1.20,
                cohesion_strength=0.18,
                gravity_strength=1.15,
            ),
        ),
        Candidate(
            "cosmo_condense_heavy",
            HybridParams(
                **base,
                k=10,
                expansion_steps=34,
                gravity_steps=52,
                expansion_step_size=0.026,
                gravity_step_size=0.046,
                repulsion_strength=0.90,
                cohesion_strength=0.26,
                gravity_strength=1.35,
            ),
        ),
    ]


def pulsed_candidates(seed: int) -> list[Candidate]:
    base = dict(
        random_state=seed,
        max_repulsion_neighbors=64,
        damping=0.85,
        mass_power=1.0,
        field_mode="pulsed",
        k=12,
        expansion_steps=42,
        gravity_steps=42,
        expansion_step_size=0.031,
        gravity_step_size=0.041,
        repulsion_strength=1.00,
        cohesion_strength=0.22,
        gravity_strength=1.22,
    )
    return [
        Candidate(
            "pulse_early_repulsion_late_gravity",
            HybridParams(
                **base,
                field_pulses=(
                    FieldPulse(start=0.00, end=0.55, period=0.12, duty_cycle=0.36, repulsion_gain=0.70),
                    FieldPulse(start=0.45, end=1.00, period=0.15, duty_cycle=0.50, gravity_gain=0.90),
                ),
            ),
        ),
        Candidate(
            "pulse_alternating_focus",
            HybridParams(
                **base,
                field_pulses=(
                    FieldPulse(
                        start=0.05,
                        end=0.60,
                        period=0.10,
                        duty_cycle=0.45,
                        repulsion_gain=0.55,
                        cohesion_gain=0.25,
                    ),
                    FieldPulse(
                        start=0.40,
                        end=1.00,
                        period=0.16,
                        duty_cycle=0.42,
                        gravity_gain=0.95,
                        cohesion_gain=0.20,
                    ),
                ),
            ),
        ),
        Candidate(
            "pulse_bridge_protect",
            HybridParams(
                **base,
                k=14,
                field_pulses=(
                    FieldPulse(start=0.00, end=0.70, period=0.14, duty_cycle=0.30, repulsion_gain=0.80),
                    FieldPulse(start=0.30, end=0.85, period=0.20, duty_cycle=0.55, cohesion_gain=0.35),
                    FieldPulse(start=0.55, end=1.00, period=0.14, duty_cycle=0.50, gravity_gain=1.00),
                ),
            ),
        ),
    ]


def evaluate_candidates(cases, candidates: list[Candidate], seed: int) -> tuple[pd.DataFrame, Candidate]:
    rows: list[dict[str, float | str]] = []
    for cand in candidates:
        for case_idx, case in enumerate(cases):
            params = cand.params
            params.random_state = seed + case_idx
            model = CosmoClusterMinimal(params)
            start = time.perf_counter()
            y_pred = model.fit_predict(case.x)
            elapsed = time.perf_counter() - start
            row = {"candidate": cand.name, "case": case.name, "fit_seconds": elapsed}
            row.update(_metrics(case.y, y_pred))
            rows.append(row)

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("candidate", as_index=False)
        .agg(
            ari_mean=("ari", "mean"),
            nmi_mean=("nmi", "mean"),
            noise_frac_mean=("noise_frac", "mean"),
            fit_seconds_median=("fit_seconds", "median"),
        )
        .sort_values(["ari_mean", "nmi_mean"], ascending=False)
    )
    summary["search_score"] = summary.apply(_score, axis=1)
    summary = summary.sort_values("search_score", ascending=False).reset_index(drop=True)
    best_name = str(summary.loc[0, "candidate"])
    best = next(c for c in candidates if c.name == best_name)
    return summary, best


def compare_algorithms(cases, seed: int, best_cosmo: Candidate, best_pulsed: Candidate) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    algos = [
        ("cosmocluster_cosmology_best", lambda n, s: CosmoClusterMinimal(best_cosmo.params)),
        ("cosmocluster_pulsed_best", lambda n, s: CosmoClusterMinimal(best_pulsed.params)),
        ("kmeans", lambda n, s: KMeans(n_clusters=n, random_state=s, n_init=10)),
        ("dbscan", lambda n, s: DBSCAN(eps=0.27, min_samples=8)),
        ("agglomerative", lambda n, s: AgglomerativeClustering(n_clusters=n, linkage="ward")),
    ]
    if hdbscan is not None:
        algos.append(
            (
                "hdbscan",
                lambda n, s: hdbscan.HDBSCAN(min_cluster_size=25, min_samples=8, cluster_selection_method="eom"),
            )
        )

    for case_idx, case in enumerate(cases):
        n_clusters = int(np.unique(case.y).size)
        for algo_name, factory in algos:
            model = factory(n_clusters, seed + case_idx)
            start = time.perf_counter()
            if hasattr(model, "fit_predict"):
                y_pred = np.asarray(model.fit_predict(case.x))
            else:
                fitted = model.fit(case.x)
                y_pred = np.asarray(fitted.labels_)
            elapsed = time.perf_counter() - start
            row = {"algorithm": algo_name, "case": case.name, "fit_seconds": elapsed}
            row.update(_metrics(case.y, y_pred))
            rows.append(row)

    return pd.DataFrame(rows)


def write_outputs(
    outdir: Path,
    search_cosmo: pd.DataFrame,
    search_pulsed: pd.DataFrame,
    compare_df: pd.DataFrame,
    best_cosmo: Candidate,
    best_pulsed: Candidate,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    search_cosmo.to_csv(outdir / "search_cosmology.csv", index=False)
    search_pulsed.to_csv(outdir / "search_pulsed.csv", index=False)
    compare_df.to_csv(outdir / "comparison.csv", index=False)

    summary = (
        compare_df.groupby("algorithm", as_index=False)
        .agg(
            ari_mean=("ari", "mean"),
            nmi_mean=("nmi", "mean"),
            noise_frac_mean=("noise_frac", "mean"),
            fit_seconds_median=("fit_seconds", "median"),
        )
        .sort_values(["ari_mean", "nmi_mean"], ascending=False)
    )
    summary.to_csv(outdir / "comparison_summary.csv", index=False)

    params_payload = {
        "best_cosmology": {"name": best_cosmo.name, "params": asdict(best_cosmo.params)},
        "best_pulsed": {"name": best_pulsed.name, "params": asdict(best_pulsed.params)},
    }
    (outdir / "recommended_params.json").write_text(json.dumps(params_payload, indent=2), encoding="utf-8")

    md = [
        "# Synthetic parameter experiment",
        "",
        "## Recommended parameters",
        "",
        f"- Cosmology: `{best_cosmo.name}`",
        f"- Pulsed: `{best_pulsed.name}`",
        "",
        "## Algorithm comparison (mean across selected synthetic cases)",
        "",
        summary.to_markdown(index=False),
        "",
        "## Artifacts",
        "",
        "- `search_cosmology.csv`",
        "- `search_pulsed.csv`",
        "- `comparison.csv`",
        "- `comparison_summary.csv`",
        "- `recommended_params.json`",
    ]
    (outdir / "REPORT.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune small synthetic experiment for cosmology/pulsed fields.")
    parser.add_argument("--n-samples", type=int, default=1200)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-cases", type=int, default=6, help="Use the first N synthetic cases for a quick run.")
    parser.add_argument("--outdir", type=Path, default=Path("artifacts/small_experiment"))
    args = parser.parse_args()

    cases = create_hdbscan_challenge_suite(
        n_samples=args.n_samples,
        noise=args.noise,
        random_state=args.random_state,
    )
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    search_cosmo, best_cosmo = evaluate_candidates(cases, cosmology_candidates(args.random_state), args.random_state)
    search_pulsed, best_pulsed = evaluate_candidates(cases, pulsed_candidates(args.random_state), args.random_state + 500)

    compare_df = compare_algorithms(cases, args.random_state + 900, best_cosmo, best_pulsed)

    write_outputs(args.outdir, search_cosmo, search_pulsed, compare_df, best_cosmo, best_pulsed)
    print(f"Synthetic experiment complete. Results written to: {args.outdir}")


if __name__ == "__main__":
    main()
