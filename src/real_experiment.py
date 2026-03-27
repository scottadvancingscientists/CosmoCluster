from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cosmocluster import CosmoClusterMinimal, HybridParams
from synthetic_suite import SyntheticCase, create_hdbscan_challenge_suite

try:
    from hdbscan import HDBSCAN
except Exception:  # pragma: no cover - optional dependency fallback.
    HDBSCAN = None

try:
    import umap
except Exception:  # pragma: no cover - optional dependency fallback.
    umap = None


@dataclass(frozen=True)
class RealRunResult:
    metrics: dict[str, float]
    case_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    fit_seconds: float
    epochs: int


def _choose_cases(config: dict) -> list[SyntheticCase]:
    dataset_name = str(config.get("dataset_name", "")).lower()
    suite = create_hdbscan_challenge_suite(
        n_samples=max(600, int(config.get("hyperparameters", {}).get("batch_size", 64)) * 12),
        noise=0.03,
        random_state=int(config.get("seed", 42)),
    )
    by_name = {c.name: c for c in suite}
    alias = {
        "synthetic-moons": "variable_width_moons",
        "synthetic-suite": "__all__",
        "all": "__all__",
    }
    selected = alias.get(dataset_name, dataset_name)
    if selected == "__all__":
        return suite
    return [by_name.get(selected, suite[0])]


def _majority_mapping(y_true: np.ndarray, y_pred: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for pred_label in np.unique(y_pred):
        members = y_true[y_pred == pred_label]
        if members.size == 0:
            continue
        vals, counts = np.unique(members, return_counts=True)
        mapping[int(pred_label)] = int(vals[int(np.argmax(counts))])
    return mapping


def _cluster_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mapping = _majority_mapping(y_true, y_pred)
    mapped = np.array([mapping.get(int(lbl), -1) for lbl in y_pred], dtype=int)
    return float(np.mean(mapped == y_true))


def _macro_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mapping = _majority_mapping(y_true, y_pred)
    mapped = np.array([mapping.get(int(lbl), -1) for lbl in y_pred], dtype=int)
    recalls: list[float] = []
    for cls in np.unique(y_true):
        mask = y_true == cls
        recalls.append(float(np.mean(mapped[mask] == cls)))
    return float(np.mean(recalls))


def _safe_int_labels(values: np.ndarray) -> np.ndarray:
    labels = np.asarray(values, dtype=int)
    if labels.size == 0:
        return labels
    if np.any(labels < 0):
        labels = labels.copy()
        max_label = int(labels[labels >= 0].max()) if np.any(labels >= 0) else 0
        next_label = max_label + 1
        for ix in range(labels.shape[0]):
            if labels[ix] < 0:
                labels[ix] = next_label
                next_label += 1
    return labels


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, elapsed_seconds: float) -> dict[str, float]:
    purity = _cluster_purity(y_true, y_pred)
    recall = _macro_recall(y_true, y_pred)
    f1 = 0.0 if purity + recall == 0 else (2 * purity * recall) / (purity + recall)
    loss = max(0.0, 1.0 - f1)
    latency_ms = 1000.0 * elapsed_seconds / max(1, y_true.shape[0])
    return {
        "accuracy": float(round(purity, 4)),
        "macro_recall": float(round(recall, 4)),
        "f1": float(round(f1, 4)),
        "loss": float(round(loss, 4)),
        "latency_ms": float(round(latency_ms, 4)),
    }


def _run_cosmocluster(case: SyntheticCase, config: dict) -> tuple[np.ndarray, float, dict[str, float], dict[str, float]]:
    hp = config.get("hyperparameters", {})
    epochs = int(hp.get("epochs", 20))
    k = max(4, min(24, int(np.sqrt(max(16, int(hp.get("batch_size", 64)))) * 2)))
    params = HybridParams(
        k=k,
        expansion_steps=max(8, epochs // 2),
        gravity_steps=max(8, epochs // 2),
        expansion_step_size=0.03,
        gravity_step_size=0.035,
        random_state=int(config.get("seed", 42)),
    )
    model = CosmoClusterMinimal(params=params)
    start = time.perf_counter()
    y_pred = model.fit_predict(case.x)
    fit_seconds = time.perf_counter() - start
    metrics = _evaluate_predictions(case.y, _safe_int_labels(y_pred), fit_seconds)
    protocol = {
        "k": float(params.k),
        "expansion_steps": float(params.expansion_steps),
        "gravity_steps": float(params.gravity_steps),
        "expansion_step_size": float(params.expansion_step_size),
        "gravity_step_size": float(params.gravity_step_size),
        "random_state": float(params.random_state or 0),
    }
    return y_pred.astype(int), fit_seconds, metrics, protocol


def _run_umap_hdbscan(case: SyntheticCase, seed: int) -> tuple[np.ndarray, float, str]:
    if umap is None or HDBSCAN is None:
        missing = []
        if umap is None:
            missing.append("umap-learn")
        if HDBSCAN is None:
            missing.append("hdbscan")
        raise RuntimeError(f"missing dependencies: {', '.join(missing)}")
    start = time.perf_counter()
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.0,
        n_components=2,
        metric="euclidean",
        random_state=seed,
    )
    embedding = reducer.fit_transform(case.x)
    clusterer = HDBSCAN(min_cluster_size=15, min_samples=8, metric="euclidean")
    labels = clusterer.fit_predict(embedding)
    elapsed = time.perf_counter() - start
    protocol = "umap(n_neighbors=15,min_dist=0.0,n_components=2)->hdbscan(min_cluster_size=15,min_samples=8)"
    return _safe_int_labels(labels), elapsed, protocol


def _run_hdbscan_direct(case: SyntheticCase) -> tuple[np.ndarray, float, str]:
    if HDBSCAN is None:
        raise RuntimeError("missing dependency: hdbscan")
    start = time.perf_counter()
    clusterer = HDBSCAN(min_cluster_size=15, min_samples=8, metric="euclidean")
    labels = clusterer.fit_predict(case.x)
    elapsed = time.perf_counter() - start
    return _safe_int_labels(labels), elapsed, "hdbscan(min_cluster_size=15,min_samples=8,metric=euclidean)"


def _run_control_on(case: SyntheticCase) -> tuple[np.ndarray, float, str]:
    start = time.perf_counter()
    centroid = np.mean(case.x, axis=0)
    centered = case.x - centroid
    radius = np.linalg.norm(centered, axis=1)
    angle = np.arctan2(centered[:, 1], centered[:, 0])
    radial_bins = np.digitize(radius, bins=np.quantile(radius, [0.33, 0.66]))
    angular_bins = np.digitize(angle, bins=[-np.pi / 3, np.pi / 3])
    labels = (radial_bins * 3 + angular_bins).astype(int)
    elapsed = time.perf_counter() - start
    return labels, elapsed, "control_on(radial+angular quantization baseline)"


def _algorithm_registry(config: dict) -> dict[str, bool]:
    comparators = config.get("comparators", {})
    return {
        "umap_hdbscan": bool(comparators.get("umap_hdbscan", True)),
        "hdbscan": bool(comparators.get("hdbscan", True)),
        "control_on": bool(comparators.get("control_on", True)),
    }


def run_real_experiment(config: dict, run_dir: Path) -> RealRunResult:
    hp = config.get("hyperparameters", {})
    epochs = int(hp.get("epochs", 20))
    seed = int(config.get("seed", 42))
    cases = _choose_cases(config)
    comparators = _algorithm_registry(config)

    primary_case = cases[0]
    primary_y_pred = np.zeros(primary_case.y.shape[0], dtype=int)
    primary_fit_seconds = 0.0

    model_rows: list[dict[str, str | float]] = []
    comparator_rows: list[dict[str, str | float]] = []
    train_log_lines = [
        "[train] starting real synthetic clustering run",
        f"[train] dataset_selector={config.get('dataset_name', 'density_bridge')} cases={len(cases)} seed={seed}",
        (
            "[train] protocol=cosmocluster "
            f"epochs={epochs} batch_size={int(hp.get('batch_size', 64))} "
            f"learning_rate={float(hp.get('learning_rate', 0.001))}"
        ),
    ]
    eval_log_lines = ["[eval] evaluating clustering labels against synthetic truth"]
    comparator_log_lines = ["[comparators] evaluating control algorithms against synthetic truth"]

    for case in cases:
        y_pred, fit_seconds, metrics, protocol = _run_cosmocluster(case, config)
        model_rows.append(
            {
                "algorithm": "cosmocluster",
                "case": case.name,
                "samples": float(case.x.shape[0]),
                "accuracy": metrics["accuracy"],
                "macro_recall": metrics["macro_recall"],
                "f1": metrics["f1"],
                "loss": metrics["loss"],
                "latency_ms": metrics["latency_ms"],
                "fit_seconds": float(round(fit_seconds, 4)),
                "protocol": json.dumps(protocol, sort_keys=True),
                "notes": case.notes,
            }
        )
        train_log_lines.append(
            (
                f"[train] case={case.name} samples={case.x.shape[0]} "
                f"k={int(protocol['k'])} expansion_steps={int(protocol['expansion_steps'])} "
                f"gravity_steps={int(protocol['gravity_steps'])} fit_seconds={fit_seconds:.4f}"
            )
        )
        eval_log_lines.append(
            (
                f"[eval] algorithm=cosmocluster case={case.name} "
                f"purity={metrics['accuracy']:.4f} macro_recall={metrics['macro_recall']:.4f} "
                f"f1={metrics['f1']:.4f} loss={metrics['loss']:.4f} "
                f"latency_ms_per_sample={metrics['latency_ms']:.4f}"
            )
        )
        if case.name == primary_case.name:
            primary_y_pred = y_pred
            primary_fit_seconds = fit_seconds

        if comparators.get("umap_hdbscan", False):
            try:
                labels, elapsed, protocol_text = _run_umap_hdbscan(case, seed)
                comp_metrics = _evaluate_predictions(case.y, labels, elapsed)
                comparator_rows.append(
                    {
                        "algorithm": "umap_hdbscan",
                        "case": case.name,
                        "samples": float(case.x.shape[0]),
                        "accuracy": comp_metrics["accuracy"],
                        "macro_recall": comp_metrics["macro_recall"],
                        "f1": comp_metrics["f1"],
                        "loss": comp_metrics["loss"],
                        "latency_ms": comp_metrics["latency_ms"],
                        "fit_seconds": float(round(elapsed, 4)),
                        "protocol": protocol_text,
                        "notes": "",
                    }
                )
                comparator_log_lines.append(
                    f"[comparators] algorithm=umap_hdbscan case={case.name} status=ok protocol={protocol_text}"
                )
            except Exception as exc:
                comparator_log_lines.append(
                    f"[comparators] algorithm=umap_hdbscan case={case.name} status=skipped reason={exc}"
                )

        if comparators.get("hdbscan", False):
            try:
                labels, elapsed, protocol_text = _run_hdbscan_direct(case)
                comp_metrics = _evaluate_predictions(case.y, labels, elapsed)
                comparator_rows.append(
                    {
                        "algorithm": "hdbscan",
                        "case": case.name,
                        "samples": float(case.x.shape[0]),
                        "accuracy": comp_metrics["accuracy"],
                        "macro_recall": comp_metrics["macro_recall"],
                        "f1": comp_metrics["f1"],
                        "loss": comp_metrics["loss"],
                        "latency_ms": comp_metrics["latency_ms"],
                        "fit_seconds": float(round(elapsed, 4)),
                        "protocol": protocol_text,
                        "notes": "",
                    }
                )
                comparator_log_lines.append(
                    f"[comparators] algorithm=hdbscan case={case.name} status=ok protocol={protocol_text}"
                )
            except Exception as exc:
                comparator_log_lines.append(
                    f"[comparators] algorithm=hdbscan case={case.name} status=skipped reason={exc}"
                )

        if comparators.get("control_on", False):
            try:
                labels, elapsed, protocol_text = _run_control_on(case)
                comp_metrics = _evaluate_predictions(case.y, labels, elapsed)
                comparator_rows.append(
                    {
                        "algorithm": "control_on",
                        "case": case.name,
                        "samples": float(case.x.shape[0]),
                        "accuracy": comp_metrics["accuracy"],
                        "macro_recall": comp_metrics["macro_recall"],
                        "f1": comp_metrics["f1"],
                        "loss": comp_metrics["loss"],
                        "latency_ms": comp_metrics["latency_ms"],
                        "fit_seconds": float(round(elapsed, 4)),
                        "protocol": protocol_text,
                        "notes": "",
                    }
                )
                comparator_log_lines.append(
                    f"[comparators] algorithm=control_on case={case.name} status=ok protocol={protocol_text}"
                )
            except Exception as exc:
                comparator_log_lines.append(
                    f"[comparators] algorithm=control_on case={case.name} status=skipped reason={exc}"
                )

    train_log_lines.append("[train] done")
    eval_log_lines.append("[eval] done")
    comparator_log_lines.append("[comparators] done")

    with (run_dir / "logs/train.log").open("w", encoding="utf-8") as f:
        f.write("\n".join(train_log_lines) + "\n")
    with (run_dir / "logs/eval.log").open("w", encoding="utf-8") as f:
        f.write("\n".join(eval_log_lines) + "\n")
    with (run_dir / "logs/comparators.log").open("w", encoding="utf-8") as f:
        f.write("\n".join(comparator_log_lines) + "\n")

    case_metrics_path = run_dir / "case_metrics.csv"
    with case_metrics_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "algorithm",
            "case",
            "samples",
            "accuracy",
            "macro_recall",
            "f1",
            "loss",
            "latency_ms",
            "fit_seconds",
            "protocol",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(model_rows)

    comparator_metrics_path = run_dir / "comparator_metrics.csv"
    with comparator_metrics_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "algorithm",
            "case",
            "samples",
            "accuracy",
            "macro_recall",
            "f1",
            "loss",
            "latency_ms",
            "fit_seconds",
            "protocol",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparator_rows)

    (run_dir / "comparator_metrics.json").write_text(
        json.dumps({"rows": comparator_rows, "comparators": comparators}, indent=2),
        encoding="utf-8",
    )

    aggregated = {
        "accuracy": round(float(np.mean([float(r["accuracy"]) for r in model_rows])), 4),
        "f1": round(float(np.mean([float(r["f1"]) for r in model_rows])), 4),
        "loss": round(float(np.mean([float(r["loss"]) for r in model_rows])), 4),
        "latency_ms": round(float(np.mean([float(r["latency_ms"]) for r in model_rows])), 4),
    }

    case_label = primary_case.name if len(cases) == 1 else f"suite:{len(cases)}"
    return RealRunResult(
        metrics=aggregated,
        case_name=case_label,
        y_true=primary_case.y.copy(),
        y_pred=primary_y_pred.copy(),
        fit_seconds=float(np.sum([float(r["fit_seconds"]) for r in model_rows])),
        epochs=epochs,
    )
