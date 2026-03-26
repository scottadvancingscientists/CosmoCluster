from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cosmocluster import CosmoClusterMinimal, HybridParams
from synthetic_suite import SyntheticCase, create_hdbscan_challenge_suite


@dataclass(frozen=True)
class RealRunResult:
    metrics: dict[str, float]
    case_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    fit_seconds: float
    epochs: int


def _choose_case(config: dict) -> SyntheticCase:
    dataset_name = str(config.get("dataset_name", "")).lower()
    suite = create_hdbscan_challenge_suite(
        n_samples=max(600, int(config.get("hyperparameters", {}).get("batch_size", 64)) * 12),
        noise=0.03,
        random_state=int(config.get("seed", 42)),
    )
    by_name = {c.name: c for c in suite}
    alias = {
        "synthetic-moons": "variable_width_moons",
        "synthetic-suite": "density_bridge",
    }
    selected = alias.get(dataset_name, dataset_name)
    return by_name.get(selected, suite[0])


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


def run_real_experiment(config: dict, run_dir: Path) -> RealRunResult:
    hp = config.get("hyperparameters", {})
    epochs = int(hp.get("epochs", 20))
    k = max(4, min(24, int(np.sqrt(max(16, int(hp.get("batch_size", 64)))) * 2)))
    case = _choose_case(config)

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

    purity = _cluster_purity(case.y, y_pred)
    recall = _macro_recall(case.y, y_pred)
    f1 = 0.0 if purity + recall == 0 else (2 * purity * recall) / (purity + recall)
    loss = max(0.0, 1.0 - f1)
    latency_ms = 1000.0 * fit_seconds / max(1, case.x.shape[0])

    metrics = {
        "accuracy": round(purity, 4),
        "f1": round(f1, 4),
        "loss": round(loss, 4),
        "latency_ms": round(latency_ms, 4),
    }

    train_log = run_dir / "logs/train.log"
    eval_log = run_dir / "logs/eval.log"
    train_log.write_text(
        "\n".join(
            [
                "[train] starting real synthetic clustering run",
                f"[train] case={case.name}",
                f"[train] samples={case.x.shape[0]}",
                f"[train] expansion_steps={params.expansion_steps} gravity_steps={params.gravity_steps}",
                f"[train] fit_seconds={fit_seconds:.4f}",
                "[train] done",
                "",
            ]
        ),
        encoding="utf-8",
    )
    eval_log.write_text(
        "\n".join(
            [
                "[eval] evaluating clustering labels against synthetic truth",
                f"[eval] purity={metrics['accuracy']:.4f} macro_recall={recall:.4f} f1={metrics['f1']:.4f}",
                f"[eval] loss={metrics['loss']:.4f} latency_ms_per_sample={metrics['latency_ms']:.4f}",
                "[eval] done",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return RealRunResult(
        metrics=metrics,
        case_name=case.name,
        y_true=case.y.copy(),
        y_pred=y_pred.copy(),
        fit_seconds=fit_seconds,
        epochs=epochs,
    )
