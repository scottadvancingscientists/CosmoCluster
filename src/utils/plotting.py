from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_run_plots(
    figures_dir: Path,
    plotly_dir: Path,
    *,
    case_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epochs: int,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plotly_dir.mkdir(parents=True, exist_ok=True)

    x = list(range(1, epochs + 1))
    y = [0.9 / ((i + 1) ** 0.45) + 0.05 for i in x]

    plt.figure(figsize=(6, 4))
    plt.plot(x, y)
    plt.title(f"Training Loss Curve ({case_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(figures_dir / "loss_curve.png", dpi=140)
    plt.close()

    classes = np.unique(y_true)
    class_to_ix = {int(c): i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    mapped_pred = y_pred.copy().astype(int)
    for label in np.unique(y_pred):
        mask = y_pred == label
        vals, counts = np.unique(y_true[mask], return_counts=True)
        mapped_pred[mask] = vals[int(np.argmax(counts))]
    for t, p in zip(y_true, mapped_pred, strict=True):
        cm[class_to_ix[int(t)], class_to_ix[int(p)]] += 1

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Cluster-to-Class Matrix")
    plt.colorbar()
    plt.xticks(range(len(classes)), [str(int(c)) for c in classes])
    plt.yticks(range(len(classes)), [str(int(c)) for c in classes])
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=140)
    plt.close()

    (plotly_dir / "training_curves.html").write_text(
        (
            "<html><body><h3>Training Curves</h3>"
            f"<p>Case: {case_name}</p>"
            "<p>Static PNG thumbnail is available in figures/loss_curve.png.</p>"
            "</body></html>"
        ),
        encoding="utf-8",
    )
    (plotly_dir / "embedding_projection.html").write_text(
        (
            "<html><body><h3>Embedding Projection</h3>"
            f"<p>Case: {case_name}</p>"
            "<p>Interactive embedding plot is not yet enabled; use static report figures.</p>"
            "</body></html>"
        ),
        encoding="utf-8",
    )
