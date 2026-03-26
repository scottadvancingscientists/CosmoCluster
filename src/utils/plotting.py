from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def make_dummy_plots(figures_dir: Path, plotly_dir: Path) -> None:
    x = list(range(1, 21))
    y = [1 / (i**0.5) for i in x]

    plt.figure(figsize=(6, 4))
    plt.plot(x, y)
    plt.title("Dummy Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(figures_dir / "loss_curve.png", dpi=140)
    plt.close()

    plt.figure(figsize=(4, 4))
    plt.imshow([[8, 1], [2, 9]], cmap="Blues")
    plt.title("Dummy Confusion Matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=140)
    plt.close()

    (plotly_dir / "training_curves.html").write_text(
        "<html><body><h3>Dummy Plotly Training Curves</h3><p>Phase 1 placeholder.</p></body></html>",
        encoding="utf-8",
    )
    (plotly_dir / "embedding_projection.html").write_text(
        "<html><body><h3>Dummy Plotly Embedding Projection</h3><p>Phase 1 placeholder.</p></body></html>",
        encoding="utf-8",
    )
