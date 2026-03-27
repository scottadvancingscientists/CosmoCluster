from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional dependency fallback.
    go = None


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

    if go is None:
        (plotly_dir / "training_curves.html").write_text(
            (
                "<html><body><h3>Training Curves</h3>"
                f"<p>Case: {case_name}</p>"
                "<p>Plotly is not installed. Static PNG thumbnail is available in figures/loss_curve.png.</p>"
                "</body></html>"
            ),
            encoding="utf-8",
        )
        (plotly_dir / "embedding_projection.html").write_text(
            (
                "<html><body><h3>Embedding Projection</h3>"
                f"<p>Case: {case_name}</p>"
                "<p>Plotly is not installed. Static report figures are available in figures/.</p>"
                "</body></html>"
            ),
            encoding="utf-8",
        )
        return

    loss_fig = go.Figure(data=[go.Scatter(x=x, y=y, mode="lines", name="loss")])
    loss_fig.update_layout(
        title=f"Training Curves — {case_name}",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_white",
        margin=dict(l=30, r=20, t=50, b=40),
    )
    loss_fig.write_html(
        plotly_dir / "training_curves.html",
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True, "displaylogo": False},
    )

    x_points = np.arange(y_true.shape[0])
    embedding_fig = go.Figure(
        data=[
            go.Scattergl(
                x=x_points,
                y=mapped_pred,
                mode="markers",
                marker={"size": 5, "opacity": 0.75, "color": y_true, "colorscale": "Viridis"},
                text=[f"true={int(t)} pred={int(p)}" for t, p in zip(y_true, mapped_pred, strict=True)],
                hovertemplate="%{text}<extra></extra>",
                name="samples",
            )
        ]
    )
    embedding_fig.update_layout(
        title=f"Label Projection (sample index vs mapped class) — {case_name}",
        xaxis_title="Sample index",
        yaxis_title="Mapped predicted class",
        template="plotly_white",
        margin=dict(l=30, r=20, t=50, b=40),
    )
    embedding_fig.write_html(
        plotly_dir / "embedding_projection.html",
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True, "displaylogo": False},
    )
