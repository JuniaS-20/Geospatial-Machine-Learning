"""Classification metrics, reports, and figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def calculate_normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes).")
    if p.shape[1] <= 1:
        return np.zeros(p.shape[0], dtype=np.float32)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    entropy = -np.sum(p * np.log(p), axis=1) / np.log(p.shape[1])
    return np.clip(entropy, 0.0, 1.0).astype(np.float32)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] | np.ndarray) -> dict:
    labels = np.asarray(labels)
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": [int(v) for v in labels],
    }


def save_confusion_matrix_figure(
    metrics: dict,
    output_path: Path,
    normalized: bool = False,
    class_names: dict[int, str] | None = None,
) -> None:
    """
    Save a confusion matrix figure.

    Numeric class IDs are kept internally for computation, while
    human-readable class names can be used only for figure display.

    Parameters
    ----------
    metrics : dict
        Evaluation metrics containing:
        - "confusion_matrix"
        - "labels"

    output_path : Path
        Path where the PNG figure will be written.

    normalized : bool, default=False
        If True, normalize each row by the number of observations
        belonging to the corresponding true class.

    class_names : dict[int, str] | None
        Optional mapping from numeric class IDs to human-readable names.

        Example:
        {
            4: "Batis",
            5: "Eau",
            7: "sol_nu",
            8: "Veg_dense",
        }
    """

    cm = np.asarray(
        metrics["confusion_matrix"],
        dtype=float,
    )

    if normalized:
        sums = cm.sum(
            axis=1,
            keepdims=True,
        )

        cm = np.divide(
            cm,
            sums,
            out=np.zeros_like(cm),
            where=sums != 0,
        )

    # Numeric IDs used internally by the model/evaluation.
    labels = metrics["labels"]

    # Human-readable names used only for figure display.
    if class_names is None:
        display_labels = [
            str(label)
            for label in labels
        ]
    else:
        display_labels = [
            class_names.get(
                int(label),
                str(label),
            )
            for label in labels
        ]

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    image = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues",
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=display_labels,
        yticklabels=display_labels,
        xlabel="Predicted class",
        ylabel="True class",
        title=(
            "Normalized confusion matrix"
            if normalized
            else "Confusion matrix"
        ),
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
    )

    threshold = (
        cm.max() / 2
        if cm.size
        else 0
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            text = (
                f"{cm[i, j]:.2f}"
                if normalized
                else str(int(cm[i, j]))
            )

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                ),
            )

    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)
def save_per_class_table(
    metrics: dict,
    class_names: dict[int, str],
    output_path: Path,
) -> None:
    """
    Save per-class classification metrics as a CSV table.

    Numeric class IDs are preserved for reproducibility, while
    human-readable class names are added for interpretation.

    Parameters
    ----------
    metrics : dict
        Evaluation metrics returned by classification_metrics().

    class_names : dict[int, str]
        Mapping between numeric class IDs and human-readable names.

        Example:
        {
            4: "Batis",
            5: "Eau",
            7: "sol_nu",
            8: "Veg_dense",
        }

    output_path : Path
        Destination CSV file.
    """

    rows = []

    report = metrics["classification_report"]

    for label in metrics["labels"]:

        values = report.get(
            str(label),
            {}
        )

        rows.append({
            "label": int(label),
            "class_name": class_names.get(
                int(label),
                str(label),
            ),
            "precision": values.get(
                "precision",
                np.nan,
            ),
            "recall": values.get(
                "recall",
                np.nan,
            ),
            "f1_score": values.get(
                "f1-score",
                np.nan,
            ),
            "support": values.get(
                "support",
                0,
            ),
        })

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )