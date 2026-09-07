import numpy as np

from src.evaluation import calculate_normalized_entropy, classification_metrics


def test_entropy_range_and_extremes():
    p = np.array([[1.0, 0.0], [0.5, 0.5], [0.9, 0.1]])
    e = calculate_normalized_entropy(p)
    assert np.all((0 <= e) & (e <= 1))
    assert e[0] < 1e-5
    assert abs(e[1] - 1.0) < 1e-5


def test_metrics():
    y_true = np.array([1, 1, 2, 2])
    y_pred = np.array([1, 2, 2, 2])
    m = classification_metrics(y_true, y_pred, [1, 2])
    assert 0 <= m["balanced_accuracy"] <= 1
    assert 0 <= m["macro_f1"] <= 1
    assert len(m["confusion_matrix"]) == 2
