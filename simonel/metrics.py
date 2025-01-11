import numpy as np
from collections import Counter

def accuracy_score_custom(y_true, y_pred):
    """Calculate accuracy."""
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    return correct / len(y_true)

def classification_report_custom(y_true, y_pred):
    """Generate classification report."""
    tp = sum(yt == yp == 1 for yt, yp in zip(y_true, y_pred))
    tn = sum(yt == yp == 0 for yt, yp in zip(y_true, y_pred))
    fp = sum(yt == 0 and yp == 1 for yt, yp in zip(y_true, y_pred))
    fn = sum(yt == 1 and yp == 0 for yt, yp in zip(y_true, y_pred))

    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0

    precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else 0

    report = {
        "Negative": {"precision": precision_neg, "recall": recall_neg, "f1-score": f1_neg},
        "Positive": {"precision": precision_pos, "recall": recall_pos, "f1-score": f1_pos},
    }
    return report

def confusion_matrix_custom(y_true, y_pred):
    """Generate confusion matrix."""
    tp = sum(yt == yp == 1 for yt, yp in zip(y_true, y_pred))
    tn = sum(yt == yp == 0 for yt, yp in zip(y_true, y_pred))
    fp = sum(yt == 0 and yp == 1 for yt, yp in zip(y_true, y_pred))
    fn = sum(yt == 1 and yp == 0 for yt, yp in zip(y_true, y_pred))
    return np.array([[tn, fp], [fn, tp]])