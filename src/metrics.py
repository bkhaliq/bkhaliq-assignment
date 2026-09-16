"""Metric computation from saved prediction tables (independent of the model
that produced them, so results can be re-verified)."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


def compute_metrics(df, actual_col="actual", pred_col="predicted", labels=None):
    """Compute accuracy, per-class counts, confusion matrix and per-class
    recall/accuracy from a DataFrame of actual/predicted labels."""
    df = df.copy()
    if labels is None:
        labels = sorted(set(df[actual_col].tolist()) | set(df[pred_col].tolist()))
    y_act = df[actual_col].tolist()
    y_pre = df[pred_col].tolist()

    correct = sum(a == p for a, p in zip(y_act, y_pre))
    total = len(df)
    accuracy = correct / total if total else 0.0

    cm = confusion_matrix(y_act, y_pre, labels=labels).tolist()

    actual_counts = df[actual_col].value_counts().reindex(labels, fill_value=0)
    predicted_counts = df[pred_col].value_counts().reindex(labels, fill_value=0)

    per_class = {}
    for i, lab in enumerate(labels):
        row = cm[i]
        tp = row[i]
        n_act = int(actual_counts.get(lab, 0))
        # recall = correctly predicted / actual in class
        recall = tp / n_act if n_act else None
        # precision = correctly predicted / predicted in class
        n_pred = int(predicted_counts.get(lab, 0))
        precision = tp / n_pred if n_pred else None
        # predicted-as-confusion direction: how many of class X predicted as Y
        # misclassification counts
        per_class[lab] = {
            "actual": n_act,
            "predicted": n_pred,
            "correct": int(tp),
            "misclassified": n_act - int(tp),
            "recall": round(recall, 4) if recall is not None else None,
            "precision": round(precision, 4) if precision is not None else None,
        }

    # misclassification direction matrix: {actual: {predicted: count}}
    misdir = {}
    for i, a in enumerate(labels):
        for j, p in enumerate(labels):
            if i != j and cm[i][j] > 0:
                misdir.setdefault(a, {})[p] = int(cm[i][j])

    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": round(accuracy, 4),
        "labels": labels,
        "confusion_matrix": cm,
        "confusion_matrix_labels": labels,
        "actual_counts": actual_counts.to_dict(),
        "predicted_counts": predicted_counts.to_dict(),
        "per_class": per_class,
        "misclassification_direction": misdir,
    }


def misclassified_rows(df, actual_col="actual", pred_col="predicted"):
    m = df[df[actual_col] != df[pred_col]].copy()
    return m.sort_values(actual_col)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"wrote {path}")
