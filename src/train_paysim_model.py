
from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DATA_DIR = Path("models/paysim_preprocessed")
MODEL_DIR = Path("models/paysim")

BATCH_SIZE = 100_000
EPOCHS = 3
RANDOM_STATE = 42


def evaluate(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(np.uint8)

    precision, recall, f1, _ = (
        precision_score(y_true, predictions, zero_division=0),
        recall_score(y_true, predictions, zero_division=0),
        f1_score(y_true, predictions, zero_division=0),
        None,
    )

    capacity = max(1, int(np.ceil(len(y_true) * 0.01)))
    top_indices = np.argsort(scores)[-capacity:]

    return {
        "average_precision": float(
            average_precision_score(y_true, scores)
        ),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(
            y_true, predictions
        ).tolist(),
        "precision_at_1pct_capacity": float(
            y_true[top_indices].mean()
        ),
        "recall_at_1pct_capacity": float(
            y_true[top_indices].sum() / y_true.sum()
        ),
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("PaySim — Incremental Logistic Regression")
    print("=" * 65)

    X_train = np.load(
        DATA_DIR / "X_train.npy", mmap_mode="r"
    )
    y_train = np.load(
        DATA_DIR / "y_train.npy", mmap_mode="r"
    )

    X_val = np.load(
        DATA_DIR / "X_validation.npy", mmap_mode="r"
    )
    y_val = np.load(
        DATA_DIR / "y_validation.npy", mmap_mode="r"
    )

    X_test = np.load(
        DATA_DIR / "X_test.npy", mmap_mode="r"
    )
    y_test = np.load(
        DATA_DIR / "y_test.npy", mmap_mode="r"
    )

    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    total = negatives + positives

    class_weights = {
        0: total / (2 * negatives),
        1: total / (2 * positives),
    }

    print(f"Training rows: {len(y_train):,}")
    print(f"Legitimate: {negatives:,}")
    print(f"Fraud: {positives:,}")
    print(f"Class weights: {class_weights}")

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        learning_rate="optimal",
        random_state=RANDOM_STATE,
    )

    rng = np.random.default_rng(RANDOM_STATE)

    for epoch in range(EPOCHS):
        order = rng.permutation(len(y_train))

        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]

            X_batch = X_train[indices]
            y_batch = y_train[indices]

            sample_weights = np.where(
                y_batch == 1,
                class_weights[1],
                class_weights[0],
            )

            model.partial_fit(
                X_batch,
                y_batch,
                classes=np.array([0, 1]),
                sample_weight=sample_weights,
            )

        print(f"Completed epoch {epoch + 1}/{EPOCHS}")

    print("\nSelecting decision threshold on validation data...")

    validation_scores = model.decision_function(X_val)

    precision, recall, thresholds = precision_recall_curve(
        y_val,
        validation_scores,
    )

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_values))
    threshold = float(thresholds[best_index])

    validation_metrics = evaluate(
        y_val, validation_scores, threshold
    )

    print(f"Selected threshold: {threshold:.6f}")
    print("\nValidation metrics:")
    print(json.dumps(validation_metrics, indent=4))

    print("\nEvaluating untouched test partition...")

    test_scores = model.decision_function(X_test)

    test_metrics = evaluate(
        y_test, test_scores, threshold
    )

    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=4))

    joblib.dump(model, MODEL_DIR / "logistic_regression.joblib")

    results = {
        "model": "SGD Logistic Regression",
        "epochs": EPOCHS,
        "threshold": threshold,
        "validation": validation_metrics,
        "test": test_metrics,
        "features": [
            "log_amount",
            "log_step",
            "log_oldbalanceOrg",
            "type_CASH_IN",
            "type_CASH_OUT",
            "type_DEBIT",
            "type_PAYMENT",
            "type_TRANSFER",
        ],
    }

    with open(MODEL_DIR / "evaluation.json", "w") as file:
        json.dump(results, file, indent=4)

    print("\nModel and evaluation results saved.")
    print(f"Model: {MODEL_DIR / 'logistic_regression.joblib'}")
    print(f"Results: {MODEL_DIR / 'evaluation.json'}")


if __name__ == "__main__":
    main()