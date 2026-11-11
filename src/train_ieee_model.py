
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = Path("eda_outputs")
MODEL_DIR = Path("models/ieee_cis")

NUMERIC_FEATURES = [
    "TransactionAmt",
    "TransactionDT",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "dist1",
    *[f"C{i}" for i in range(1, 15)],
    "id_01",
    "id_02",
    "id_05",
    "id_06",
    "id_11",
    "id_13",
    "id_17",
    "id_19",
    "id_20",
    "has_identity",
]

CATEGORICAL_FEATURES = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
]


def load_split(name):
    return pd.read_csv(DATA_DIR / f"ieee_{name}.csv")


def build_pipeline():
    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        ),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="missing",
                keep_empty_features=True,
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=100,
            ),
        ),
    ])

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.00001,
        max_iter=30,
        tol=0.001,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifier),
    ])


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions
        ).tolist(),
    }


def main():
    print("IEEE-CIS — Logistic Regression Baseline")
    print("=" * 65)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")

    X_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train["isFraud"].astype(int)

    X_validation = validation[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ]
    y_validation = validation["isFraud"].astype(int)

    X_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test["isFraud"].astype(int)

    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Test rows: {len(test):,}")
    print("\nFitting preprocessing and classifier...")

    model = build_pipeline()
    model.fit(X_train, y_train)

    print("Training completed.")

    print("\nScoring validation data...")
    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_validation,
        validation_probabilities,
    )

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_values))
    best_threshold = float(thresholds[best_index])

    validation_results = evaluate(
        y_validation,
        validation_probabilities,
        best_threshold,
    )

    print(f"Selected validation threshold: {best_threshold:.6f}")
    print("Validation metrics:")
    print(json.dumps(validation_results, indent=2))

    print("\nScoring test data...")
    test_probabilities = model.predict_proba(X_test)[:, 1]

    test_results = evaluate(
        y_test,
        test_probabilities,
        best_threshold,
    )

    ranking = np.argsort(test_probabilities)[::-1]
    review_count = max(1, int(len(y_test) * 0.01))
    top_indices = ranking[:review_count]

    test_results["precision_at_1_percent"] = float(
        y_test.iloc[top_indices].mean()
    )
    test_results["recall_at_1_percent"] = float(
        y_test.iloc[top_indices].sum() / y_test.sum()
    )

    print("\nTest metrics:")
    print(json.dumps(test_results, indent=2))

    results = {
        "model": "SGDClassifier logistic regression",
        "threshold": best_threshold,
        "validation": validation_results,
        "test": test_results,
        "feature_policy": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "excluded": [
                "TransactionID",
                "isFraud from input",
                "unselected transaction fields",
            ],
        },
    }

    joblib.dump(
        model,
        MODEL_DIR / "logistic_regression.joblib",
    )

    with open(MODEL_DIR / "evaluation.json", "w") as file:
        json.dump(results, file, indent=2)

    print("\nSaved model and evaluation results.")
    print(f"Model directory: {MODEL_DIR}")


if __name__ == "__main__":
    main()