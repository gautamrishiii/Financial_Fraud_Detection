
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_DIR = Path("eda_outputs")

TRAIN_FILE = DATA_DIR / "momtsim_v1_train.csv"
VALIDATION_FILE = DATA_DIR / "momtsim_v1_validation.csv"

NUMERIC_FEATURES = [
    "amount",
    "step",
    "oldBalInitiator",
    "oldBalRecipient",
]

CATEGORICAL_FEATURES = ["transactionType"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "isFraud"


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

print("Loading training and validation data...")

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_validation = validation_df[FEATURES]
y_validation = validation_df[TARGET]

print(f"Training rows: {len(X_train):,}")
print(f"Validation rows: {len(X_validation):,}")


# --------------------------------------------------
# 3. Build preprocessing and model pipeline
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=300,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)


# --------------------------------------------------
# 4. Train using training data only
# --------------------------------------------------

print("\nTraining experiment D...")

model.fit(X_train, y_train)

print("Training completed.")


# --------------------------------------------------
# 5. Get validation probabilities
# --------------------------------------------------

print("\nGenerating validation probabilities...")

probabilities = model.predict_proba(X_validation)[:, 1]


# --------------------------------------------------
# 6. Evaluate multiple thresholds
# --------------------------------------------------

thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99,
]

results = []

print("\n" + "=" * 75)
print("VALIDATION THRESHOLD ANALYSIS")
print("=" * 75)

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1],
    ).ravel()

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    })


results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


# --------------------------------------------------
# 7. Find the best F1 threshold
# --------------------------------------------------

best_row = results_df.loc[
    results_df["f1"].idxmax()
]

print("\nBest F1 among tested thresholds:")
print(f"Threshold: {best_row['threshold']:.2f}")
print(f"Precision: {best_row['precision']:.4f}")
print(f"Recall:    {best_row['recall']:.4f}")
print(f"F1-score:  {best_row['f1']:.4f}")


# --------------------------------------------------
# 8. Save validation results
# --------------------------------------------------

output_file = DATA_DIR / "momtsim_v1_threshold_results.csv"

results_df.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")

print("\nImportant:")
print("- Only validation data was used for threshold analysis.")
print("- No final threshold has been selected.")
print("- The test set remains untouched.")
print("- The best F1 threshold is not automatically the best business threshold.")
