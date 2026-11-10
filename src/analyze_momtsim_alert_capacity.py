
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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

REVIEW_CAPACITIES = [0.5, 1, 2, 5, 10, 20]


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

print("Loading training and validation data...")

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

X_train = train_df[FEATURES]
y_train = train_df[TARGET].to_numpy()

X_validation = validation_df[FEATURES]
y_validation = validation_df[TARGET].to_numpy()

print(f"Training transactions: {len(X_train):,}")
print(f"Validation transactions: {len(X_validation):,}")

fraud_count = int(y_validation.sum())
fraud_rate = y_validation.mean()

print(f"Validation fraud count: {fraud_count:,}")
print(f"Validation fraud rate: {fraud_rate:.2%}")


# --------------------------------------------------
# 3. Build the experiment D pipeline
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
# 4. Train and generate validation scores
# --------------------------------------------------

print("\nTraining experiment D...")

model.fit(X_train, y_train)

print("Generating validation risk scores...")

probabilities = model.predict_proba(X_validation)[:, 1]


# --------------------------------------------------
# 5. Examine score distribution
# --------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION RISK SCORE DISTRIBUTION")
print("=" * 70)

quantiles = [0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]

for q, value in zip(
    quantiles,
    np.quantile(probabilities, quantiles),
):
    print(f"{q:>6.0%} quantile: {value:.6f}")

print(f"\nUnique score values: {len(np.unique(probabilities)):,}")


# --------------------------------------------------
# 6. Rank transactions by predicted risk
# --------------------------------------------------

print("\n" + "=" * 85)
print("ALERT CAPACITY ANALYSIS")
print("=" * 85)

# Highest predicted risk first.
ranked_indices = np.argsort(
    -probabilities,
    kind="stable",
)

results = []

total_transactions = len(y_validation)

for capacity in REVIEW_CAPACITIES:

    review_count = int(
        np.ceil(total_transactions * capacity / 100)
    )

    selected_indices = ranked_indices[:review_count]

    selected_labels = y_validation[selected_indices]

    true_positives = int(selected_labels.sum())
    false_positives = review_count - true_positives
    false_negatives = fraud_count - true_positives

    precision = (
        true_positives / review_count
        if review_count else 0
    )

    recall = (
        true_positives / fraud_count
        if fraud_count else 0
    )

    lift = (
        precision / fraud_rate
        if fraud_rate else 0
    )

    cutoff_score = probabilities[
        ranked_indices[review_count - 1]
    ]

    results.append({
        "review_capacity_percent": capacity,
        "transactions_reviewed": review_count,
        "cutoff_score": cutoff_score,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "lift_over_prevalence": lift,
    })


results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


# --------------------------------------------------
# 7. Save results
# --------------------------------------------------

output_file = DATA_DIR / "momtsim_v1_alert_capacity_results.csv"

results_df.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")

print("\nImportant limitations:")
print("- Ranking uses validation predictions only.")
print("- The model was fitted on training data only.")
print("- Exact review counts are simulated by ranking transactions.")
print("- Tied risk scores at the cutoff may require a tie-breaking rule.")
print("- This analysis does not establish real-world investigation capacity.")
print("- The test set remains untouched.")
