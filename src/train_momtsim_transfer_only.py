
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_DIR = Path("eda_outputs")

TRAIN_FILE = DATA_DIR / "momtsim_v1_train.csv"
VALIDATION_FILE = DATA_DIR / "momtsim_v1_validation.csv"

FEATURES = [
    "amount",
    "step",
    "oldBalInitiator",
    "oldBalRecipient",
]

TARGET = "isFraud"

REVIEW_CAPACITIES = [0.5, 1, 2, 5, 10, 20]


# --------------------------------------------------
# 2. Load and filter TRANSFER transactions
# --------------------------------------------------

print("Loading training and validation data...")

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

train_transfer = train_df[
    train_df["transactionType"] == "TRANSFER"
].copy()

validation_transfer = validation_df[
    validation_df["transactionType"] == "TRANSFER"
].copy()

print("\nTRANSFER-only dataset sizes:")
print(f"Training:   {len(train_transfer):,}")
print(f"Validation: {len(validation_transfer):,}")

X_train = train_transfer[FEATURES]
y_train = train_transfer[TARGET].to_numpy()

X_validation = validation_transfer[FEATURES]
y_validation = validation_transfer[TARGET].to_numpy()

train_fraud_rate = y_train.mean()
validation_fraud_rate = y_validation.mean()

print(f"\nTraining fraud rate:   {train_fraud_rate:.2%}")
print(f"Validation fraud rate: {validation_fraud_rate:.2%}")


# --------------------------------------------------
# 3. Build numerical preprocessing and model
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
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
# 4. Train model
# --------------------------------------------------

print("\nTraining TRANSFER-only Logistic Regression...")

model.fit(X_train, y_train)

print("Training completed.")


# --------------------------------------------------
# 5. Evaluate validation ranking
# --------------------------------------------------

print("\nGenerating validation probabilities...")

probabilities = model.predict_proba(X_validation)[:, 1]

pr_auc = average_precision_score(
    y_validation,
    probabilities,
)

roc_auc = roc_auc_score(
    y_validation,
    probabilities,
)

print("\n" + "=" * 70)
print("TRANSFER-ONLY VALIDATION PERFORMANCE")
print("=" * 70)

print(f"PR-AUC:  {pr_auc:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")


# --------------------------------------------------
# 6. Risk score distribution
# --------------------------------------------------

print("\nTRANSFER-only risk score distribution:")

for quantile in [0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1]:

    value = np.quantile(probabilities, quantile)

    print(f"{quantile:>6.0%} quantile: {value:.6f}")


# --------------------------------------------------
# 7. Alert capacity analysis within TRANSFER
# --------------------------------------------------

total_transactions = len(y_validation)
fraud_count = int(y_validation.sum())

ranked_indices = np.argsort(
    -probabilities,
    kind="stable",
)

results = []

print("\n" + "=" * 90)
print("TRANSFER-ONLY ALERT CAPACITY ANALYSIS")
print("=" * 90)

for capacity in REVIEW_CAPACITIES:

    review_count = int(
        np.ceil(total_transactions * capacity / 100)
    )

    selected_indices = ranked_indices[:review_count]

    selected_labels = y_validation[selected_indices]

    true_positives = int(selected_labels.sum())
    false_positives = review_count - true_positives
    false_negatives = fraud_count - true_positives

    precision = true_positives / review_count

    recall = (
        true_positives / fraud_count
        if fraud_count else 0
    )

    lift = (
        precision / validation_fraud_rate
        if validation_fraud_rate else 0
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
# 8. Save results
# --------------------------------------------------

output_file = DATA_DIR / "momtsim_v1_transfer_only_results.csv"

results_df.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")

print("\nImportant:")
print("- Only TRANSFER transactions were used.")
print("- transactionType was excluded because it is constant.")
print("- Model fitted on training TRANSFER transactions only.")
print("- Evaluation used validation TRANSFER transactions only.")
print("- Test data remains untouched.")
