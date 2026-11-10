
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_DIR = Path("eda_outputs")

TRAIN_FILE = DATA_DIR / "momtsim_v1_train.csv"
TEST_FILE = DATA_DIR / "momtsim_v1_test.csv"

FEATURES = [
    "amount",
    "oldBalInitiator",
    "oldBalRecipient",
]

TARGET = "isFraud"

REVIEW_CAPACITIES = [0.01, 0.05, 0.10]


# --------------------------------------------------
# 2. Load chronological datasets
# --------------------------------------------------

print("Loading training and final test datasets...")

columns = FEATURES + [TARGET, "transactionType"]

train_df = pd.read_csv(TRAIN_FILE, usecols=columns)
test_df = pd.read_csv(TEST_FILE, usecols=columns)

train_df = train_df[
    train_df["transactionType"] == "TRANSFER"
].copy()

test_df = test_df[
    test_df["transactionType"] == "TRANSFER"
].copy()

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

print(f"Training TRANSFER rows: {len(train_df):,}")
print(f"Test TRANSFER rows: {len(test_df):,}")
print(f"Training fraud rate: {y_train.mean():.4%}")
print(f"Test fraud rate: {y_test.mean():.4%}")


# --------------------------------------------------
# 3. Train the frozen candidate model
# --------------------------------------------------

print("\nTraining frozen Gradient Boosting candidate...")

model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.1,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=42,
            ),
        ),
    ]
)

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

model.fit(
    X_train,
    y_train,
    classifier__sample_weight=sample_weights,
)


# --------------------------------------------------
# 4. Evaluate on untouched test data
# --------------------------------------------------

print("\nGenerating final test predictions...")

test_scores = model.predict_proba(X_test)[:, 1]

test_pr_auc = average_precision_score(y_test, test_scores)
test_roc_auc = roc_auc_score(y_test, test_scores)

print("\n" + "=" * 75)
print("FINAL CHRONOLOGICAL TEST EVALUATION")
print("=" * 75)

print(f"Test transactions: {len(y_test):,}")
print(f"Test fraud cases: {int(y_test.sum()):,}")
print(f"Test fraud rate: {y_test.mean():.4%}")
print(f"Dummy PR-AUC baseline: {y_test.mean():.4f}")
print(f"Model PR-AUC: {test_pr_auc:.4f}")
print(f"Model ROC-AUC: {test_roc_auc:.4f}")


# --------------------------------------------------
# 5. Review-capacity evaluation
# --------------------------------------------------

print("\n" + "=" * 75)
print("TEST ALERT-CAPACITY ANALYSIS")
print("=" * 75)

number_of_test_rows = len(y_test)

capacity_results = []

for capacity in REVIEW_CAPACITIES:

    review_count = int(
        np.ceil(number_of_test_rows * capacity)
    )

    selected_indices = np.argsort(-test_scores)[:review_count]

    selected_labels = y_test.iloc[selected_indices]

    true_positives = int(selected_labels.sum())
    false_positives = review_count - true_positives
    false_negatives = int(y_test.sum()) - true_positives

    precision = true_positives / review_count

    recall = (
        true_positives / y_test.sum()
        if y_test.sum() > 0
        else 0.0
    )

    lift = (
        precision / y_test.mean()
        if y_test.mean() > 0
        else 0.0
    )

    capacity_results.append({
        "review_capacity": capacity,
        "transactions_reviewed": review_count,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "lift": lift,
    })

capacity_df = pd.DataFrame(capacity_results)

print(
    capacity_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


# --------------------------------------------------
# 6. Save final evaluation results
# --------------------------------------------------

metrics_df = pd.DataFrame([
    {
        "model": "HistGradientBoosting_TRANSFER",
        "features": ", ".join(FEATURES),
        "test_transactions": len(y_test),
        "test_fraud_rate": y_test.mean(),
        "dummy_pr_auc": y_test.mean(),
        "pr_auc": test_pr_auc,
        "roc_auc": test_roc_auc,
    }
])

metrics_file = DATA_DIR / "momtsim_v1_transfer_final_metrics.csv"

capacity_file = DATA_DIR / "momtsim_v1_transfer_final_capacity.csv"

metrics_df.to_csv(metrics_file, index=False)
capacity_df.to_csv(capacity_file, index=False)

print("\nResults saved:")
print(metrics_file)
print(capacity_file)

print("\nIMPORTANT:")
print("- The model was trained only on the training period.")
print("- The final test period was not used for model selection.")
print("- No post-transaction features or raw account IDs were used.")
print("- Balance availability before authorization remains an assumption.")
print("- Results apply only to TRANSFER transactions in this simulator.")
