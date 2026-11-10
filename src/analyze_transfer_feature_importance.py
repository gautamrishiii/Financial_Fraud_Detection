
from pathlib import Path

import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_DIR = Path("eda_outputs")

TRAIN_FILE = DATA_DIR / "momtsim_v1_train.csv"
VALIDATION_FILE = DATA_DIR / "momtsim_v1_validation.csv"

FEATURES = [
    "amount",
    "oldBalInitiator",
    "oldBalRecipient",
]

TARGET = "isFraud"


# --------------------------------------------------
# 2. Load TRANSFER-only data
# --------------------------------------------------

print("Loading chronological datasets...")

columns = FEATURES + [TARGET, "transactionType"]

train_df = pd.read_csv(TRAIN_FILE, usecols=columns)
validation_df = pd.read_csv(VALIDATION_FILE, usecols=columns)

train_df = train_df[
    train_df["transactionType"] == "TRANSFER"
].copy()

validation_df = validation_df[
    validation_df["transactionType"] == "TRANSFER"
].copy()

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_validation = validation_df[FEATURES]
y_validation = validation_df[TARGET]

print(f"Training rows: {len(train_df):,}")
print(f"Validation rows: {len(validation_df):,}")


# --------------------------------------------------
# 3. Train Gradient Boosting
# --------------------------------------------------

print("\nTraining Gradient Boosting model...")

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
# 4. Calculate baseline validation PR-AUC
# --------------------------------------------------

from sklearn.metrics import average_precision_score

baseline_scores = model.predict_proba(X_validation)[:, 1]

baseline_pr_auc = average_precision_score(
    y_validation,
    baseline_scores,
)

print(f"\nBaseline validation PR-AUC: {baseline_pr_auc:.4f}")


# --------------------------------------------------
# 5. Calculate permutation importance
# --------------------------------------------------

print("\nCalculating permutation importance...")
print("This may take a little time.")

importance = permutation_importance(
    estimator=model,
    X=X_validation,
    y=y_validation,
    scoring="average_precision",
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
)

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance_mean": importance.importances_mean,
    "importance_std": importance.importances_std,
})

importance_df = importance_df.sort_values(
    "importance_mean",
    ascending=False,
)


# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print("\n" + "=" * 75)
print("TRANSFER-ONLY PERMUTATION FEATURE IMPORTANCE")
print("=" * 75)

print(
    importance_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

output_file = DATA_DIR / "momtsim_v1_transfer_feature_importance.csv"

importance_df.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")

print("\nInterpretation:")
print("- Larger positive importance means shuffling hurt validation PR-AUC.")
print("- Near-zero importance means little measured contribution.")
print("- Negative importance can occur due to sampling variability.")
print("- Correlated features can share or mask each other's importance.")
print("- Importance is not proof of causation or real-world usefulness.")
print("- The test set remains untouched.")
