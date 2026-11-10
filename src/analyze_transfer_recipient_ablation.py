
from pathlib import Path

import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight


DATA_DIR = Path("eda_outputs")

TRAIN_FILE = DATA_DIR / "momtsim_v1_train.csv"
VALIDATION_FILE = DATA_DIR / "momtsim_v1_validation.csv"

TARGET = "isFraud"

FEATURE_GROUPS = {
    "A_full_features": [
        "amount",
        "oldBalInitiator",
        "oldBalRecipient",
    ],
    "B_without_recipient_balance": [
        "amount",
        "oldBalInitiator",
    ],
}


# --------------------------------------------------
# 1. Load TRANSFER-only datasets
# --------------------------------------------------

print("Loading chronological datasets...")

columns = list(dict.fromkeys(
    FEATURE_GROUPS["A_full_features"] + [TARGET, "transactionType"]
))

train_df = pd.read_csv(TRAIN_FILE, usecols=columns)
validation_df = pd.read_csv(VALIDATION_FILE, usecols=columns)

train_df = train_df[
    train_df["transactionType"] == "TRANSFER"
].copy()

validation_df = validation_df[
    validation_df["transactionType"] == "TRANSFER"
].copy()

y_train = train_df[TARGET]
y_validation = validation_df[TARGET]

print(f"Training TRANSFER rows: {len(train_df):,}")
print(f"Validation TRANSFER rows: {len(validation_df):,}")
print(f"Validation fraud rate: {y_validation.mean():.4%}")


# --------------------------------------------------
# 2. Train and evaluate both feature groups
# --------------------------------------------------

results = []

for group_name, features in FEATURE_GROUPS.items():

    print(f"\nTraining {group_name}...")
    print(f"Features: {features}")

    X_train = train_df[features]
    X_validation = validation_df[features]

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

    scores = model.predict_proba(X_validation)[:, 1]

    pr_auc = average_precision_score(y_validation, scores)
    roc_auc = roc_auc_score(y_validation, scores)

    results.append({
        "model": group_name,
        "feature_count": len(features),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    })

    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")


# --------------------------------------------------
# 3. Compare results
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 75)
print("RECIPIENT BALANCE FEATURE ABLATION")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

pr_change = (
    results_df.loc[1, "pr_auc"]
    - results_df.loc[0, "pr_auc"]
)

roc_change = (
    results_df.loc[1, "roc_auc"]
    - results_df.loc[0, "roc_auc"]
)

print("\nChange after removing oldBalRecipient:")
print(f"PR-AUC change:  {pr_change:+.6f}")
print(f"ROC-AUC change: {roc_change:+.6f}")

output_file = DATA_DIR / "momtsim_v1_transfer_recipient_ablation.csv"

results_df.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")
print("\nThe test set remains untouched.")
