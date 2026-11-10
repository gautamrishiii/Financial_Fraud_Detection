
from pathlib import Path

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_DIR = Path("eda_outputs")

TRAIN_FILE = DATA_DIR / "momtsim_v1_train.csv"
VALIDATION_FILE = DATA_DIR / "momtsim_v1_validation.csv"

TARGET = "isFraud"

FEATURE_GROUPS = {
    "A - Amount": [
        "amount",
    ],
    "B - Amount + step": [
        "amount",
        "step",
    ],
    "C - Amount + balances": [
        "amount",
        "oldBalInitiator",
        "oldBalRecipient",
    ],
    "D - All numerical features": [
        "amount",
        "step",
        "oldBalInitiator",
        "oldBalRecipient",
    ],
}


# --------------------------------------------------
# 2. Load and filter TRANSFER data
# --------------------------------------------------

print("Loading chronological training and validation data...")

required_features = sorted({
    feature
    for features in FEATURE_GROUPS.values()
    for feature in features
})

required_columns = required_features + [TARGET, "transactionType"]

train_df = pd.read_csv(
    TRAIN_FILE,
    usecols=required_columns,
)

validation_df = pd.read_csv(
    VALIDATION_FILE,
    usecols=required_columns,
)

train_df = train_df[
    train_df["transactionType"] == "TRANSFER"
].copy()

validation_df = validation_df[
    validation_df["transactionType"] == "TRANSFER"
].copy()

X_train_all = train_df[required_features]
y_train = train_df[TARGET]

X_validation_all = validation_df[required_features]
y_validation = validation_df[TARGET]

print(f"Training TRANSFER rows: {len(train_df):,}")
print(f"Validation TRANSFER rows: {len(validation_df):,}")

print(f"Training fraud rate: {y_train.mean():.2%}")
print(f"Validation fraud rate: {y_validation.mean():.2%}")


# --------------------------------------------------
# 3. Dummy baseline
# --------------------------------------------------

print("\n" + "=" * 65)
print("DUMMY BASELINE")
print("=" * 65)

dummy = DummyClassifier(strategy="prior")

dummy.fit(
    X_train_all[["amount"]],
    y_train,
)

dummy_probabilities = dummy.predict_proba(
    X_validation_all[["amount"]]
)[:, 1]

dummy_pr_auc = average_precision_score(
    y_validation,
    dummy_probabilities,
)

dummy_roc_auc = roc_auc_score(
    y_validation,
    dummy_probabilities,
)

print(f"Dummy PR-AUC:  {dummy_pr_auc:.4f}")
print(f"Dummy ROC-AUC: {dummy_roc_auc:.4f}")


# --------------------------------------------------
# 4. Run feature ablation experiments
# --------------------------------------------------

results = []

for experiment_name, features in FEATURE_GROUPS.items():

    print("\n" + "=" * 65)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 65)

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

    print(f"Features: {features}")
    print("Training model...")

    model.fit(
        X_train_all[features],
        y_train,
    )

    probabilities = model.predict_proba(
        X_validation_all[features]
    )[:, 1]

    pr_auc = average_precision_score(
        y_validation,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_validation,
        probabilities,
    )

    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    results.append({
        "experiment": experiment_name,
        "features": ", ".join(features),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "pr_auc_above_dummy": pr_auc - dummy_pr_auc,
    })


# --------------------------------------------------
# 5. Compare results
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 75)
print("TRANSFER-ONLY FEATURE ABLATION COMPARISON")
print("=" * 75)

print(f"Dummy PR-AUC reference: {dummy_pr_auc:.4f}")

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

output_file = DATA_DIR / "momtsim_v1_transfer_feature_ablation.csv"

results_df.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")

print("\nImportant:")
print("- All experiments use TRANSFER transactions only.")
print("- All models use chronological training and validation splits.")
print("- No post-transaction features are included.")
print("- Validation results are for feature comparison, not final testing.")
print("- The test set remains untouched.")
