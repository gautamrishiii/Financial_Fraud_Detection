
from pathlib import Path

import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

print("Loading chronological splits...")

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

print(f"Training TRANSFER rows: {len(train_df):,}")
print(f"Validation TRANSFER rows: {len(validation_df):,}")
print(f"Training fraud rate: {y_train.mean():.2%}")
print(f"Validation fraud rate: {y_validation.mean():.2%}")


# --------------------------------------------------
# 3. Model 1 — Logistic Regression
# --------------------------------------------------

print("\nTraining Logistic Regression...")

logistic_model = Pipeline(
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

logistic_model.fit(X_train, y_train)

logistic_probabilities = logistic_model.predict_proba(
    X_validation
)[:, 1]


# --------------------------------------------------
# 4. Model 2 — Histogram Gradient Boosting
# --------------------------------------------------

print("\nTraining Histogram Gradient Boosting...")

boosting_model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.1,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                early_stopping=True,
                random_state=42,
            ),
        ),
    ]
)

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

boosting_model.fit(
    X_train,
    y_train,
    classifier__sample_weight=sample_weights,
)

boosting_probabilities = boosting_model.predict_proba(
    X_validation
)[:, 1]


# --------------------------------------------------
# 5. Compare validation ranking metrics
# --------------------------------------------------

results = []

for model_name, probabilities in [
    ("Logistic Regression", logistic_probabilities),
    ("Histogram Gradient Boosting", boosting_probabilities),
]:

    pr_auc = average_precision_score(
        y_validation,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_validation,
        probabilities,
    )

    results.append({
        "model": model_name,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    })

    print(f"\n{model_name}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")


# --------------------------------------------------
# 6. Print score distribution
# --------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

output_file = DATA_DIR / "momtsim_v1_transfer_model_comparison.csv"

results_df.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")

print("\nImportant:")
print("- Both models use the same three pre-transaction features.")
print("- Both models use TRANSFER-only data.")
print("- Training and validation remain chronological.")
print("- The test set remains untouched.")
print("- Validation performance is not a real-world performance estimate.")
