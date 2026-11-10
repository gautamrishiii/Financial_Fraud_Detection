
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

# Six consecutive four-step windows.
STEP_WINDOWS = [
    (93, 96),
    (97, 100),
    (101, 104),
    (105, 108),
    (109, 112),
    (113, 116),
]


# --------------------------------------------------
# 2. Load TRANSFER-only data
# --------------------------------------------------

print("Loading chronological datasets...")

columns = FEATURES + [TARGET, "transactionType", "step"]

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

print(f"Training TRANSFER rows: {len(train_df):,}")
print(f"Validation TRANSFER rows: {len(validation_df):,}")


# --------------------------------------------------
# 3. Train Logistic Regression
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


# --------------------------------------------------
# 4. Train Gradient Boosting
# --------------------------------------------------

print("\nTraining Histogram Gradient Boosting...")

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

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
                early_stopping=False,
                random_state=42,
            ),
        ),
    ]
)

boosting_model.fit(
    X_train,
    y_train,
    classifier__sample_weight=sample_weights,
)


# --------------------------------------------------
# 5. Generate validation predictions
# --------------------------------------------------

print("\nGenerating validation predictions...")

X_validation = validation_df[FEATURES]

logistic_scores = logistic_model.predict_proba(
    X_validation
)[:, 1]

boosting_scores = boosting_model.predict_proba(
    X_validation
)[:, 1]

validation_df = validation_df.copy()

validation_df["logistic_score"] = logistic_scores
validation_df["boosting_score"] = boosting_scores


# --------------------------------------------------
# 6. Evaluate each chronological window
# --------------------------------------------------

results = []

print("\n" + "=" * 90)
print("TRANSFER-ONLY TEMPORAL VALIDATION STABILITY")
print("=" * 90)

for start_step, end_step in STEP_WINDOWS:

    window_df = validation_df[
        validation_df["step"].between(start_step, end_step)
    ]

    y_window = window_df[TARGET]

    if y_window.nunique() < 2:
        print(
            f"Skipping steps {start_step}-{end_step}: "
            "only one target class is present."
        )
        continue

    logistic_pr = average_precision_score(
        y_window,
        window_df["logistic_score"],
    )

    logistic_roc = roc_auc_score(
        y_window,
        window_df["logistic_score"],
    )

    boosting_pr = average_precision_score(
        y_window,
        window_df["boosting_score"],
    )

    boosting_roc = roc_auc_score(
        y_window,
        window_df["boosting_score"],
    )

    row = {
        "step_window": f"{start_step}-{end_step}",
        "transactions": len(window_df),
        "fraud_rate": y_window.mean(),
        "logistic_pr_auc": logistic_pr,
        "logistic_roc_auc": logistic_roc,
        "boosting_pr_auc": boosting_pr,
        "boosting_roc_auc": boosting_roc,
    }

    results.append(row)


# --------------------------------------------------
# 7. Display and save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

output_file = DATA_DIR / "momtsim_v1_transfer_temporal_stability.csv"

results_df.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")

print("\nImportant:")
print("- Only TRANSFER transactions were evaluated.")
print("- Each window contains four consecutive simulation steps.")
print("- Both models were trained on the earlier training period.")
print("- Gradient Boosting early stopping was disabled.")
print("- PR-AUC should be interpreted alongside each window's fraud rate.")
print("- These are validation results; the test set remains untouched.")
