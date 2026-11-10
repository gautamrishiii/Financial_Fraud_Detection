
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "eda_outputs"

SPLITS = {
    "TRAINING": "momtsim_v1_train.csv",
    "VALIDATION": "momtsim_v1_validation.csv",
    "TESTING": "momtsim_v1_test.csv",
}

NUMERICAL_FEATURES = [
    "step",
    "amount",
    "oldBalInitiator",
    "newBalInitiator",
    "oldBalRecipient",
    "newBalRecipient",
    "initiator_balance_change",
    "recipient_balance_change",
    "initiator_balance_difference",
    "recipient_balance_difference",
]

CATEGORICAL_FEATURES = [
    "transactionType",
]

TARGET = "isFraud"

ALLOWED_FEATURES = (
    NUMERICAL_FEATURES + CATEGORICAL_FEATURES
)

EXCLUDED_COLUMNS = [
    "initiator",
    "recipient",
]


# --------------------------------------------------
# 2. Validation function
# --------------------------------------------------

def validate_split(name, filename):

    file_path = DATA_DIR / filename

    print(f"\n{name}")
    print("=" * 60)

    df = pd.read_csv(file_path)

    # Required columns
    required_columns = ALLOWED_FEATURES + [TARGET] + EXCLUDED_COLUMNS

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    assert not missing_columns, (
        f"Missing columns: {missing_columns}"
    )

    # Separate features and target
    X = df[ALLOWED_FEATURES].copy()
    y = df[TARGET].copy()

    # Ensure target is not part of X
    assert TARGET not in X.columns

    # Ensure raw identifiers are excluded
    assert "initiator" not in X.columns
    assert "recipient" not in X.columns

    # Check missing values
    missing_features = int(X.isna().sum().sum())
    missing_target = int(y.isna().sum())

    # Check numerical features
    numerical_data = X[NUMERICAL_FEATURES].to_numpy(
        dtype=np.float64
    )

    non_finite_values = int(
        (~np.isfinite(numerical_data)).sum()
    )

    # Check categorical values
    missing_categories = int(
        X[CATEGORICAL_FEATURES].isna().sum().sum()
    )

    # Target validation
    observed_labels = set(y.unique())

    assert observed_labels == {0, 1}, (
        f"Unexpected target labels: {observed_labels}"
    )

    # Report
    print(f"Transactions: {len(df):,}")
    print(f"Numerical features: {len(NUMERICAL_FEATURES)}")
    print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
    print(f"Total candidate features: {len(X.columns)}")
    print(f"Missing feature values: {missing_features}")
    print(f"Missing target values: {missing_target}")
    print(f"Missing categorical values: {missing_categories}")
    print(f"Non-finite numerical values: {non_finite_values}")

    # Final checks
    assert missing_features == 0
    assert missing_target == 0
    assert missing_categories == 0
    assert non_finite_values == 0

    print("STATUS: PASSED")


# --------------------------------------------------
# 3. Validate all splits
# --------------------------------------------------

print("MoMTSim V1 — Model Input Validation")
print("=" * 60)

for split_name, filename in SPLITS.items():

    validate_split(split_name, filename)

print("\nAll model-input validation checks passed.")
