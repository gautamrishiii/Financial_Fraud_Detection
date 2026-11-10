
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "momtsim"
    / "synthetic_mobile_money_transaction_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "eda_outputs"
OUTPUT_FILE = OUTPUT_DIR / "momtsim_v1_engineered.csv"


# --------------------------------------------------
# 2. Load the original dataset
# --------------------------------------------------

print("MoMTSim V1 — Feature Engineering")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"Original dataset shape: {df.shape}")


# --------------------------------------------------
# 3. Create derived balance features
# --------------------------------------------------

df["initiator_balance_change"] = (
    df["oldBalInitiator"] - df["newBalInitiator"]
)

df["recipient_balance_change"] = (
    df["newBalRecipient"] - df["oldBalRecipient"]
)

df["initiator_balance_difference"] = (
    df["amount"] - df["initiator_balance_change"]
)

df["recipient_balance_difference"] = (
    df["amount"] - df["recipient_balance_change"]
)


# --------------------------------------------------
# 4. Check numerical validity
# --------------------------------------------------

derived_features = [
    "initiator_balance_change",
    "recipient_balance_change",
    "initiator_balance_difference",
    "recipient_balance_difference",
]

print("\nDerived feature validation:")

for column in derived_features:
    invalid_count = int(
        (~np.isfinite(df[column])).sum()
    )

    print(f"{column}: {invalid_count} non-finite values")


# --------------------------------------------------
# 5. Define initial model feature columns
# --------------------------------------------------

feature_columns = [
    "amount",
    "step",
    "oldBalInitiator",
    "newBalInitiator",
    "oldBalRecipient",
    "newBalRecipient",
    *derived_features,
]

target_column = "isFraud"

X = df[feature_columns].copy()
y = df[target_column].copy()


# --------------------------------------------------
# 6. Verify feature and target separation
# --------------------------------------------------

assert target_column not in X.columns
assert "initiator" not in X.columns
assert "recipient" not in X.columns

assert len(X) == len(y)
assert X.columns.is_unique

print("\nFeature matrix validation:")
print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Features: {list(X.columns)}")


# --------------------------------------------------
# 7. Check missing and non-finite values
# --------------------------------------------------

print("\nFeature quality checks:")

print(f"Missing feature values: {int(X.isna().sum().sum())}")

numeric_values = X.to_numpy(dtype=np.float64)

print(
    "Non-finite feature values:",
    int((~np.isfinite(numeric_values)).sum()),
)

print(f"Missing target values: {int(y.isna().sum())}")


# --------------------------------------------------
# 8. Save engineered dataset
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nOutput:")
print(f"Saved engineered dataset: {OUTPUT_FILE}")
print(f"Final dataset shape: {df.shape}")

print("\nFeature engineering completed.")
