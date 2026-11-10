
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "eda_outputs"
    / "momtsim_v1_engineered.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "eda_outputs"


# --------------------------------------------------
# 2. Define chronological boundaries
# --------------------------------------------------

TRAIN_END_STEP = 92
VALIDATION_END_STEP = 116

# Training:   steps 0–92
# Validation: steps 93–116
# Testing:    steps 117–143


# --------------------------------------------------
# 3. Load engineered dataset
# --------------------------------------------------

print("MoMTSim V1 — Chronological Data Splitting")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)

print(f"Total transactions: {len(df):,}")
print(f"Minimum step: {df['step'].min()}")
print(f"Maximum step: {df['step'].max()}")


# --------------------------------------------------
# 4. Create chronological splits
# --------------------------------------------------

train_df = df.loc[
    df["step"] <= TRAIN_END_STEP
].copy()

validation_df = df.loc[
    (df["step"] > TRAIN_END_STEP)
    & (df["step"] <= VALIDATION_END_STEP)
].copy()

test_df = df.loc[
    df["step"] > VALIDATION_END_STEP
].copy()


# --------------------------------------------------
# 5. Verify split boundaries
# --------------------------------------------------

assert train_df["step"].max() == TRAIN_END_STEP

assert validation_df["step"].min() == TRAIN_END_STEP + 1
assert validation_df["step"].max() == VALIDATION_END_STEP

assert test_df["step"].min() == VALIDATION_END_STEP + 1

assert train_df["step"].max() < validation_df["step"].min()
assert validation_df["step"].max() < test_df["step"].min()

assert (
    len(train_df) + len(validation_df) + len(test_df)
    == len(df)
)


# --------------------------------------------------
# 6. Display split distributions
# --------------------------------------------------

def display_split_summary(name, split_df):

    fraud_count = int(split_df["isFraud"].sum())

    legitimate_count = len(split_df) - fraud_count

    fraud_rate = (
        fraud_count / len(split_df) * 100
        if len(split_df) > 0
        else 0
    )

    print(f"\n{name}")
    print("-" * 45)
    print(f"Transactions: {len(split_df):,}")
    print(f"Minimum step: {split_df['step'].min()}")
    print(f"Maximum step: {split_df['step'].max()}")
    print(f"Legitimate: {legitimate_count:,}")
    print(f"Fraud: {fraud_count:,}")
    print(f"Fraud rate: {fraud_rate:.2f}%")


display_split_summary("TRAINING", train_df)

display_split_summary("VALIDATION", validation_df)

display_split_summary("TESTING", test_df)


# --------------------------------------------------
# 7. Save split datasets
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_file = OUTPUT_DIR / "momtsim_v1_train.csv"
validation_file = OUTPUT_DIR / "momtsim_v1_validation.csv"
test_file = OUTPUT_DIR / "momtsim_v1_test.csv"

train_df.to_csv(train_file, index=False)
validation_df.to_csv(validation_file, index=False)
test_df.to_csv(test_file, index=False)


# --------------------------------------------------
# 8. Final verification
# --------------------------------------------------

print("\nSaved files:")
print(train_file)
print(validation_file)
print(test_file)

print("\nChronological splitting completed successfully.")
