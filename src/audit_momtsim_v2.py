from collections import Counter

import pandas as pd

DATA_PATH = "dataset/raw/momtsim/MoMTSim_20240722202413_1000_dataset.csv"
CHUNK_SIZE = 250_000

label_counts = Counter()
type_counts = Counter()
fraud_by_type = Counter()
step_min = None
step_max = None
amount_min = float("inf")
amount_max = float("-inf")
total_rows = 0
missing_values = Counter()

for chunk in pd.read_csv(
    DATA_PATH,
    chunksize=CHUNK_SIZE,
    dtype={"initiator": "string", "recipient": "string"},
):
    total_rows += len(chunk)

    label_counts.update(chunk["isFraud"].value_counts().to_dict())
    type_counts.update(chunk["transactionType"].value_counts().to_dict())

    fraud_counts = chunk.groupby("transactionType")["isFraud"].sum()
    fraud_by_type.update(fraud_counts.to_dict())

    current_min = chunk["step"].min()
    current_max = chunk["step"].max()
    step_min = current_min if step_min is None else min(step_min, current_min)
    step_max = current_max if step_max is None else max(step_max, current_max)

    amount_min = min(amount_min, chunk["amount"].min())
    amount_max = max(amount_max, chunk["amount"].max())

    missing_values.update(chunk.isna().sum().to_dict())

print("MoMTSim V2 — Initial Data Audit")
print("=" * 55)

print(f"\nTotal transactions: {total_rows:,}")

print("\nLabel distribution:")
for label, count in sorted(label_counts.items()):
    name = "Legitimate" if label == 0 else "Fraud" if label == 1 else f"Other ({label})"
    print(f"{name}: {count:,} ({count / total_rows * 100:.2f}%)")

print("\nTransaction type distribution:")
for transaction_type, count in sorted(type_counts.items()):
    fraud_count = fraud_by_type[transaction_type]
    fraud_rate = fraud_count / count * 100
    print(
        f"{transaction_type}: {count:,} transactions | "
        f"{fraud_count:,} fraud | {fraud_rate:.2f}% fraud rate"
    )

print("\nNumeric ranges:")
print(f"Step: {step_min} to {step_max}")
print(f"Amount: {amount_min:,.2f} to {amount_max:,.2f}")

print("\nMissing values:")
for column, count in missing_values.items():
    if count > 0:
        print(f"{column}: {count:,}")

if not any(missing_values.values()):
    print("No missing values detected.")