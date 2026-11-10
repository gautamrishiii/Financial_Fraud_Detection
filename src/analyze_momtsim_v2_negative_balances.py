
from collections import Counter

import pandas as pd

DATA_PATH = "dataset/raw/momtsim/MoMTSim_20240722202413_1000_dataset.csv"
CHUNK_SIZE = 250_000

balance_columns = [
    "oldBalInitiator",
    "newBalInitiator",
    "oldBalRecipient",
    "newBalRecipient",
]

negative_by_label = Counter()
negative_by_type = Counter()
zero_amount_by_label = Counter()
zero_amount_by_type = Counter()

negative_rows = []

for chunk in pd.read_csv(
    DATA_PATH,
    chunksize=CHUNK_SIZE,
    dtype={"initiator": "string", "recipient": "string"},
):
    for column in balance_columns:
        mask = chunk[column] < 0

        negative_by_label.update(
            chunk.loc[mask, "isFraud"].value_counts().to_dict()
        )

        negative_by_type.update(
            chunk.loc[mask, "transactionType"].value_counts().to_dict()
        )

    zero_mask = chunk["amount"] == 0

    zero_amount_by_label.update(
        chunk.loc[zero_mask, "isFraud"].value_counts().to_dict()
    )

    zero_amount_by_type.update(
        chunk.loc[zero_mask, "transactionType"].value_counts().to_dict()
    )

    if len(negative_rows) < 10:
        sample = chunk.loc[
            (chunk[balance_columns] < 0).any(axis=1),
            ["step", "transactionType", "amount",
             "oldBalInitiator", "newBalInitiator",
             "oldBalRecipient", "newBalRecipient", "isFraud"],
        ]

        remaining = 10 - len(negative_rows)
        negative_rows.extend(sample.head(remaining).to_dict("records"))


print("Negative balance counts by fraud label (per balance field)")
print("=" * 65)

for label, count in sorted(negative_by_label.items()):
    name = "Legitimate" if label == 0 else "Fraud"
    print(f"{name}: {count:,}")

print("\nNegative balance counts by transaction type (per balance field)")
for transaction_type, count in sorted(negative_by_type.items()):
    print(f"{transaction_type}: {count:,}")

print("\nZero-amount transactions by label")
for label, count in sorted(zero_amount_by_label.items()):
    name = "Legitimate" if label == 0 else "Fraud"
    print(f"{name}: {count:,}")

print("\nZero-amount transactions by type")
for transaction_type, count in sorted(zero_amount_by_type.items()):
    print(f"{transaction_type}: {count:,}")

print("\nFirst 10 rows containing a negative balance")
if negative_rows:
    print(pd.DataFrame(negative_rows).to_string(index=False))
else:
    print("No negative balance rows found.")
