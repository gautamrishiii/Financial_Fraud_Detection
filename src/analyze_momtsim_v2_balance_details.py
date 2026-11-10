
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

negative_by_column_label = Counter()
negative_by_column_type = Counter()
rows_with_negative_by_label = Counter()
rows_with_negative_by_type = Counter()
negative_fields_per_row = Counter()

total_negative_rows = 0

for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):

    negative_masks = chunk[balance_columns] < 0

    for column in balance_columns:
        mask = negative_masks[column]

        for label, count in chunk.loc[mask, "isFraud"].value_counts().items():
            negative_by_column_label[(column, label)] += int(count)

        for transaction_type, count in chunk.loc[
            mask, "transactionType"
        ].value_counts().items():
            negative_by_column_type[(column, transaction_type)] += int(count)

    rows_with_negative = negative_masks.any(axis=1)
    negative_count_per_row = negative_masks.sum(axis=1)

    total_negative_rows += int(rows_with_negative.sum())

    for label, count in chunk.loc[
        rows_with_negative, "isFraud"
    ].value_counts().items():
        rows_with_negative_by_label[label] += int(count)

    for transaction_type, count in chunk.loc[
        rows_with_negative, "transactionType"
    ].value_counts().items():
        rows_with_negative_by_type[transaction_type] += int(count)

    negative_fields_per_row.update(
        negative_count_per_row[rows_with_negative].value_counts().to_dict()
    )

print("MoMTSim V2 — Negative Balance Breakdown")
print("=" * 60)

print("\nNegative values by balance column and fraud label:")
for column in balance_columns:
    print(f"\n{column}:")
    for label in [0, 1]:
        count = negative_by_column_label[(column, label)]
        name = "Legitimate" if label == 0 else "Fraud"
        print(f"  {name}: {count:,}")

print("\nNegative values by balance column and transaction type:")
for column in balance_columns:
    print(f"\n{column}:")
    for transaction_type in sorted(
        {key[1] for key in negative_by_column_type if key[0] == column}
    ):
        count = negative_by_column_type[(column, transaction_type)]
        print(f"  {transaction_type}: {count:,}")

print("\nUnique transactions containing at least one negative balance:")
print(f"Total: {total_negative_rows:,}")

for label in [0, 1]:
    name = "Legitimate" if label == 0 else "Fraud"
    print(f"{name}: {rows_with_negative_by_label[label]:,}")

print("\nUnique transactions with negative balances by type:")
for transaction_type, count in sorted(rows_with_negative_by_type.items()):
    print(f"{transaction_type}: {count:,}")

print("\nNumber of negative balance fields per affected transaction:")
for field_count, row_count in sorted(negative_fields_per_row.items()):
    print(f"{field_count} field(s): {row_count:,} transactions")
