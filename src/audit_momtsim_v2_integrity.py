
from collections import Counter

import pandas as pd

DATA_PATH = "dataset/raw/momtsim/MoMTSim_20240722202413_1000_dataset.csv"
CHUNK_SIZE = 250_000

numeric_columns = [
    "amount",
    "oldBalInitiator",
    "newBalInitiator",
    "oldBalRecipient",
    "newBalRecipient",
]

negative_counts = Counter()
zero_counts = Counter()
identifier_lengths = {
    "initiator": Counter(),
    "recipient": Counter(),
}
invalid_labels = Counter()

previous_step = None
out_of_order_count = 0
total_rows = 0

for chunk in pd.read_csv(
    DATA_PATH,
    chunksize=CHUNK_SIZE,
    dtype={"initiator": "string", "recipient": "string"},
):
    total_rows += len(chunk)

    for column in numeric_columns:
        negative_counts[column] += int((chunk[column] < 0).sum())
        zero_counts[column] += int((chunk[column] == 0).sum())

    for column in identifier_lengths:
        lengths = chunk[column].str.len().value_counts()
        identifier_lengths[column].update(lengths.to_dict())

    invalid_labels.update(
        chunk.loc[~chunk["isFraud"].isin([0, 1]), "isFraud"].value_counts().to_dict()
    )

    steps = chunk["step"]

    if previous_step is not None and len(steps) > 0:
        if steps.iloc[0] < previous_step:
            out_of_order_count += 1

    out_of_order_count += int((steps.diff().dropna() < 0).sum())

    if len(steps) > 0:
        previous_step = steps.iloc[-1]


print("MoMTSim V2 — Data Integrity Audit")
print("=" * 55)

print(f"\nTotal rows checked: {total_rows:,}")

print("\nNegative values:")
for column, count in negative_counts.items():
    print(f"{column}: {count:,}")

print("\nZero values:")
for column, count in zero_counts.items():
    print(f"{column}: {count:,}")

print("\nIdentifier length distribution:")
for column, counts in identifier_lengths.items():
    print(f"\n{column}:")
    for length, count in sorted(counts.items()):
        print(f"  Length {length}: {count:,}")

print("\nLabel validation:")
if invalid_labels:
    print("Unexpected labels found:")
    for label, count in invalid_labels.items():
        print(f"  {label}: {count:,}")
else:
    print("Only labels 0 and 1 found.")

print("\nStep ordering:")
if out_of_order_count == 0:
    print("Steps are non-decreasing across the dataset.")
else:
    print(f"Out-of-order step transitions: {out_of_order_count:,}")
