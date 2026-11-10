
import numpy as np
import pandas as pd

DATA_PATH = "dataset/raw/momtsim/MoMTSim_20240722202413_1000_dataset.csv"
CHUNK_SIZE = 250_000

features = ["amount", "oldBalInitiator", "oldBalRecipient"]

values = {
    0: {column: [] for column in features},
    1: {column: [] for column in features},
}

counts = {0: 0, 1: 0}

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=["transactionType", "isFraud"] + features,
    chunksize=CHUNK_SIZE,
):
    transfers = chunk[chunk["transactionType"] == "TRANSFER"]

    for label in [0, 1]:
        group = transfers[transfers["isFraud"] == label]
        counts[label] += len(group)

        for column in features:
            values[label][column].append(
                group[column].to_numpy()
            )

print("MoMTSim V2 — TRANSFER Feature Profile")
print("=" * 65)

for label in [0, 1]:
    name = "Legitimate" if label == 0 else "Fraud"

    print(f"\n{name} transfers: {counts[label]:,}")

    for column in features:
        data = np.concatenate(values[label][column])

        print(f"\n{column}:")
        print(f"  Mean:   {np.mean(data):,.2f}")
        print(f"  Median: {np.median(data):,.2f}")
        print(f"  P90:    {np.percentile(data, 90):,.2f}")
        print(f"  P95:    {np.percentile(data, 95):,.2f}")
        print(f"  P99:    {np.percentile(data, 99):,.2f}")
        print(f"  Min:    {np.min(data):,.2f}")
        print(f"  Max:    {np.max(data):,.2f}")
        print(f"  Zero:   {np.sum(data == 0):,}")
        print(f"  Negative: {np.sum(data < 0):,}")
