import pandas as pd

DATA_PATH = "dataset/raw/momtsim/MoMTSim_20240722202413_1000_dataset.csv"
CHUNK_SIZE = 250_000
WINDOW_SIZE = 16

results = {}

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=["step", "transactionType", "isFraud"],
    chunksize=CHUNK_SIZE,
):
    chunk["window"] = (chunk["step"] // WINDOW_SIZE) * WINDOW_SIZE

    for window, group in chunk.groupby("window"):
        stats = results.setdefault(
            int(window),
            {
                "total": 0,
                "fraud": 0,
                "transfers": 0,
                "transfer_fraud": 0,
            },
        )

        stats["total"] += len(group)
        stats["fraud"] += int(group["isFraud"].sum())

        transfers = group[group["transactionType"] == "TRANSFER"]

        stats["transfers"] += len(transfers)
        stats["transfer_fraud"] += int(transfers["isFraud"].sum())

print("MoMTSim V2 — Temporal Dataset Distribution")
print("=" * 80)

grand_total = 0
grand_fraud = 0

for window in sorted(results):
    stats = results[window]

    total = stats["total"]
    fraud = stats["fraud"]
    transfers = stats["transfers"]
    transfer_fraud = stats["transfer_fraud"]

    grand_total += total
    grand_fraud += fraud

    print(f"\nSteps {window}-{window + WINDOW_SIZE - 1}")
    print(f"  All transactions:  {total:,}")
    print(f"  All fraud:         {fraud:,}")
    print(f"  Overall fraud rate:{fraud / total * 100:8.2f}%")
    print(f"  Transfers:         {transfers:,}")
    print(f"  Transfer fraud:    {transfer_fraud:,}")
    print(
        f"  Transfer fraud rate:"
        f"{transfer_fraud / transfers * 100:8.2f}%"
    )

print("\n" + "=" * 80)
print(f"Total transactions: {grand_total:,}")
print(f"Total fraud:        {grand_fraud:,}")
print(f"Overall fraud rate: {grand_fraud / grand_total * 100:.2f}%")