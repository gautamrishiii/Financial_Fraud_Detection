
import pandas as pd

DATA_PATH = "dataset/raw/momtsim/MoMTSim_20240722202413_1000_dataset.csv"
CHUNK_SIZE = 250_000

conditions = {
    "amount_below_1000": lambda df: df["amount"] < 1_000,
    "amount_below_10000": lambda df: df["amount"] < 10_000,
    "initiator_balance_below_1000": lambda df: df["oldBalInitiator"] < 1_000,
    "initiator_balance_below_10000": lambda df: df["oldBalInitiator"] < 10_000,
    "initiator_balance_below_100000": lambda df: df["oldBalInitiator"] < 100_000,
    "initiator_balance_zero": lambda df: df["oldBalInitiator"] == 0,
    "recipient_balance_zero": lambda df: df["oldBalRecipient"] == 0,
}

results = {
    name: {"total": 0, "fraud": 0}
    for name in conditions
}

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=[
        "transactionType",
        "amount",
        "oldBalInitiator",
        "oldBalRecipient",
        "isFraud",
    ],
    chunksize=CHUNK_SIZE,
):
    transfers = chunk[chunk["transactionType"] == "TRANSFER"]

    for name, condition in conditions.items():
        selected = transfers[condition(transfers)]

        results[name]["total"] += len(selected)
        results[name]["fraud"] += int(selected["isFraud"].sum())

print("MoMTSim V2 — Potential Fraud Rule Analysis")
print("=" * 75)

for name, result in results.items():
    total = result["total"]
    fraud = result["fraud"]

    if total == 0:
        continue

    print(f"\n{name}")
    print(f"  Transactions: {total:,}")
    print(f"  Fraud:        {fraud:,}")
    print(f"  Legitimate:   {total - fraud:,}")
    print(f"  Fraud rate:   {fraud / total * 100:.4f}%")

print("\nNote: These are overlapping conditions, not independent groups.")
