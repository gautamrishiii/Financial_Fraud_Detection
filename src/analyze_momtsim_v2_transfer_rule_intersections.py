import pandas as pd

DATA_PATH = "dataset/raw/momtsim/MoMTSim_20240722202413_1000_dataset.csv"
CHUNK_SIZE = 250_000

thresholds = [1_000, 10_000]

results = {
    threshold: {
        (amount_low, balance_low): {"total": 0, "fraud": 0}
        for amount_low in [True, False]
        for balance_low in [True, False]
    }
    for threshold in thresholds
}

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=[
        "transactionType",
        "amount",
        "oldBalInitiator",
        "isFraud",
    ],
    chunksize=CHUNK_SIZE,
):
    transfers = chunk[chunk["transactionType"] == "TRANSFER"]

    for threshold in thresholds:
        amount_low = transfers["amount"] < threshold
        balance_low = transfers["oldBalInitiator"] < threshold

        for amount_condition in [True, False]:
            for balance_condition in [True, False]:
                mask = (
                    (amount_low == amount_condition)
                    & (balance_low == balance_condition)
                )

                selected = transfers[mask]
                key = (amount_condition, balance_condition)

                results[threshold][key]["total"] += len(selected)
                results[threshold][key]["fraud"] += int(
                    selected["isFraud"].sum()
                )

print("MoMTSim V2 — Transfer Rule Intersections")
print("=" * 75)

for threshold in thresholds:
    print(f"\nThreshold: {threshold:,}")
    print("-" * 55)

    for (amount_low, balance_low), counts in results[threshold].items():
        total = counts["total"]
        fraud = counts["fraud"]
        legitimate = total - fraud

        rate = fraud / total * 100 if total else 0

        print(
            f"Amount {'<' if amount_low else '>='} {threshold:,}, "
            f"initiator balance {'<' if balance_low else '>='} "
            f"{threshold:,}"
        )
        print(f"  Transactions: {total:,}")
        print(f"  Fraud:        {fraud:,}")
        print(f"  Legitimate:   {legitimate:,}")
        print(f"  Fraud rate:   {rate:.4f}%")