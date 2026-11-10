import pandas as pd

DATA_PATH = "dataset/raw/momtsim/MoMTSim_20240722202413_1000_dataset.csv"
CHUNK_SIZE = 250_000
WINDOW_SIZE = 16

results = {}

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=[
        "step",
        "transactionType",
        "amount",
        "oldBalInitiator",
        "isFraud",
    ],
    chunksize=CHUNK_SIZE,
):
    transfers = chunk[chunk["transactionType"] == "TRANSFER"].copy()

    transfers["window"] = (
        transfers["step"] // WINDOW_SIZE
    ) * WINDOW_SIZE

    transfers["low_amount"] = transfers["amount"] < 1_000
    transfers["low_balance"] = transfers["oldBalInitiator"] < 1_000

    transfers["both_low"] = (
        transfers["low_amount"] & transfers["low_balance"]
    )

    for window, group in transfers.groupby("window"):
        stats = results.setdefault(
            int(window),
            {
                "total": 0,
                "fraud": 0,
                "low_amount_total": 0,
                "low_amount_fraud": 0,
                "low_balance_total": 0,
                "low_balance_fraud": 0,
                "both_low_total": 0,
                "both_low_fraud": 0,
            },
        )

        stats["total"] += len(group)
        stats["fraud"] += int(group["isFraud"].sum())

        for condition, prefix in [
            ("low_amount", "low_amount"),
            ("low_balance", "low_balance"),
            ("both_low", "both_low"),
        ]:
            selected = group[group[condition]]

            stats[f"{prefix}_total"] += len(selected)
            stats[f"{prefix}_fraud"] += int(
                selected["isFraud"].sum()
            )

print("MoMTSim V2 — Temporal Stability of Transfer Rules")
print("=" * 85)

for window in sorted(results):
    stats = results[window]

    total = stats["total"]
    fraud = stats["fraud"]

    print(f"\nSteps {window}-{window + WINDOW_SIZE - 1}")
    print(f"  Transfers:       {total:,}")
    print(f"  Overall fraud:   {fraud / total * 100:.2f}%")

    for prefix, label in [
        ("low_amount", "Amount < 1,000"),
        ("low_balance", "Initiator balance < 1,000"),
        ("both_low", "Both conditions"),
    ]:
        condition_total = stats[f"{prefix}_total"]
        condition_fraud = stats[f"{prefix}_fraud"]

        rate = (
            condition_fraud / condition_total * 100
            if condition_total else 0
        )

        coverage = condition_total / total * 100

        print(
            f"  {label}: "
            f"{condition_total:,} transactions "
            f"({coverage:.2f}% of transfers), "
            f"fraud rate {rate:.2f}%"
        )