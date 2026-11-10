import pandas as pd
from pathlib import Path

SPLITS = {
    "TRAIN": Path("eda_outputs/momtsim_v2_train.csv"),
    "VALIDATION": Path("eda_outputs/momtsim_v2_validation.csv"),
}

AMOUNT_BINS = [-0.01, 1_000, 10_000, 100_000, 1_000_000, float("inf")]
AMOUNT_LABELS = [
    "0-1K",
    "1K-10K",
    "10K-100K",
    "100K-1M",
    "Above 1M",
]

BALANCE_BINS = [
    float("-inf"), 0, 1_000, 10_000, 100_000,
    1_000_000, 10_000_000, 100_000_000, float("inf")
]

BALANCE_LABELS = [
    "Negative",
    "0-1K",
    "1K-10K",
    "10K-100K",
    "100K-1M",
    "1M-10M",
    "10M-100M",
    "Above 100M",
]

for split_name, path in SPLITS.items():

    df = pd.read_csv(
        path,
        usecols=[
            "transactionType",
            "amount",
            "oldBalInitiator",
            "isFraud",
        ],
    )

    df = df[df["transactionType"] == "TRANSFER"].copy()

    df["amount_group"] = pd.cut(
        df["amount"],
        bins=AMOUNT_BINS,
        labels=AMOUNT_LABELS,
        right=False,
    )

    df["balance_group"] = pd.cut(
        df["oldBalInitiator"],
        bins=BALANCE_BINS,
        labels=BALANCE_LABELS,
        right=False,
    )

    summary = (
        df.groupby(
            ["amount_group", "balance_group"],
            observed=True,
        )["isFraud"]
        .agg(
            transactions="count",
            fraud="sum",
        )
        .reset_index()
    )

    summary["legitimate"] = (
        summary["transactions"] - summary["fraud"]
    )

    summary["fraud_rate_pct"] = (
        summary["fraud"] / summary["transactions"] * 100
    )

    print(f"\n{split_name} — Amount and Initiator Balance Grid")
    print("=" * 90)

    for _, row in summary.iterrows():
        # Show substantial groups and every group containing legitimate records.
        if row["transactions"] >= 100 or row["legitimate"] > 0:
            print(
                f"Amount: {str(row['amount_group']):>12} | "
                f"Balance: {str(row['balance_group']):>12} | "
                f"N: {int(row['transactions']):>10,} | "
                f"Fraud: {int(row['fraud']):>10,} | "
                f"Legit: {int(row['legitimate']):>8,} | "
                f"Rate: {row['fraud_rate_pct']:7.2f}%"
            )

    del df

print("\nJoint distribution analysis completed.")