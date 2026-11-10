import pandas as pd
from pathlib import Path

SPLITS = {
    "TRAIN": Path("eda_outputs/momtsim_v2_train.csv"),
    "VALIDATION": Path("eda_outputs/momtsim_v2_validation.csv"),
    "TEST": Path("eda_outputs/momtsim_v2_test.csv"),
}

print("MoMTSim V2 — Transaction Type Distribution by Split")
print("=" * 75)

for split_name, path in SPLITS.items():

    df = pd.read_csv(
        path,
        usecols=["transactionType", "isFraud"],
    )

    print(f"\n{split_name}")
    print("-" * 60)

    summary = (
        df.groupby("transactionType")["isFraud"]
        .agg(
            transactions="count",
            fraud="sum",
        )
    )

    summary["legitimate"] = (
        summary["transactions"] - summary["fraud"]
    )

    summary["fraud_rate_pct"] = (
        summary["fraud"] / summary["transactions"] * 100
    )

    print(summary.to_string())

    transfer = df[df["transactionType"] == "TRANSFER"]

    print("\nTRANSFER-only summary")
    print(f"  Transactions: {len(transfer):,}")
    print(f"  Fraud:        {int(transfer['isFraud'].sum()):,}")
    print(
        f"  Legitimate:   "
        f"{len(transfer) - int(transfer['isFraud'].sum()):,}"
    )
    print(
        f"  Fraud rate:   "
        f"{transfer['isFraud'].mean() * 100:.2f}%"
    )

    del df