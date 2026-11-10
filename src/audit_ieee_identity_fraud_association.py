from pathlib import Path

import pandas as pd


BASE = Path("dataset/raw/ieee_cis")


def main():
    transactions = pd.read_csv(
        BASE / "train_transaction.csv",
        usecols=["TransactionID", "isFraud"],
    )

    identity_ids = pd.read_csv(
        BASE / "train_identity.csv",
        usecols=["TransactionID"],
    )

    transactions["has_identity"] = (
        transactions["TransactionID"].isin(identity_ids["TransactionID"])
    )

    print("\nIEEE-CIS: Identity Availability vs Fraud")
    print("=" * 65)

    summary = (
        transactions.groupby("has_identity", observed=True)["isFraud"]
        .agg(["count", "sum", "mean"])
    )

    summary.index = ["No identity", "Has identity"]

    for group, row in summary.iterrows():
        print(f"\n{group}")
        print(f"  Transactions: {int(row['count']):,}")
        print(f"  Fraud cases:  {int(row['sum']):,}")
        print(f"  Fraud rate:   {row['mean']:.4%}")


if __name__ == "__main__":
    main()