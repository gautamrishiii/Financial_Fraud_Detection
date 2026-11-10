from pathlib import Path

import pandas as pd


BASE = Path("dataset/raw/ieee_cis")


def audit_identity(split):
    print("\n" + "=" * 65)
    print(f"IEEE-CIS {split.upper()} IDENTITY AUDIT")
    print("=" * 65)

    transaction_path = BASE / f"{split}_transaction.csv"
    identity_path = BASE / f"{split}_identity.csv"

    transactions = pd.read_csv(
        transaction_path,
        usecols=["TransactionID"],
    )

    identity = pd.read_csv(identity_path)

    transaction_ids = set(transactions["TransactionID"])
    identity_ids = set(identity["TransactionID"])

    print(f"Transaction rows: {len(transactions):,}")
    print(f"Identity rows:    {len(identity):,}")
    print(f"Identity columns: {len(identity.columns)}")

    print(f"Duplicate transaction IDs: {transactions['TransactionID'].duplicated().sum():,}")
    print(f"Duplicate identity IDs:    {identity['TransactionID'].duplicated().sum():,}")

    print(f"Identity IDs matching transactions: {len(identity_ids & transaction_ids):,}")
    print(f"Identity IDs without transaction:   {len(identity_ids - transaction_ids):,}")
    print(f"Transactions without identity:      {len(transaction_ids - identity_ids):,}")

    print("\nIdentity feature columns:")
    print([column for column in identity.columns if column != "TransactionID"])


def main():
    audit_identity("train")
    audit_identity("test")


if __name__ == "__main__":
    main()