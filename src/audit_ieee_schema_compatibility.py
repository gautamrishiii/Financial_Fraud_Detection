from pathlib import Path

import pandas as pd


BASE = Path("dataset/raw/ieee_cis")


def get_columns(filename):
    return list(pd.read_csv(BASE / filename, nrows=0).columns)


def compare_schemas(name, train_columns, test_columns):
    train_set = set(train_columns)
    test_set = set(test_columns)

    print(f"\n{'=' * 65}")
    print(name)
    print("=" * 65)

    print(f"Train columns: {len(train_columns)}")
    print(f"Test columns:  {len(test_columns)}")

    print("\nColumns only in train:")
    print(sorted(train_set - test_set) or "None")

    print("\nColumns only in test:")
    print(sorted(test_set - train_set) or "None")

    print(f"\nSame column order: {train_columns == test_columns}")


def main():
    train_transaction = get_columns("train_transaction.csv")
    test_transaction = get_columns("test_transaction.csv")

    # The target exists only in the labeled training data.
    train_features = [
        column for column in train_transaction
        if column != "isFraud"
    ]

    compare_schemas(
        "TRANSACTION FEATURES",
        train_features,
        test_transaction,
    )

    train_identity = get_columns("train_identity.csv")
    test_identity = get_columns("test_identity.csv")

    # Normalize identity feature names.
    train_identity = [
        column.replace("-", "_") if column.startswith("id-") else column
        for column in train_identity
    ]

    test_identity = [
        column.replace("-", "_") if column.startswith("id-") else column
        for column in test_identity
    ]

    compare_schemas(
        "IDENTITY FEATURES",
        train_identity,
        test_identity,
    )


if __name__ == "__main__":
    main()