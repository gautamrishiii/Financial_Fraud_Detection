from pathlib import Path

import pandas as pd


BASE = Path("dataset/raw/ieee_cis")


def audit_identity(split):
    path = BASE / f"{split}_identity.csv"
    df = pd.read_csv(path)

    # Normalize test names: id-01 -> id_01
    df.columns = [
        column.replace("-", "_") if column.startswith("id-") else column
        for column in df.columns
    ]

    print(f"\n{'=' * 65}")
    print(f"{split.upper()} IDENTITY FEATURE QUALITY")
    print("=" * 65)

    print(f"Rows: {len(df):,}")
    print(f"Features excluding TransactionID: {len(df.columns) - 1}")

    missing = df.drop(columns=["TransactionID"]).isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    print(f"\nFeatures with missing values: {len(missing)}")

    for column, count in missing.items():
        print(f"  {column}: {count:,} ({count / len(df):.2%})")

    print("\nFeature data types:")
    print(df.drop(columns=["TransactionID"]).dtypes.value_counts())


def main():
    audit_identity("train")
    audit_identity("test")


if __name__ == "__main__":
    main()