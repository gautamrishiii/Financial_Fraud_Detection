
from pathlib import Path

import numpy as np
import pandas as pd

TRANSACTION_FILE = Path(
    "dataset/raw/ieee_cis/train_transaction.csv"
)
IDENTITY_FILE = Path(
    "dataset/raw/ieee_cis/train_identity.csv"
)

OUTPUT_DIR = Path("eda_outputs")

NUMERIC_FEATURES = [
    "TransactionAmt",
    "TransactionDT",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "dist1",
    *[f"C{i}" for i in range(1, 15)],
    "id_01",
    "id_02",
    "id_05",
    "id_06",
    "id_11",
    "id_13",
    "id_17",
    "id_19",
    "id_20",
]

CATEGORICAL_FEATURES = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
]

def main():
    print("IEEE-CIS — Preparing Chronological Splits")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transaction_columns = [
        "TransactionID",
        "TransactionDT",
        "isFraud",
        *[
            col for col in NUMERIC_FEATURES
            if not col.startswith("id_")
        ],
        *[
            col for col in CATEGORICAL_FEATURES
            if col not in ["DeviceType", "DeviceInfo"]
        ],
    ]

    transactions = pd.read_csv(
        TRANSACTION_FILE,
        usecols=transaction_columns,
    )

    identity_columns = [
        "TransactionID",
        "DeviceType",
        "DeviceInfo",
        "id_01",
        "id_02",
        "id_05",
        "id_06",
        "id_11",
        "id_13",
        "id_17",
        "id_19",
        "id_20",
    ]

    identity = pd.read_csv(
        IDENTITY_FILE,
        usecols=identity_columns,
    )

    if identity["TransactionID"].duplicated().any():
        raise ValueError("Duplicate TransactionID in identity data.")

    transactions = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )

    transactions["has_identity"] = (
        transactions["DeviceType"].notna()
    ).astype("int8")

    transactions.drop(columns=["TransactionID"], inplace=True)

    q70 = 10437998.1
    q85 = 13151846.0

    transactions["split"] = np.select(
        [
            transactions["TransactionDT"] <= q70,
            transactions["TransactionDT"] <= q85,
        ],
        ["train", "validation"],
        default="test",
    )

    feature_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + ["has_identity"]
    )

    output_columns = feature_columns + ["isFraud"]

    for split in ["train", "validation", "test"]:
        subset = transactions.loc[
            transactions["split"] == split,
            output_columns,
        ].copy()

        output_file = OUTPUT_DIR / f"ieee_{split}.csv"
        subset.to_csv(output_file, index=False)

        fraud_count = int(subset["isFraud"].sum())

        print(f"\n{split.upper()}")
        print(f"Transactions: {len(subset):,}")
        print(f"Fraud cases: {fraud_count:,}")
        print(
            f"Fraud rate: "
            f"{subset['isFraud'].mean() * 100:.4f}%"
        )
        print(f"Saved: {output_file}")

    print("\nIEEE-CIS split preparation completed.")


if __name__ == "__main__":
    main()