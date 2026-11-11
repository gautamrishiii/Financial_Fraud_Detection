from pathlib import Path

import pandas as pd

DATA_FILE = Path(
    "dataset/raw/paysim/PS_20174392719_1491204439457_log.csv"
)

CHUNK_SIZE = 200_000


def main():
    print("PaySim — Transaction Type Distribution")
    print("=" * 65)

    counts = {}
    fraud_counts = {}

    for chunk in pd.read_csv(
        DATA_FILE,
        usecols=["type", "isFraud"],
        chunksize=CHUNK_SIZE,
    ):
        grouped = chunk.groupby("type").agg(
            transactions=("isFraud", "size"),
            fraud_cases=("isFraud", "sum"),
        )

        for transaction_type, row in grouped.iterrows():
            counts[transaction_type] = (
                counts.get(transaction_type, 0)
                + int(row["transactions"])
            )
            fraud_counts[transaction_type] = (
                fraud_counts.get(transaction_type, 0)
                + int(row["fraud_cases"])
            )

    result = pd.DataFrame({
        "transactions": counts,
        "fraud_cases": fraud_counts,
    })

    result["transaction_share"] = (
        result["transactions"] / result["transactions"].sum() * 100
    )

    result["fraud_rate"] = (
        result["fraud_cases"] / result["transactions"] * 100
    )

    print(result.sort_values(
        "transactions",
        ascending=False,
    ).to_string())

    print("\nTotal fraud cases:", f"{result['fraud_cases'].sum():,}")


if __name__ == "__main__":
    main()