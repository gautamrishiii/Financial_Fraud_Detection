from pathlib import Path

import pandas as pd

DATA_FILE = Path(
    "dataset/raw/paysim/PS_20174392719_1491204439457_log.csv"
)

CHUNK_SIZE = 200_000


def main():
    print("PaySim — Temporal Structure Audit")
    print("=" * 65)

    total_rows = 0
    fraud_by_step = {}
    transaction_counts = {}

    for chunk in pd.read_csv(
        DATA_FILE,
        usecols=["step", "isFraud", "type"],
        chunksize=CHUNK_SIZE,
    ):
        total_rows += len(chunk)

        step_summary = chunk.groupby("step").agg(
            transactions=("isFraud", "size"),
            fraud_cases=("isFraud", "sum"),
        )

        for step, row in step_summary.iterrows():
            step = int(step)
            transaction_counts[step] = (
                transaction_counts.get(step, 0)
                + int(row["transactions"])
            )
            fraud_by_step[step] = (
                fraud_by_step.get(step, 0)
                + int(row["fraud_cases"])
            )

    print(f"\nTotal transactions: {total_rows:,}")
    print(f"Unique steps: {len(transaction_counts):,}")
    print(f"Minimum step: {min(transaction_counts)}")
    print(f"Maximum step: {max(transaction_counts)}")

    summary = pd.DataFrame({
        "transactions": transaction_counts,
        "fraud_cases": fraud_by_step,
    }).sort_index()

    summary["fraud_rate"] = (
        summary["fraud_cases"] / summary["transactions"]
    )
    print("\nFraud distribution by time period:")
    print("-" * 65)

    periods = [
        (1, 100),
        (101, 200),
        (201, 300),
        (301, 400),
        (401, 500),
        (501, 600),
        (601, 700),
        (701, 743),
    ]

    for start, end in periods:
        period = summary.loc[start:end]

        transactions = int(period["transactions"].sum())
        fraud_cases = int(period["fraud_cases"].sum())

        fraud_rate = (
            fraud_cases / transactions * 100
            if transactions
            else 0
        )

        print(
            f"Steps {start:3d}-{end:3d} | "
            f"Transactions: {transactions:>10,} | "
            f"Fraud: {fraud_cases:>6,} | "
            f"Fraud rate: {fraud_rate:>7.3f}%"
        )

    print("\nFirst 10 steps:")
    print(summary.head(10).to_string())

    print("\nLast 10 steps:")
    print(summary.tail(10).to_string())

    print("\nSteps with highest fraud rates:")
    print(
        summary.sort_values(
            "fraud_rate",
            ascending=False,
        ).head(10).to_string()
    )


if __name__ == "__main__":
    main()