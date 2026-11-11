from pathlib import Path

import pandas as pd

DATA_FILE = Path(
    "dataset/raw/paysim/PS_20174392719_1491204439457_log.csv"
)

CHUNK_SIZE = 200_000


def main():
    print("PaySim — Temporal Transaction Type Audit")
    print("=" * 70)

    summaries = []

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

    for chunk in pd.read_csv(
        DATA_FILE,
        usecols=["step", "type", "isFraud"],
        chunksize=CHUNK_SIZE,
    ):
        grouped = (
            chunk.groupby(["step", "type"], observed=True)
            .agg(
                transactions=("isFraud", "size"),
                fraud_cases=("isFraud", "sum"),
            )
            .reset_index()
        )
        summaries.append(grouped)

    summary = (
        pd.concat(summaries)
        .groupby(["step", "type"], as_index=False)
        .sum()
    )

    summary["period"] = pd.cut(
        summary["step"],
        bins=[0, 100, 200, 300, 400, 500, 600, 700, 743],
        labels=[
            "1-100", "101-200", "201-300", "301-400",
            "401-500", "501-600", "601-700", "701-743",
        ],
    )

    for period in summary["period"].cat.categories:
        period_data = summary[summary["period"] == period]

        print(f"\nPeriod: {period}")
        print("-" * 60)

        result = (
            period_data.groupby("type")
            .agg(
                transactions=("transactions", "sum"),
                fraud_cases=("fraud_cases", "sum"),
            )
        )

        result["fraud_rate"] = (
            result["fraud_cases"] / result["transactions"] * 100
        )

        print(result.to_string())


if __name__ == "__main__":
    main()