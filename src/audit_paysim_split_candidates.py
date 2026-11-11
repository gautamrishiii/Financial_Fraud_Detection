
from pathlib import Path

import pandas as pd

DATA_FILE = Path(
    "dataset/raw/paysim/PS_20174392719_1491204439457_log.csv"
)

CHUNK_SIZE = 200_000

SPLITS = {
    "train": (1, 400),
    "validation": (401, 600),
    "test": (601, 743),
}


def main():
    print("PaySim — Chronological Split Audit")
    print("=" * 70)

    summaries = []

    for chunk in pd.read_csv(
        DATA_FILE,
        usecols=["step", "type", "isFraud"],
        chunksize=CHUNK_SIZE,
    ):
        chunk["split"] = pd.cut(
            chunk["step"],
            bins=[0, 400, 600, 743],
            labels=["train", "validation", "test"],
        )

        grouped = chunk.groupby(
            ["split", "type"],
            observed=True,
        ).agg(
            transactions=("isFraud", "size"),
            fraud_cases=("isFraud", "sum"),
        ).reset_index()

        summaries.append(grouped)

    result = (
        pd.concat(summaries)
        .groupby(["split", "type"], observed=True)[
            ["transactions", "fraud_cases"]
        ]
        .sum()
        .reset_index()
    )

    for split, (start, end) in SPLITS.items():
        data = result[result["split"] == split]

        transactions = data["transactions"].sum()
        fraud_cases = data["fraud_cases"].sum()

        print(f"\n{split.upper()} — Steps {start}-{end}")
        print("-" * 60)
        print(data.to_string(index=False))

        print(f"\nTotal transactions: {transactions:,}")
        print(f"Fraud cases: {fraud_cases:,}")
        print(
            f"Fraud rate: {fraud_cases / transactions * 100:.4f}%"
        )


if __name__ == "__main__":
    main()