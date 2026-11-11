
from pathlib import Path

import pandas as pd

SOURCE = Path(
    "dataset/raw/paysim/PS_20174392719_1491204439457_log.csv"
)

OUTPUT_DIR = Path("eda_outputs")
CHUNK_SIZE = 200_000

SPLITS = {
    "train": (1, 400),
    "validation": (401, 600),
    "test": (601, 743),
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_files = {
        name: OUTPUT_DIR / f"paysim_{name}.csv"
        for name in SPLITS
    }

    for path in output_files.values():
        path.unlink(missing_ok=True)

    counts = {
        name: {"transactions": 0, "fraud_cases": 0}
        for name in SPLITS
    }

    for chunk in pd.read_csv(SOURCE, chunksize=CHUNK_SIZE):
        for name, (start, end) in SPLITS.items():
            subset = chunk[
                chunk["step"].between(start, end)
            ]

            if subset.empty:
                continue

            subset.to_csv(
                output_files[name],
                mode="a",
                header=not output_files[name].exists(),
                index=False,
            )

            counts[name]["transactions"] += len(subset)
            counts[name]["fraud_cases"] += int(
                subset["isFraud"].sum()
            )

    total = 0

    for name, values in counts.items():
        transactions = values["transactions"]
        fraud = values["fraud_cases"]
        total += transactions

        print(f"\n{name.upper()}")
        print(f"Transactions: {transactions:,}")
        print(f"Fraud cases: {fraud:,}")
        print(f"Fraud rate: {fraud / transactions * 100:.4f}%")
        print(f"Saved: {output_files[name]}")

    assert total == 6_362_620, "Transaction count mismatch!"
    assert sum(
        values["fraud_cases"] for values in counts.values()
    ) == 8_213, "Fraud count mismatch!"

    print("\nAll split integrity checks passed.")


if __name__ == "__main__":
    main()
