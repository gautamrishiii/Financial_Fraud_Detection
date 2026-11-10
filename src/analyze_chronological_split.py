
import csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE_PATH = (
    PROJECT_ROOT
    / "dataset/raw/momtsim/"
    / "synthetic_mobile_money_transaction_dataset.csv"
)

TRAIN_PROPORTION = 0.80


def main():
    step_counts = defaultdict(
        lambda: {"legitimate": 0, "fraud": 0}
    )

    print("MoMTSim V1 — Chronological Split Analysis")
    print("=" * 65)

    with FILE_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            step = int(row["step"])
            label = row["isFraud"].strip()

            if label == "0":
                step_counts[step]["legitimate"] += 1
            elif label == "1":
                step_counts[step]["fraud"] += 1

    steps = sorted(step_counts)

    total_rows = sum(
        step_counts[step]["legitimate"]
        + step_counts[step]["fraud"]
        for step in steps
    )

    target_train_rows = total_rows * TRAIN_PROPORTION

    cumulative_rows = 0
    best_split_step = None
    best_difference = float("inf")

    for index, step in enumerate(steps[:-1]):
        current = step_counts[step]
        cumulative_rows += current["legitimate"] + current["fraud"]

        difference = abs(cumulative_rows - target_train_rows)

        if difference < best_difference:
            best_difference = difference
            best_split_step = step

    train_steps = [
        step for step in steps
        if step <= best_split_step
    ]

    test_steps = [
        step for step in steps
        if step > best_split_step
    ]

    def summarize(selected_steps):
        legitimate = sum(
            step_counts[s]["legitimate"]
            for s in selected_steps
        )

        fraud = sum(
            step_counts[s]["fraud"]
            for s in selected_steps
        )

        total = legitimate + fraud
        fraud_rate = fraud / total * 100 if total else 0

        return legitimate, fraud, total, fraud_rate

    train = summarize(train_steps)
    test = summarize(test_steps)

    print(f"\nTotal transactions: {total_rows:,}")
    print(f"Target training proportion: {TRAIN_PROPORTION:.0%}")

    print("\n1. Recommended chronological boundary")
    print(f"Training steps: {train_steps[0]}–{train_steps[-1]}")
    print(f"Testing steps: {test_steps[0]}–{test_steps[-1]}")

    print("\n2. Training set")
    print(f"Legitimate: {train[0]:,}")
    print(f"Fraud: {train[1]:,}")
    print(f"Total: {train[2]:,}")
    print(f"Fraud rate: {train[3]:.2f}%")
    print(f"Share of all records: {train[2] / total_rows * 100:.2f}%")

    print("\n3. Testing set")
    print(f"Legitimate: {test[0]:,}")
    print(f"Fraud: {test[1]:,}")
    print(f"Total: {test[2]:,}")
    print(f"Fraud rate: {test[3]:.2f}%")
    print(f"Share of all records: {test[2] / total_rows * 100:.2f}%")

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
