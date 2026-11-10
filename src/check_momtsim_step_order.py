
import csv
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE_PATH = (
    PROJECT_ROOT
    / "dataset/raw/momtsim/"
    / "synthetic_mobile_money_transaction_dataset.csv"
)


def main():
    step_counts = Counter()

    total_rows = 0
    decreasing_transitions = 0
    equal_transitions = 0
    increasing_transitions = 0

    previous_step = None
    minimum_step = float("inf")
    maximum_step = float("-inf")

    print("MoMTSim V1 — Simulation Step Order Check")
    print("=" * 60)

    with FILE_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            step = int(row["step"])
            total_rows += 1

            step_counts[step] += 1

            minimum_step = min(minimum_step, step)
            maximum_step = max(maximum_step, step)

            if previous_step is not None:
                if step < previous_step:
                    decreasing_transitions += 1
                elif step == previous_step:
                    equal_transitions += 1
                else:
                    increasing_transitions += 1

            previous_step = step

    print(f"\nTotal rows: {total_rows:,}")
    print(f"Minimum step: {minimum_step}")
    print(f"Maximum step: {maximum_step}")
    print(f"Unique simulation steps: {len(step_counts):,}")

    print("\n1. Row-to-row step transitions")
    print(f"Decreasing transitions: {decreasing_transitions:,}")
    print(f"Equal transitions: {equal_transitions:,}")
    print(f"Increasing transitions: {increasing_transitions:,}")

    print("\n2. Transaction counts by individual step")

    for step in sorted(step_counts):
        print(f"Step {step:3d}: {step_counts[step]:,} transactions")

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
