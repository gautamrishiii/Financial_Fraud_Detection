
import csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE_PATH = (
    PROJECT_ROOT
    / "dataset/raw/momtsim/"
    / "synthetic_mobile_money_transaction_dataset.csv"
)

TRAIN_END_STEP = 116
TOLERANCE = 0.02


def main():
    groups = defaultdict(lambda: {
        "count": 0,
        "initiator_match": 0,
        "recipient_match": 0,
        "negative_initiator_old": 0,
        "negative_initiator_new": 0,
        "negative_recipient_old": 0,
        "negative_recipient_new": 0,
    })

    print("MoMTSim V1 — Transfer Balance Analysis by Split and Label")
    print("=" * 70)

    with FILE_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            if row["transactionType"].strip() != "TRANSFER":
                continue

            step = int(row["step"])
            label = row["isFraud"].strip()

            split = (
                "Training"
                if step <= TRAIN_END_STEP
                else "Testing"
            )

            current = groups[(split, label)]

            amount = float(row["amount"])

            old_i = float(row["oldBalInitiator"])
            new_i = float(row["newBalInitiator"])

            old_r = float(row["oldBalRecipient"])
            new_r = float(row["newBalRecipient"])

            current["count"] += 1

            if abs((old_i - new_i) - amount) <= TOLERANCE:
                current["initiator_match"] += 1

            if abs((new_r - old_r) - amount) <= TOLERANCE:
                current["recipient_match"] += 1

            if old_i < 0:
                current["negative_initiator_old"] += 1

            if new_i < 0:
                current["negative_initiator_new"] += 1

            if old_r < 0:
                current["negative_recipient_old"] += 1

            if new_r < 0:
                current["negative_recipient_new"] += 1

    for split in ["Training", "Testing"]:
        for label in ["0", "1"]:
            current = groups[(split, label)]
            count = current["count"]

            if count == 0:
                continue

            description = (
                "Fraud" if label == "1" else "Legitimate"
            )

            print(f"\n{split} — {description}")
            print("-" * 50)
            print(f"Transactions: {count:,}")

            print(
                f"Initiator balance matches: "
                f"{current['initiator_match']:,} "
                f"({current['initiator_match'] / count * 100:.2f}%)"
            )

            print(
                f"Recipient balance matches: "
                f"{current['recipient_match']:,} "
                f"({current['recipient_match'] / count * 100:.2f}%)"
            )

            for key, description in [
                ("negative_initiator_old", "Negative old initiator balance"),
                ("negative_initiator_new", "Negative new initiator balance"),
                ("negative_recipient_old", "Negative old recipient balance"),
                ("negative_recipient_new", "Negative new recipient balance"),
            ]:
                value = current[key]

                print(
                    f"{description}: {value:,} "
                    f"({value / count * 100:.2f}%)"
                )

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
