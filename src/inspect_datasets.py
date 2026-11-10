
import csv
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset files to inspect
DATASET_FILES = {
    "MoMTSim file 1": "dataset/raw/momtsim/MoMTSim_20240722202413_1000_dataset.csv",
    "MoMTSim file 2": "dataset/raw/momtsim/synthetic_mobile_money_transaction_dataset.csv",
    "IEEE-CIS train transactions": "dataset/raw/ieee_cis/train_transaction.csv",
    "IEEE-CIS train identity": "dataset/raw/ieee_cis/train_identity.csv",
    "IEEE-CIS test transactions": "dataset/raw/ieee_cis/test_transaction.csv",
    "IEEE-CIS test identity": "dataset/raw/ieee_cis/test_identity.csv",
    "PaySim": "dataset/raw/paysim/PS_20174392719_1491204439457_log.csv",
}


def inspect_csv(name, relative_path):
    """Display CSV columns and a few sample records."""

    file_path = PROJECT_ROOT / relative_path

    print("\n" + "=" * 80)
    print(f"DATASET: {name}")
    print("=" * 80)

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return

    print(f"File size: {file_path.stat().st_size / (1024 ** 2):.2f} MB")

    try:
        with file_path.open(
            mode="r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.reader(file)

            # Read column names
            columns = next(reader)

            print(f"\nTotal columns: {len(columns)}")
            print("\nColumn names:")

            for index, column in enumerate(columns, start=1):
                print(f"{index:3}. {column}")

            # Display three sample records
            print("\nFirst 3 data records:")

            for index, row in enumerate(reader, start=1):
                print(f"\nRecord {index}:")
                print(row[:10])

                if index == 3:
                    break

    except Exception as error:
        print(f"ERROR while reading file: {error}")


def main():
    print("FINANCIAL FRAUD DETECTION - DATASET INSPECTION")

    for name, relative_path in DATASET_FILES.items():
        inspect_csv(name, relative_path)

    print("\n" + "=" * 80)
    print("INITIAL DATASET INSPECTION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
