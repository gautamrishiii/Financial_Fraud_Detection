
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "eda_outputs"

FILES = {
    "Training": DATA_DIR / "momtsim_v1_train.csv",
    "Validation": DATA_DIR / "momtsim_v1_validation.csv",
    "Testing": DATA_DIR / "momtsim_v1_test.csv",
}

EXPECTED_ROWS = {
    "Training": 1_161_938,
    "Validation": 215_320,
    "Testing": 342_923,
}

print("MoMTSim V1 — Saved Split Verification")
print("=" * 60)

total_rows = 0

for name, file_path in FILES.items():

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    df = pd.read_csv(file_path)

    expected = EXPECTED_ROWS[name]

    assert len(df) == expected, (
        f"{name}: expected {expected:,}, got {len(df):,}"
    )

    assert df["isFraud"].notna().all()
    assert df["step"].notna().all()

    print(f"\n{name}")
    print("-" * 40)
    print(f"File: {file_path.name}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Step range: {df['step'].min()}–{df['step'].max()}")
    print(f"Fraud records: {int(df['isFraud'].sum()):,}")

    total_rows += len(df)

assert total_rows == 1_720_181

print("\n" + "=" * 60)
print(f"Total verified rows: {total_rows:,}")
print("All saved split files passed verification.")
