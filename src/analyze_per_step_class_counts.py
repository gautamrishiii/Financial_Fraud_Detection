
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "eda_outputs"

SPLITS = {
    "TRAINING": "momtsim_v1_train.csv",
    "VALIDATION": "momtsim_v1_validation.csv",
    "TESTING": "momtsim_v1_test.csv",
}

print("MoMTSim V1 — Per-Step Class Count Analysis")
print("=" * 70)

for split_name, filename in SPLITS.items():

    df = pd.read_csv(
        DATA_DIR / filename,
        usecols=["step", "isFraud"]
    )

    counts = (
        df.groupby("step")["isFraud"]
        .agg(
            total_transactions="count",
            fraud_count="sum"
        )
    )

    counts["legitimate_count"] = (
        counts["total_transactions"] - counts["fraud_count"]
    )

    print(f"\n{split_name}")
    print("-" * 70)

    for column in [
        "total_transactions",
        "fraud_count",
        "legitimate_count",
    ]:

        values = counts[column]

        print(f"\n{column}")
        print(f"  Mean:   {values.mean():,.2f}")
        print(f"  Median: {values.median():,.2f}")
        print(f"  Minimum: {values.min():,}")
        print(f"  Maximum: {values.max():,}")
        print(f"  Std dev: {values.std():,.2f}")

print("\nPer-step class count analysis completed.")
