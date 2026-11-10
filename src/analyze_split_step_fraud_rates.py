
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "eda_outputs"

SPLITS = {
    "TRAINING": "momtsim_v1_train.csv",
    "VALIDATION": "momtsim_v1_validation.csv",
    "TESTING": "momtsim_v1_test.csv",
}

print("MoMTSim V1 — Fraud Rate by Simulation Step")
print("=" * 70)

for split_name, filename in SPLITS.items():

    df = pd.read_csv(
        DATA_DIR / filename,
        usecols=["step", "isFraud"]
    )

    step_stats = (
        df.groupby("step")["isFraud"]
        .agg(
            transactions="count",
            fraud_count="sum",
            fraud_rate="mean"
        )
    )

    step_stats["fraud_rate"] *= 100

    print(f"\n{split_name}")
    print("-" * 70)

    print(step_stats.to_string(
        formatters={
            "fraud_rate": lambda value: f"{value:.2f}%"
        }
    ))

    print("\nSummary:")
    print(f"Number of steps: {len(step_stats)}")
    print(
        f"Lowest step fraud rate: "
        f"{step_stats['fraud_rate'].min():.2f}%"
    )
    print(
        f"Highest step fraud rate: "
        f"{step_stats['fraud_rate'].max():.2f}%"
    )
    print(
        f"Average step fraud rate: "
        f"{step_stats['fraud_rate'].mean():.2f}%"
    )
