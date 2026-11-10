import pandas as pd
from pathlib import Path

SPLITS = {
    "TRAIN": Path("eda_outputs/momtsim_v2_train.csv"),
    "VALIDATION": Path("eda_outputs/momtsim_v2_validation.csv"),
    "TEST": Path("eda_outputs/momtsim_v2_test.csv"),
}

FEATURES = [
    "amount",
    "oldBalInitiator",
    "oldBalRecipient",
]

print("MoMTSim V2 — Transfer Feature Profiles by Split")
print("=" * 80)

for split_name, path in SPLITS.items():

    df = pd.read_csv(
        path,
        usecols=[
            "transactionType",
            "isFraud",
            *FEATURES,
        ],
    )

    transfers = df[df["transactionType"] == "TRANSFER"].copy()
    del df

    print(f"\n{split_name}")
    print("-" * 80)
    print(f"Transfer transactions: {len(transfers):,}")

    for label, label_name in [
        (0, "LEGITIMATE"),
        (1, "FRAUD"),
    ]:
        subset = transfers[transfers["isFraud"] == label]

        print(f"\n{label_name}: {len(subset):,} transactions")

        for feature in FEATURES:
            values = subset[feature]

            print(f"\n  {feature}")
            print(f"    Mean:   {values.mean():,.2f}")
            print(f"    Median: {values.median():,.2f}")
            print(f"    P90:    {values.quantile(0.90):,.2f}")
            print(f"    P95:    {values.quantile(0.95):,.2f}")
            print(f"    P99:    {values.quantile(0.99):,.2f}")
            print(f"    Min:    {values.min():,.2f}")
            print(f"    Max:    {values.max():,.2f}")
            print(f"    Zero:   {(values == 0).sum():,}")
            print(f"    Missing:{values.isna().sum():,}")

    del transfers

print("\nFeature profiling completed.")
