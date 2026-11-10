
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "eda_outputs"
    / "momtsim_v1_engineered.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "eda_outputs"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "momtsim_v1_feature_correlations.csv"
)


# --------------------------------------------------
# 2. Load engineered data
# --------------------------------------------------

print("MoMTSim V1 — Feature Correlation Analysis")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"Dataset shape: {df.shape}")


# --------------------------------------------------
# 3. Select numeric model features
# --------------------------------------------------

feature_columns = [
    "amount",
    "step",
    "oldBalInitiator",
    "newBalInitiator",
    "oldBalRecipient",
    "newBalRecipient",
    "initiator_balance_change",
    "recipient_balance_change",
    "initiator_balance_difference",
    "recipient_balance_difference",
]

X = df[feature_columns]


# --------------------------------------------------
# 4. Calculate Pearson correlations
# --------------------------------------------------

correlation_matrix = X.corr(method="pearson")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

correlation_matrix.to_csv(OUTPUT_FILE)


# --------------------------------------------------
# 5. Display highly correlated feature pairs
# --------------------------------------------------

print("\nFeature pairs with absolute correlation >= 0.95:")

found_pair = False

for i in range(len(feature_columns)):
    for j in range(i + 1, len(feature_columns)):
        feature_a = feature_columns[i]
        feature_b = feature_columns[j]

        correlation = correlation_matrix.loc[
            feature_a, feature_b
        ]

        if abs(correlation) >= 0.95:
            found_pair = True

            print(
                f"{feature_a} <-> {feature_b}: "
                f"{correlation:.4f}"
            )

if not found_pair:
    print("No pairs met the threshold.")


# --------------------------------------------------
# 6. Display correlations with transaction amount
# --------------------------------------------------

print("\nCorrelation with transaction amount:")

amount_correlations = (
    correlation_matrix["amount"]
    .drop("amount")
    .sort_values(
        key=lambda values: values.abs(),
        ascending=False,
    )
)

for feature, correlation in amount_correlations.items():
    print(f"{feature}: {correlation:.4f}")


# --------------------------------------------------
# 7. Save results
# --------------------------------------------------

print(f"\nCorrelation matrix saved to: {OUTPUT_FILE}")
print("\nCorrelation analysis completed.")
