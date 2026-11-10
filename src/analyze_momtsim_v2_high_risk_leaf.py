
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

FEATURE = "oldBalInitiator"

CANDIDATE_FEATURES = [
    "amount",
    "oldBalRecipient",
    "step",
]

train = pd.read_csv(
    "eda_outputs/momtsim_v2_train.csv",
    usecols=[
        "transactionType",
        FEATURE,
        *CANDIDATE_FEATURES,
        "isFraud",
    ],
)

validation = pd.read_csv(
    "eda_outputs/momtsim_v2_validation.csv",
    usecols=[
        "transactionType",
        FEATURE,
        *CANDIDATE_FEATURES,
    ],
)

train = train[
    train["transactionType"] == "TRANSFER"
].copy()

validation = validation[
    validation["transactionType"] == "TRANSFER"
].copy()

X_train = train[[FEATURE]]
y_train = train["isFraud"]

model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=500,
    random_state=42,
)

model.fit(X_train, y_train)

# Identify the highest-scoring leaf using the training-fitted model.
train_scores = model.predict_proba(X_train)[:, 1]
validation_scores = model.predict_proba(
    validation[[FEATURE]]
)[:, 1]

highest_score = train_scores.max()

train_high = train.loc[
    train_scores == highest_score
].copy()

validation_high = validation.loc[
    validation_scores == highest_score
].copy()

print("MoMTSim V2 — Highest-Risk Leaf Feature Investigation")
print("=" * 90)

print(f"\nHighest predicted score: {highest_score:.6f}")

print("\nTraining leaf:")
print("Transactions:", len(train_high))
print("Fraud:", int(train_high["isFraud"].sum()))
print("Legitimate:", int((train_high["isFraud"] == 0).sum()))

print("\nTraining feature distributions by label:")
print("-" * 90)

for feature in CANDIDATE_FEATURES:

    print(f"\nFeature: {feature}")

    summary = train_high.groupby("isFraud")[feature].agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    )

    summary["p90"] = train_high.groupby("isFraud")[
        feature
    ].quantile(0.90)

    summary["p99"] = train_high.groupby("isFraud")[
        feature
    ].quantile(0.99)

    print(summary.to_string(float_format=lambda x: f"{x:,.2f}"))

print("\nValidation highest-scoring leaf:")
print("Transactions:", len(validation_high))

print("\nValidation feature distributions (labels not used):")
print("-" * 90)

for feature in CANDIDATE_FEATURES:

    print(f"\nFeature: {feature}")

    print(
        validation_high[feature]
        .describe(
            percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
        )
        .to_string(float_format=lambda x: f"{x:,.2f}")
    )

print("\nFeature investigation completed.")
