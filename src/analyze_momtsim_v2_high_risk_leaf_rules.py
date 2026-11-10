
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

FEATURE = "oldBalInitiator"

train = pd.read_csv(
    "eda_outputs/momtsim_v2_train.csv",
    usecols=[
        "transactionType",
        FEATURE,
        "amount",
        "oldBalRecipient",
        "isFraud",
    ],
)

train = train[
    train["transactionType"] == "TRANSFER"
].copy()

X_train = train[[FEATURE]]
y_train = train["isFraud"]

model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=500,
    random_state=42,
)

model.fit(X_train, y_train)

scores = model.predict_proba(X_train)[:, 1]
highest_score = scores.max()

high_leaf = train.loc[
    scores == highest_score
].copy()

print("MoMTSim V2 — Highest-Risk Leaf Rule Diagnostics")
print("=" * 90)

print("\nLeaf size:", len(high_leaf))
print("Fraud:", int(high_leaf["isFraud"].sum()))
print("Legitimate:", int((high_leaf["isFraud"] == 0).sum()))

conditions = {
    "Recipient balance equals zero": (
        high_leaf["oldBalRecipient"] == 0
    ),
    "Amount greater than 10,000": (
        high_leaf["amount"] > 10_000
    ),
    "Amount greater than 100,000": (
        high_leaf["amount"] > 100_000
    ),
    "Amount greater than 143,656.73": (
        high_leaf["amount"] > 143_656.73
    ),
}

print("\nCandidate conditions")
print("-" * 90)

for name, condition in conditions.items():

    selected = high_leaf.loc[condition]

    legitimate = int((selected["isFraud"] == 0).sum())
    fraud = int(selected["isFraud"].sum())

    print(f"\n{name}")
    print("Transactions satisfying condition:", len(selected))
    print("Legitimate:", legitimate)
    print("Fraud:", fraud)

print("\nCombined condition")
print("-" * 90)

combined = (
    (high_leaf["oldBalRecipient"] == 0)
    | (high_leaf["amount"] > 143_656.73)
)

selected = high_leaf.loc[combined]

print("Transactions satisfying either condition:", len(selected))
print("Legitimate:", int((selected["isFraud"] == 0).sum()))
print("Fraud:", int(selected["isFraud"].sum()))

print("\nTraining-leaf feature ranges")
print("-" * 90)

for label, group in high_leaf.groupby("isFraud"):

    print(
        "Legitimate" if label == 0 else "Fraud",
        "| amount min:", group["amount"].min(),
        "| amount max:", group["amount"].max(),
        "| recipient balance zero:",
        int((group["oldBalRecipient"] == 0).sum()),
    )

print("\nRule diagnostics completed.")
