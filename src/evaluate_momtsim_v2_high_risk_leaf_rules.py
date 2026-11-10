
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

FEATURE = "oldBalInitiator"
AMOUNT_THRESHOLD = 143_656.73

train = pd.read_csv(
    "eda_outputs/momtsim_v2_train.csv",
    usecols=["transactionType", FEATURE, "isFraud"],
)

validation = pd.read_csv(
    "eda_outputs/momtsim_v2_validation.csv",
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

validation = validation[
    validation["transactionType"] == "TRANSFER"
].copy()

model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=500,
    random_state=42,
)

model.fit(train[[FEATURE]], train["isFraud"])

# Identify the highest-scoring leaf using the training-fitted model.
highest_score = model.predict_proba(
    train[[FEATURE]]
)[:, 1].max()

validation_scores = model.predict_proba(
    validation[[FEATURE]]
)[:, 1]

high_leaf = validation.loc[
    validation_scores == highest_score
].copy()

conditions = {
    "Recipient balance equals zero": (
        high_leaf["oldBalRecipient"] == 0
    ),
    "Amount greater than 100,000": (
        high_leaf["amount"] > 100_000
    ),
    "Amount greater than 143,656.73": (
        high_leaf["amount"] > AMOUNT_THRESHOLD
    ),
    "Combined candidate condition": (
        (high_leaf["oldBalRecipient"] == 0)
        | (high_leaf["amount"] > AMOUNT_THRESHOLD)
    ),
}

total_legitimate = int((high_leaf["isFraud"] == 0).sum())
total_fraud = int(high_leaf["isFraud"].sum())

print("MoMTSim V2 — High-Risk Leaf Rule Validation")
print("=" * 100)

print("\nValidation high-risk leaf:")
print("Transactions:", len(high_leaf))
print("Legitimate:", total_legitimate)
print("Fraud:", total_fraud)

print("\nFixed training-derived conditions")
print("-" * 100)

results = []

for name, condition in conditions.items():

    selected = high_leaf.loc[condition]

    legitimate_captured = int(
        (selected["isFraud"] == 0).sum()
    )

    fraud_included = int(
        selected["isFraud"].sum()
    )

    precision_for_legitimate = (
        legitimate_captured / len(selected)
        if len(selected) else 0
    )

    legitimate_recall = (
        legitimate_captured / total_legitimate
        if total_legitimate else 0
    )

    results.append({
        "condition": name,
        "selected": len(selected),
        "legitimate_captured": legitimate_captured,
        "fraud_included": fraud_included,
        "legitimate_precision": precision_for_legitimate,
        "legitimate_recall": legitimate_recall,
    })

    print(f"\n{name}")
    print("Selected:", len(selected))
    print("Legitimate captured:", legitimate_captured)
    print("Fraud included:", fraud_included)
    print(f"Legitimate precision: {precision_for_legitimate:.6f}")
    print(f"Legitimate recall: {legitimate_recall:.6f}")

results_df = pd.DataFrame(results)

results_df.to_csv(
    "eda_outputs/momtsim_v2_high_risk_leaf_rule_validation.csv",
    index=False,
)

print("\nSaved:")
print("eda_outputs/momtsim_v2_high_risk_leaf_rule_validation.csv")

print("\nValidation rule evaluation completed.")
