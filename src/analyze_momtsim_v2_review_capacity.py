
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

FEATURE = "oldBalInitiator"

train = pd.read_csv(
    "eda_outputs/momtsim_v2_train.csv",
    usecols=["transactionType", FEATURE, "isFraud"],
)

validation = pd.read_csv(
    "eda_outputs/momtsim_v2_validation.csv",
    usecols=["transactionType", FEATURE, "isFraud"],
)

train = train[train["transactionType"] == "TRANSFER"].copy()
validation = validation[
    validation["transactionType"] == "TRANSFER"
].copy()

X_train = train[[FEATURE]]
y_train = train["isFraud"]

X_val = validation[[FEATURE]]
y_val = validation["isFraud"]

model = DecisionTreeClassifier(
    max_depth=2,
    min_samples_leaf=500,
    random_state=42,
)

model.fit(X_train, y_train)

risk_mapping = {
    2: "LOW",
    3: "HIGH",
    5: "MEDIUM",
    6: "LOW",
}

policies = {
    "HIGH_ONLY": {"HIGH"},
    "MEDIUM_AND_HIGH": {"MEDIUM", "HIGH"},
}

def evaluate_policy(X, y, split_name):
    leaves = model.apply(X)

    categories = [
        risk_mapping[int(leaf)] for leaf in leaves
    ]

    rows = []

    for policy_name, selected_categories in policies.items():
        reviewed = pd.Series(
            [category in selected_categories for category in categories],
            index=y.index,
        )

        tp = int(((reviewed) & (y == 1)).sum())
        fp = int(((reviewed) & (y == 0)).sum())
        fn = int(((~reviewed) & (y == 1)).sum())
        tn = int(((~reviewed) & (y == 0)).sum())

        total = len(y)
        alerts = tp + fp

        precision = tp / alerts if alerts else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        review_rate = alerts / total if total else 0

        rows.append({
            "split": split_name,
            "policy": policy_name,
            "transactions": total,
            "reviewed": alerts,
            "review_rate_pct": review_rate * 100,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": precision,
            "recall": recall,
        })

    return rows

results = []

results.extend(
    evaluate_policy(X_train, y_train, "TRAIN")
)

results.extend(
    evaluate_policy(X_val, y_val, "VALIDATION")
)

results_df = pd.DataFrame(results)

print("MoMTSim V2 — Review Capacity Analysis")
print("=" * 110)

print(
    results_df.to_string(
        index=False,
        formatters={
            "review_rate_pct": "{:.3f}%".format,
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format,
        },
    )
)

results_df.to_csv(
    "eda_outputs/momtsim_v2_review_capacity.csv",
    index=False,
)

print("\nSaved: eda_outputs/momtsim_v2_review_capacity.csv")
print("Review capacity analysis completed.")
