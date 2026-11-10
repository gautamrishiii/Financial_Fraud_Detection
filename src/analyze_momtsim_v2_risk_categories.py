
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

# Map the four learned leaves into three risk categories.
risk_mapping = {
    2: "LOW",
    3: "HIGH",
    5: "MEDIUM",
    6: "LOW",
}

def build_risk_summary(X, y, split_name):
    leaf_ids = model.apply(X)

    result = pd.DataFrame({
        "leaf": leaf_ids,
        "risk_category": [
            risk_mapping[int(leaf)] for leaf in leaf_ids
        ],
        "isFraud": y.to_numpy(),
    })

    summary = (
        result.groupby("risk_category")["isFraud"]
        .agg(
            transactions="count",
            fraud="sum",
        )
        .reset_index()
    )

    summary["legitimate"] = (
        summary["transactions"] - summary["fraud"]
    )

    summary["fraud_rate_pct"] = (
        summary["fraud"] / summary["transactions"] * 100
    )

    summary["split"] = split_name

    return summary

train_summary = build_risk_summary(
    X_train, y_train, "TRAIN"
)

validation_summary = build_risk_summary(
    X_val, y_val, "VALIDATION"
)

results = pd.concat(
    [train_summary, validation_summary],
    ignore_index=True,
)

category_order = ["LOW", "MEDIUM", "HIGH"]

results["risk_category"] = pd.Categorical(
    results["risk_category"],
    categories=category_order,
    ordered=True,
)

results = results.sort_values(
    ["split", "risk_category"]
)

print("MoMTSim V2 — Three-Level Risk Categorization")
print("=" * 90)

print(
    results[
        [
            "split",
            "risk_category",
            "transactions",
            "fraud",
            "legitimate",
            "fraud_rate_pct",
        ]
    ].to_string(
        index=False,
        formatters={
            "fraud_rate_pct": "{:.3f}%".format,
        },
    )
)

results.to_csv(
    "eda_outputs/momtsim_v2_risk_categories.csv",
    index=False,
)

print("\nSaved: eda_outputs/momtsim_v2_risk_categories.csv")
print("Risk categorization completed.")
