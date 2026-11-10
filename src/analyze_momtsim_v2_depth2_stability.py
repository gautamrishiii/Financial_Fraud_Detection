
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, export_text

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
validation = validation[validation["transactionType"] == "TRANSFER"].copy()

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

print("MoMTSim V2 — Depth-2 Tree Stability")
print("=" * 85)

print("\nLearned decision rules:")
print(export_text(model, feature_names=[FEATURE], decimals=2))

def summarize_leaves(leaf_ids, labels, prefix):
    data = pd.DataFrame({
        "leaf": leaf_ids,
        "fraud": labels.to_numpy(),
    })

    result = (
        data.groupby("leaf")["fraud"]
        .agg(
            transactions="count",
            fraud="sum",
        )
        .reset_index()
    )

    result[f"{prefix}_legitimate"] = (
        result["transactions"] - result["fraud"]
    )

    result[f"{prefix}_fraud_rate_pct"] = (
        result["fraud"] / result["transactions"] * 100
    )

    return result.rename(columns={
        "transactions": f"{prefix}_transactions",
        "fraud": f"{prefix}_fraud",
    })

train_summary = summarize_leaves(
    model.apply(X_train),
    y_train,
    "train",
)

validation_summary = summarize_leaves(
    model.apply(X_val),
    y_val,
    "validation",
)

comparison = train_summary.merge(
    validation_summary,
    on="leaf",
    how="outer",
).fillna(0)

comparison = comparison.sort_values("leaf")

print("\nTraining versus validation leaf statistics:")
print("-" * 85)

columns = [
    "leaf",
    "train_transactions",
    "train_fraud_rate_pct",
    "validation_transactions",
    "validation_fraud_rate_pct",
]

print(
    comparison[columns].to_string(
        index=False,
        formatters={
            "train_fraud_rate_pct": "{:.3f}%".format,
            "validation_fraud_rate_pct": "{:.3f}%".format,
        },
    )
)

comparison.to_csv(
    "eda_outputs/momtsim_v2_depth2_leaf_stability.csv",
    index=False,
)

print("\nNumber of leaves:", model.get_n_leaves())
print("Training transfers:", len(train))
print("Validation transfers:", len(validation))
print("\nSaved: eda_outputs/momtsim_v2_depth2_leaf_stability.csv")
