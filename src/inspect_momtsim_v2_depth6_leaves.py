import pandas as pd

from sklearn.tree import DecisionTreeClassifier, export_text

FEATURE = "oldBalInitiator"

train = pd.read_csv(
    "eda_outputs/momtsim_v2_train.csv",
    usecols=["transactionType", FEATURE, "isFraud"],
)

train = train[train["transactionType"] == "TRANSFER"].copy()

X = train[[FEATURE]]
y = train["isFraud"]

model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=500,
    random_state=42,
)

model.fit(X, y)

print("MoMTSim V2 — Depth-6 Decision Tree Rules")
print("=" * 85)

print("\nDecision rules:")
print(export_text(model, feature_names=[FEATURE], decimals=2))

print("\nLeaf-level training statistics:")
print("-" * 85)

leaf_ids = model.apply(X)

leaf_data = pd.DataFrame({
    "leaf": leaf_ids,
    "isFraud": y.to_numpy(),
})

summary = (
    leaf_data.groupby("leaf")["isFraud"]
    .agg(
        transactions="count",
        fraud="sum",
    )
    .reset_index()
)

summary["legitimate"] = summary["transactions"] - summary["fraud"]

summary["fraud_rate_pct"] = (
    summary["fraud"] / summary["transactions"] * 100
)

summary = summary.sort_values("leaf")

for _, row in summary.iterrows():
    print(
        f"Leaf {int(row['leaf']):3d} | "
        f"Transactions: {int(row['transactions']):10,} | "
        f"Fraud: {int(row['fraud']):10,} | "
        f"Legitimate: {int(row['legitimate']):9,} | "
        f"Fraud rate: {row['fraud_rate_pct']:7.3f}%"
    )

print("\nTotal leaves:", model.get_n_leaves())
print("Total training transactions:", len(train))
print("\nLeaf inspection completed.")