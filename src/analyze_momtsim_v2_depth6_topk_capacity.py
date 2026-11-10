
import math
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
    max_depth=6,
    min_samples_leaf=500,
    random_state=42,
)

model.fit(X_train, y_train)

scores = model.predict_proba(X_val)[:, 1]

score_data = pd.DataFrame({
    "score": scores,
    "fraud": y_val.to_numpy(),
})

score_groups = (
    score_data.groupby("score")
    .agg(
        transactions=("fraud", "count"),
        fraud=("fraud", "sum"),
    )
    .sort_index(ascending=False)
)

score_groups["legitimate"] = (
    score_groups["transactions"] - score_groups["fraud"]
)

print("MoMTSim V2 — Depth-6 Validation Score Distribution")
print("=" * 85)

print("Number of distinct scores:", len(score_groups))
print(score_groups.to_string())

print("\nDepth-6 Top-K Review Capacity Analysis")
print("=" * 110)

total = len(validation)
total_fraud = int(y_val.sum())
total_legitimate = total - total_fraud

results = []

for capacity_pct in [1, 2, 5, 10]:

    target = math.ceil(total * capacity_pct / 100)

    remaining = target
    expected_tp = 0.0
    expected_fp = 0.0

    cutoff_score = None
    cutoff_group_size = 0
    selected_from_cutoff = 0

    for score, row in score_groups.iterrows():

        group_size = int(row["transactions"])
        group_fraud = int(row["fraud"])
        group_legitimate = int(row["legitimate"])

        take = min(remaining, group_size)

        if take <= 0:
            break

        if take < group_size:
            cutoff_score = score
            cutoff_group_size = group_size
            selected_from_cutoff = take

        expected_tp += take * group_fraud / group_size
        expected_fp += take * group_legitimate / group_size

        remaining -= take

    expected_fn = total_fraud - expected_tp
    expected_tn = total_legitimate - expected_fp

    precision = expected_tp / target
    recall = expected_tp / total_fraud

    results.append({
        "capacity_pct": capacity_pct,
        "reviewed": target,
        "cutoff_score": cutoff_score,
        "cutoff_tie_group": cutoff_group_size,
        "selected_from_tie": selected_from_cutoff,
        "expected_tp": expected_tp,
        "expected_fp": expected_fp,
        "expected_fn": expected_fn,
        "expected_tn": expected_tn,
        "expected_precision": precision,
        "expected_recall": recall,
    })

results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False,
        formatters={
            "cutoff_score": lambda x: (
                f"{x:.6f}" if pd.notna(x) else "N/A"
            ),
            "expected_tp": "{:.1f}".format,
            "expected_fp": "{:.1f}".format,
            "expected_fn": "{:.1f}".format,
            "expected_tn": "{:.1f}".format,
            "expected_precision": "{:.4f}".format,
            "expected_recall": "{:.4f}".format,
        },
    )
)

results_df.to_csv(
    "eda_outputs/momtsim_v2_depth6_topk_capacity.csv",
    index=False,
)

print("\nSaved: eda_outputs/momtsim_v2_depth6_topk_capacity.csv")
print("Depth-6 capacity analysis completed.")
