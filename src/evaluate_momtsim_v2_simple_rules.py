import pandas as pd
from pathlib import Path

SPLITS = {
    "TRAIN": Path("eda_outputs/momtsim_v2_train.csv"),
    "VALIDATION": Path("eda_outputs/momtsim_v2_validation.csv"),
    "TEST": Path("eda_outputs/momtsim_v2_test.csv"),
}

RULES = {
    "amount_below_1000": lambda df: df["amount"] < 1_000,
    "amount_below_10000": lambda df: df["amount"] < 10_000,
    "initiator_balance_below_1000": (
        lambda df: df["oldBalInitiator"] < 1_000
    ),
    "initiator_balance_below_10000": (
        lambda df: df["oldBalInitiator"] < 10_000
    ),
    "both_amount_and_balance_below_1000": (
        lambda df: (
            (df["amount"] < 1_000)
            & (df["oldBalInitiator"] < 1_000)
        )
    ),
}

print("MoMTSim V2 — Simple Rule Evaluation")
print("=" * 85)

for split_name, path in SPLITS.items():

    df = pd.read_csv(
        path,
        usecols=[
            "transactionType",
            "amount",
            "oldBalInitiator",
            "isFraud",
        ],
    )

    df = df[df["transactionType"] == "TRANSFER"]

    actual = df["isFraud"] == 1
    total_fraud = int(actual.sum())
    total_legitimate = len(df) - total_fraud

    print(f"\n{split_name}")
    print("-" * 85)
    print(f"Transfers:   {len(df):,}")
    print(f"Fraud:       {total_fraud:,}")
    print(f"Legitimate:  {total_legitimate:,}")

    for rule_name, rule in RULES.items():

        predicted = rule(df)

        tp = int((predicted & actual).sum())
        fp = int((predicted & ~actual).sum())
        fn = int((~predicted & actual).sum())
        tn = int((~predicted & ~actual).sum())

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / total_fraud if total_fraud else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        print(f"\n  {rule_name}")
        print(f"    TP:        {tp:,}")
        print(f"    FP:        {fp:,}")
        print(f"    FN:        {fn:,}")
        print(f"    TN:        {tn:,}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1:        {f1:.4f}")
        print(f"    Flag rate: {predicted.mean() * 100:.2f}%")

    del df

print("\nRule evaluation completed.")
print("These are exploratory rules, not production recommendations.")