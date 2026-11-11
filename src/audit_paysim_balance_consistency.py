from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(
    "dataset/raw/paysim/PS_20174392719_1491204439457_log.csv"
)

CHUNK_SIZE = 200_000


def main():
    print("PaySim — Balance Consistency Audit")
    print("=" * 70)

    transaction_types = [
        "CASH_OUT",
        "TRANSFER",
        "CASH_IN",
        "PAYMENT",
        "DEBIT",
    ]

    columns = [
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    ]

    summaries = []

    for chunk in pd.read_csv(
        DATA_FILE,
        usecols=columns,
        chunksize=CHUNK_SIZE,
    ):
        chunk["origin_balance_change"] = (
            chunk["oldbalanceOrg"] - chunk["newbalanceOrig"]
        )

        chunk["destination_balance_change"] = (
            chunk["newbalanceDest"] - chunk["oldbalanceDest"]
        )

        chunk["origin_expected_change"] = (
            chunk["amount"] - chunk["origin_balance_change"]
        )

        chunk["destination_expected_change"] = (
            chunk["amount"] - chunk["destination_balance_change"]
        )

        grouped = chunk.groupby(["type", "isFraud"]).agg(
            transactions=("amount", "size"),
            mean_amount=("amount", "mean"),
            origin_balance_mismatch=(
                "origin_expected_change",
                lambda x: (np.abs(x) > 0.01).mean() * 100,
            ),
            destination_balance_mismatch=(
                "destination_expected_change",
                lambda x: (np.abs(x) > 0.01).mean() * 100,
            ),
        ).reset_index()

        summaries.append(grouped)

    combined = pd.concat(summaries, ignore_index=True)

    result = (
        combined.groupby(["type", "isFraud"])
        .apply(
            lambda group: pd.Series({
                "transactions": group["transactions"].sum(),
                "mean_amount": np.average(
                    group["mean_amount"],
                    weights=group["transactions"],
                ),
                "origin_mismatch_pct": np.average(
                    group["origin_balance_mismatch"],
                    weights=group["transactions"],
                ),
                "destination_mismatch_pct": np.average(
                    group["destination_balance_mismatch"],
                    weights=group["transactions"],
                ),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    print(result.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))


if __name__ == "__main__":
    main()