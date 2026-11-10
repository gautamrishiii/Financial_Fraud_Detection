
import pandas as pd

DATA_PATH = "dataset/raw/momtsim/MoMTSim_20240722202413_1000_dataset.csv"
CHUNK_SIZE = 250_000

amount_bins = [0, 1_000, 10_000, 100_000, 1_000_000, float("inf")]
amount_labels = [
    "0-1K",
    "1K-10K",
    "10K-100K",
    "100K-1M",
    "Above 1M",
]

balance_bins = [
    float("-inf"), 0, 1_000, 10_000, 100_000,
    1_000_000, 10_000_000, 100_000_000, float("inf")
]

balance_labels = [
    "Negative",
    "0-1K",
    "1K-10K",
    "10K-100K",
    "100K-1M",
    "1M-10M",
    "10M-100M",
    "Above 100M",
]

results = {
    "amount": pd.DataFrame(
        0, index=amount_labels, columns=["transactions", "fraud"]
    ),
    "oldBalInitiator": pd.DataFrame(
        0, index=balance_labels, columns=["transactions", "fraud"]
    ),
}

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=["transactionType", "amount", "oldBalInitiator", "isFraud"],
    chunksize=CHUNK_SIZE,
):
    transfers = chunk[chunk["transactionType"] == "TRANSFER"].copy()

    transfers["amount_band"] = pd.cut(
        transfers["amount"],
        bins=amount_bins,
        labels=amount_labels,
        right=False,
        include_lowest=True,
    )

    transfers["balance_band"] = pd.cut(
        transfers["oldBalInitiator"],
        bins=balance_bins,
        labels=balance_labels,
        right=False,
    )

    for feature, band_column in [
        ("amount", "amount_band"),
        ("oldBalInitiator", "balance_band"),
    ]:
        grouped = transfers.groupby(
            band_column, observed=False
        ).agg(
            transactions=("isFraud", "size"),
            fraud=("isFraud", "sum"),
        )

        results[feature] += grouped

for feature, title in [
    ("amount", "Transaction Amount"),
    ("oldBalInitiator", "Opening Initiator Balance"),
]:
    result = results[feature].copy()
    result["fraud_rate_pct"] = (
        result["fraud"] / result["transactions"] * 100
    )

    print("\n" + "=" * 75)
    print(f"Fraud Rate by {title} — TRANSFER Transactions")
    print("=" * 75)
    print(result.to_string())
