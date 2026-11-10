# MoMTSim V1 — Initial Data Audit

## 1. Dataset Overview

- Dataset: MoMTSim Version 1
- File: `synthetic_mobile_money_transaction_dataset.csv`
- Records: 1,720,181
- Columns: 10
- Fraud target: `isFraud`
- Transaction time field: `step`
- Transaction types: DEBIT, DEPOSIT, PAYMENT, TRANSFER, WITHDRAWAL

## 2. Label Distribution

| Label | Count | Percentage |
|---|---:|---:|
| Legitimate (0) | 1,544,663 | 89.80% |
| Fraud (1) | 175,518 | 10.20% |
| Total | 1,720,181 | 100.00% |

These figures describe the synthetic dataset, not real-world fraud prevalence.

## 3. Structural and Data Quality Checks

- Expected columns: 10
- Missing or blank cells: None detected
- Malformed CSV rows: 0
- Exact duplicate records: 0
- Unexpected transaction types: 0
- Unexpected target labels: 0

These checks do not establish that every transaction is logically realistic.

## 4. Transaction Type and Fraud Relationship

All 175,518 fraud-labelled records belong to the TRANSFER category.

No fraud-labelled records were found in DEBIT, DEPOSIT, PAYMENT, or WITHDRAWAL.

This creates a substantial risk of simulator-specific shortcut learning. Model performance must be investigated on TRANSFER-only transactions as well as on the complete dataset.

## 5. Chronological Evaluation Split

The CSV is ordered by non-decreasing simulation step, with all transactions from each step contiguous.

The proposed chronological split is:

| Partition | Steps | Transactions | Fraud | Fraud Rate |
|---|---|---:|---:|---:|
| Training | 0–116 | 1,377,258 | 142,814 | 10.37% |
| Testing | 117–143 | 342,923 | 32,704 | 9.54% |

The split preserves complete simulation steps and approximates an 80/20 transaction split.

## 6. Amount Distribution Findings

Fraudulent transfer amounts have similar median and upper-percentile values across training and testing.

Legitimate transfer amounts show a higher upper tail in testing, particularly at the 95th percentile.

These are synthetic-data distribution patterns, not real-world fraud rules.

## 7. Balance Findings

Initiator and recipient balance-match rates differ between legitimate and fraudulent transfers.

The difference is smaller in the testing period than in the training period.

Negative initiator balances occur in both classes. Negative balances must not automatically be treated as errors or fraud indicators.

## 8. Account Identifier Findings

- Unique initiators: 13,462
- Unique recipients: 16,731
- Initiator identifiers have 16 characters.
- Recipient identifiers have lengths of 10, 11, 15, or 16 characters.
- Cross-role identifier overlap: 6,174.
- Training/testing identifier overlap is high.

Raw initiator and recipient identifiers will be excluded from the initial model to reduce memorization risk.

## 9. Initial Feature Policy

Candidate features:

- `step`
- `transactionType`
- `amount`
- Balance fields and carefully derived balance features

Initially excluded:

- `initiator`
- `recipient`

All transformations must be fitted using training data only. Testing data must not be used for model fitting, feature selection, or threshold tuning.

## 10. Limitations and Research Cautions

1. MoMTSim is synthetic data.
2. Fraud labels are exclusively associated with TRANSFER transactions.
3. Amount distributions may reflect simulator-generation rules.
4. Account reuse does not automatically imply leakage.
5. Chronological evaluation is preferable to relying exclusively on random splitting.
6. Full-dataset model performance must be compared with TRANSFER-only performance.
7. No results should be presented as real-world banking performance.

## 11. Audit Status

Initial structural and exploratory audit completed.

Further feature validation, modelling, and sensitivity analysis remain necessary.

## Balance-change redundancy investigation

A row-level comparison of the initiator and recipient balance changes was conducted on all 569,328 TRANSFER transactions.

Findings:

- 55,425 transactions had exactly matching balance changes.
- 513,903 transactions had small numerical differences.
- The maximum absolute difference was approximately 0.01.
- Seven transactions failed NumPy's default `isclose` comparison.
- All seven mismatches were legitimate transactions.
- Each mismatch showed an absolute difference of approximately 0.01.

The two balance-change features had a Pearson correlation displayed as 1.00000000, but they were not exactly equal.

Interpretation:

These findings suggest a very strong linear relationship with small differences between the two balance-change features. The exact cause of the seven one-cent discrepancies has not been established.

Decision:

- Preserve the original balance values and numerical precision.
- Do not manually correct or round the observations.
- Retain the derived features as experimental candidates.
- Reassess redundancy during model experiments.
- Do not interpret the one-cent differences as evidence of fraud.

This investigation is descriptive and does not establish the simulator's internal accounting or rounding mechanism.


## Chronological Split and Per-Step Class Distribution

### Three-way chronological split

MoMTSim V1 was divided into three chronological subsets, keeping complete simulation steps together.

| Split | Steps | Transactions | Legitimate | Fraud | Fraud rate |
|---|---|---:|---:|---:|---:|
| Training | 0–92 | 1,161,938 | 1,048,122 | 113,816 | 9.80% |
| Validation | 93–116 | 215,320 | 186,322 | 28,998 | 13.47% |
| Testing | 117–143 | 342,923 | 310,219 | 32,704 | 9.54% |

All 1,720,181 transactions were accounted for, with no overlap between chronological boundaries.

### Per-step fraud-rate variation

The unweighted average of per-step fraud rates was:

| Split | Average per-step fraud rate |
|---|---:|
| Training | 12.33% |
| Validation | 14.12% |
| Testing | 11.41% |

These averages differ from the overall fraud rates because individual simulation steps contain different numbers of transactions.

### Per-step class-count analysis

| Split | Mean fraud count per step | Mean legitimate count per step |
|---|---:|---:|
| Training | 1,223.83 | 11,270.13 |
| Validation | 1,208.25 | 7,763.42 |
| Testing | 1,211.26 | 11,489.59 |

Fraud counts per step are comparatively stable, while legitimate transaction counts vary substantially.

This helps explain the higher overall fraud rate in the validation period. The observed pattern is consistent with changes in legitimate transaction volume rather than a comparable increase in fraud counts.

The precise mechanism behind this pattern has not been established.

### Evaluation implications

- Preserve the chronological training, validation, and testing split.
- Do not artificially balance validation or test data.
- Report the class distribution of every evaluation subset.
- Interpret validation metrics in light of the observed distribution differences.
- Keep the final test set untouched during model selection and threshold tuning.

**Conclusion:** The chronological split is retained because it provides a more realistic temporal evaluation than a random split for this simulation dataset, although it does not establish generalization to real-world financial transactions.
