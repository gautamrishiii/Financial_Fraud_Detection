# Cross-Dataset Compatibility Audit

## 1. Objective

Evaluate whether MoMTSim V1, MoMTSim V2, PaySim, and IEEE-CIS can be directly combined or compared for financial fraud detection.

The audit examines dataset sizes, label distributions, schemas, missing values, identity-data availability, and compatibility.

## 2. Dataset Overview

| Dataset | Transactions | Fraud Cases | Fraud Rate | Domain |
|---|---:|---:|---:|---|
| MoMTSim V1 | 1,720,181 | 175,518 | 10.2035% | Mobile money |
| MoMTSim V2 | 4,225,958 | 2,233,118 | 52.8429% | Mobile money |
| PaySim | 6,362,620 | 8,213 | 0.1291% | Mobile money |
| IEEE-CIS Train | 590,540 | 20,663 | 3.4990% | Card transactions |
| IEEE-CIS Test | 506,691 | Unknown | Unknown | Card transactions |

IEEE-CIS test data has no fraud labels and must not be used for supervised evaluation.

## 3. Schema Compatibility

### MoMTSim

Both versions contain 10 columns:

- step
- transactionType
- amount
- initiator
- oldBalInitiator
- newBalInitiator
- recipient
- oldBalRecipient
- newBalRecipient
- isFraud

### PaySim

PaySim contains 11 columns, including transaction identifiers, transaction type, amount, account balances, `isFraud`, and `isFlaggedFraud`.

The existing `isFlaggedFraud` field must not be used as a predictive feature because it represents an existing fraud-detection flag.

### IEEE-CIS

- Training transaction table: 394 columns, including `isFraud`.
- Test transaction table: 393 columns.
- Both transaction feature schemas match exactly.
- Training and test identity tables each contain 41 columns.
- Identity feature names require normalization: test names such as `id-01` must become `id_01`.

The target `isFraud` must be excluded from model features.

## 4. IEEE-CIS Identity Data

### Identity coverage

| Split | Transactions | Identity Records | Transactions Without Identity |
|---|---:|---:|---:|
| Train | 590,540 | 144,233 | 446,307 |
| Test | 506,691 | 141,907 | 364,784 |

Every identity record matches a transaction. No duplicate identity transaction IDs or unmatched identity records were found.

The transaction table should be left-joined with the identity table to preserve every transaction.

### Identity feature quality

There are 40 identity features excluding `TransactionID`:

- 23 numeric features.
- 17 string features.
- 38 of the 40 features contain missing values.
- Several identity features have more than 96% missing values.

Missingness should not automatically cause rows or columns to be removed. Missing-value indicators and identity availability may carry predictive information.

### Identity availability and fraud

| Group | Transactions | Fraud Cases | Fraud Rate |
|---|---:|---:|---:|
| No identity | 446,307 | 9,345 | 2.0939% |
| Has identity | 144,233 | 11,318 | 7.8470% |

Transactions with identity records have approximately 3.75 times the fraud prevalence of transactions without identity records.

This is an association, not evidence of causation. Identity availability may reflect transaction collection or enrichment processes.

The `has_identity` indicator should be retained as a candidate feature and evaluated using training data only.

## 5. Important Differences Between Datasets

### Fraud prevalence

Fraud rates vary substantially:

- PaySim: 0.1291%
- IEEE-CIS: 3.4990%
- MoMTSim V1: 10.2035%
- MoMTSim V2: 52.8429%

Accuracy alone is unsuitable for comparing these datasets because of their different class distributions.

Use metrics such as:

- Precision
- Recall
- F1-score
- PR-AUC
- ROC-AUC
- Precision at a defined review capacity

### Different transaction domains

MoMTSim and PaySim represent mobile-money transactions.

IEEE-CIS represents card transaction fraud with substantially different feature structures, missingness patterns, and transaction behavior.

Their raw records should not be concatenated into one training table.

### Missingness

MoMTSim and PaySim have no missing values in the audited transaction columns.

IEEE-CIS has extensive missingness, especially in its V-feature and identity-feature groups.

IEEE-CIS therefore requires its own preprocessing pipeline.

## 6. Recommended Project Architecture

Use separate dataset-specific pipelines:

1. MoMTSim V1 pipeline.
2. MoMTSim V2 pipeline.
3. PaySim pipeline.
4. IEEE-CIS pipeline.

Each pipeline should have its own:

- Data validation.
- Feature engineering.
- Missing-value handling.
- Train/validation/test strategy.
- Model training.
- Evaluation report.

A common application interface can later present results from these separate models.

Do not combine raw datasets merely because they share columns such as amount or transaction type.

## 7. Evaluation and Leakage Precautions

- Exclude post-transaction balance fields when they would not be available at prediction time.
- Exclude raw account identifiers unless a justified, leakage-safe feature-engineering strategy is developed.
- Exclude `isFraud` from all predictive features.
- Exclude PaySim's `isFlaggedFraud` from model features.
- Fit preprocessing steps using training data only.
- Preserve chronological order where transaction timestamps or steps support temporal validation.
- Do not treat an already-examined test split as an untouched final holdout.

## 8. Audit Conclusion

The four datasets are useful for studying different financial fraud scenarios, but they are not directly interchangeable.

The recommended strategy is to build and evaluate separate models for each dataset and transaction domain, then integrate their outputs through a common application interface.

The IEEE-CIS identity audit also identifies identity availability as a potentially informative feature that deserves further investigation.