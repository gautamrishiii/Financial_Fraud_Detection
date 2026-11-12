# MoMTSim V1 — Model Input Policy

## 1. Objective

Define model inputs for a pre-transaction, real-time financial fraud detection scenario.

The model must evaluate a transaction using information reasonably available before the transaction is authorized or executed.

Post-transaction information must not enter the initial model.

## 2. Detection scenario

**Selected scenario:** Pre-transaction real-time fraud detection.

The intended workflow is:

1. A transaction request is received.
2. Transaction details and eligible account information are retrieved.
3. The fraud detection model evaluates the transaction.
4. A risk prediction is generated before authorization.
5. The transaction can then be reviewed, approved, or declined according to the surrounding decision system.

This is a research simulation, not a production banking integration.

## 3. Target variable

- Target: `isFraud`
- Legitimate transaction: `0`
- Fraudulent transaction: `1`

The target must never be included in model inputs.

## 4. Revised feature policy

### 4.1 Request-time candidate features

| Feature | Decision | Explanation |
|---|---|---|
| `amount` | Include | Transaction amount is part of the request |
| `transactionType` | Include experimentally | Transaction category is known from the request |
| `step` | Experimental | Simulation time index; not necessarily equivalent to a production timestamp |

The `step` feature must be evaluated separately because it may encode simulator-specific temporal patterns.

### 4.2 Conditional pre-transaction account features

| Feature | Decision | Assumption |
|---|---|---|
| `oldBalInitiator` | Conditional inclusion | Current initiator balance is available before authorization |
| `oldBalRecipient` | Conditional inclusion | Current recipient balance is available before authorization |

These features may be included only under the assumption that the corresponding balance lookups happen before the fraud decision.

Their availability must be stated as a modelling assumption, not presented as a verified production-system capability.

### 4.3 Excluded post-transaction features

The following features must be excluded:

- `newBalInitiator`
- `newBalRecipient`
- `initiator_balance_change`
- `recipient_balance_change`
- `initiator_balance_difference`
- `recipient_balance_difference`

These features depend on updated balances or calculations involving updated balances.

They are not appropriate for the selected pre-transaction prediction scenario.

### 4.4 Excluded identifiers and target

- `initiator`
- `recipient`
- `isFraud` from the feature matrix

Raw account identifiers are excluded from initial experiments because they are high-cardinality identifiers and may encourage memorization.

The target is maintained separately as `y`.

## 5. Revised experimental feature groups

| Group | Features | Purpose |
|---|---|---|
| A — Request context | amount, transactionType | Request-time baseline |
| B — Temporal context | amount, transactionType, step | Evaluate simulation-time information |
| C — Pre-transaction balances | amount, transactionType, oldBalInitiator, oldBalRecipient | Evaluate account-state information |
| D — Combined | amount, transactionType, step, oldBalInitiator, oldBalRecipient | Combined pre-transaction candidate |

These groups will be compared using identical chronological splits.

For TRANSFER-only experiments, `transactionType` is constant and should not be included as a predictive feature.

## 6. Leakage prevention

1. Never include `isFraud` in model inputs.
2. Exclude all post-transaction features.
3. Exclude raw account identifiers initially.
4. Preserve the chronological train/validation/test split.
5. Fit imputers, encoders, scalers, and other preprocessing only on training data.
6. Apply fitted transformations to validation and test data without refitting.
7. Do not use test labels for feature selection or threshold tuning.
8. Do not apply SMOTE or other resampling before splitting.
9. Do not derive features using future transactions.
10. Any aggregated account-behavior features must use only historical information available before the transaction being evaluated.
11. Clearly document assumptions about the availability of pre-transaction balances.

## 7. Evaluation strategy

Two evaluation populations will be studied separately:

- Full MoMTSim V1 dataset.
- TRANSFER-only transactions.

This comparison is necessary because every observed fraudulent transaction in MoMTSim V1 belongs to the TRANSFER category.

The full-dataset experiment must be interpreted carefully because transaction type is strongly associated with the target.

The TRANSFER-only experiment provides a sensitivity analysis that removes this particular transaction-type shortcut.

## 8. Preprocessing

Numerical features will use a Scikit-learn preprocessing pipeline.

Categorical features will use one-hot encoding.

Imputation and scaling will be fitted using training data only.

The approved feature list must be applied before preprocessing.

Previously generated matrices containing post-transaction features are exploratory artifacts and must not be used for the selected pre-transaction scenario.

## 9. Evaluation metrics

Primary metrics:

- Precision
- Recall
- F1-score
- Precision-Recall AUC (PR-AUC)

Additional metrics:

- ROC-AUC
- Confusion matrix
- Accuracy, as a supplementary metric

Threshold selection will use validation data only.

Final test metrics will be calculated after model and threshold decisions are complete.

## 10. Limitations

MoMTSim V1 is synthetic mobile-money transaction data.

Its feature relationships and fraud-label distribution may reflect simulator-specific rules.

The availability of pre-transaction balance information is an explicit modelling assumption.

High predictive performance must not be presented as evidence of equivalent real-world banking fraud detection performance.

The selected dataset does not establish coverage of all five intended financial transaction channels.

## 11. Current status

**Detection scenario:** Pre-transaction real-time fraud detection.

**Feature policy:** Revised.

**Model training:** Not started.

**Final feature selection:** Pending experimental evaluation.
 