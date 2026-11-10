# MoMTSim V1 — Feature Engineering Plan

## 1. Objective

Define an initial, leakage-aware feature set for the MoMTSim V1 financial fraud detection experiments.

The goal is to investigate whether transaction and balance information can distinguish fraudulent from legitimate transactions, while avoiding raw account identifiers and misleading conclusions caused by synthetic data-generation rules.

## 2. Target Variable

- Target column: `isFraud`
- Legitimate transaction: `0`
- Fraudulent transaction: `1`

The target must never be included among model input features.

## 3. Initial Feature Decisions

| Original column | Initial treatment | Reason |
|---|---|---|
| `amount` | Include | Transaction amount |
| `step` | Include cautiously | Simulation time indicator |
| `transactionType` | Compare experimentally | All observed fraud cases are TRANSFER |
| `oldBalInitiator` | Include | Initiator's balance before transaction |
| `newBalInitiator` | Include | Initiator's balance after transaction |
| `oldBalRecipient` | Include | Recipient's balance before transaction |
| `newBalRecipient` | Include | Recipient's balance after transaction |
| `initiator` | Exclude initially | High-cardinality account identifier |
| `recipient` | Exclude initially | High-cardinality account identifier |
| `isFraud` | Target only | Prediction label |

## 4. Derived Features to Investigate

### 4.1 Initiator balance change

`initiator_balance_change = oldBalInitiator - newBalInitiator`

This represents the decrease in the initiator's balance.

### 4.2 Recipient balance change

`recipient_balance_change = newBalRecipient - oldBalRecipient`

This represents the increase in the recipient's balance.

### 4.3 Initiator balance difference

`initiator_balance_difference = amount - initiator_balance_change`

This measures the difference between the transaction amount and the initiator's observed balance decrease.

### 4.4 Recipient balance difference

`recipient_balance_difference = amount - recipient_balance_change`

This measures the difference between the transaction amount and the recipient's observed balance increase.

These features are descriptive differences, not guaranteed accounting errors. Their meaning may depend on transaction type and simulator behavior.

## 5. Experimental Feature Groups

We will compare separate feature groups rather than assuming that every available feature is useful.

- **Group A — Basic:** amount, step.
- **Group B — Transaction context:** amount, step, transactionType.
- **Group C — Balance information:** amount, step, four original balance columns.
- **Group D — Derived balance information:** amount, step, derived balance features.
- **Group E — Combined:** selected basic, contextual, original balance, and derived features.

A separate TRANSFER-only experiment will investigate whether results remain meaningful when transaction type cannot provide a shortcut.

## 6. Leakage Prevention Rules

1. Never include `isFraud` as an input feature.
2. Never use raw `initiator` or `recipient` identifiers in the initial models.
3. Split data chronologically using complete simulation steps.
4. Fit preprocessing, encoding, scaling, and feature selection using training data only.
5. Do not use test labels to select features or tune model thresholds.
6. Do not apply SMOTE or other resampling before splitting.
7. Calculate each feature using information available at the transaction being evaluated. Do not use future transactions.

## 7. Evaluation Strategy

- Primary split: chronological training steps 0–116 and testing steps 117–143.
- Compare the full dataset experiment with the TRANSFER-only experiment.
- Use precision, recall, F1-score, PR-AUC, ROC-AUC, and confusion matrix.
- Do not rely on accuracy alone.
- Tune decision thresholds using validation data, not the final test set.

## 8. Research Limitations

- MoMTSim V1 is synthetic data.
- Every observed fraudulent transaction is a TRANSFER.
- Transaction type and amount distributions may expose simulator-specific label-generation patterns.
- High model scores must not be presented as evidence of equivalent real-world banking performance.
- Results should be interpreted as experiments on this dataset.

## 9. Implementation Status

**Status:** Planning documented; feature implementation and model training not yet started.

## 10. Findings from Feature Analysis

### 10.1 Correlation analysis

Pearson correlation analysis was performed on all 1,720,181 transactions.

Two pairs of original balance features showed high correlation:

| Feature pair | Correlation |
|---|---:|
| oldBalInitiator and newBalInitiator | 0.9970 |
| oldBalRecipient and newBalRecipient | 0.9882 |

This indicates strong linear relationships. However, correlated features will not automatically be removed.

### 10.2 Feature relationships with the target

The analysis compared legitimate and fraudulent transactions across:

- The full dataset.
- TRANSFER transactions only.

The TRANSFER-only comparison is particularly important because every observed fraudulent transaction belongs to the TRANSFER category.

The median transaction amounts were:

| Category | Legitimate | Fraud |
|---|---:|---:|
| TRANSFER amount | 19,733.60 | 19,631.62 |

The similar medians indicate that transaction amount alone should not be assumed to distinguish fraud reliably.

### 10.3 Balance-change redundancy

The initiator and recipient balance-change features were strongly correlated, with Pearson correlation displayed as 1.00000000.

However, they were not exactly identical.

Across 569,328 TRANSFER transactions:

- Exact matches: 55,425.
- Different rows: 513,903.
- Maximum absolute difference: approximately 0.01.
- Seven records failed NumPy's default `isclose` comparison.

All seven mismatches were legitimate transactions and showed approximately 0.01 differences.

The exact cause has not been established.

### 10.4 Feature-selection decision

For initial experiments:

- Preserve the original balance columns.
- Retain derived balance features as experimental candidates.
- Do not automatically remove highly correlated features.
- Do not round or manually correct the seven discrepancies.
- Compare feature groups experimentally.
- Evaluate full-dataset and TRANSFER-only models separately.

### 10.5 Research caution

These findings describe relationships within a synthetic dataset. They do not establish real-world financial fraud behavior or causal relationships.

**Status:** Feature analysis documented. Model training has not started.
