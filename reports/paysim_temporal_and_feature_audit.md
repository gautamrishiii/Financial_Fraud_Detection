# PaySim Temporal and Feature Audit

## 1. Dataset overview

Dataset: PaySim

Total transactions: 6,362,620  
Unique time steps: 743  
Time-step range: 1–743  
Fraud cases: 8,213  
Overall fraud prevalence: 0.1291%

Target: `isFraud`

## 2. Transaction-type distribution

| Type | Transactions | Transaction share | Fraud cases | Fraud rate |
|---|---:|---:|---:|---:|
| CASH_OUT | 2,237,500 | 35.17% | 4,116 | 0.1840% |
| PAYMENT | 2,151,495 | 33.81% | 0 | 0.0000% |
| CASH_IN | 1,399,284 | 21.99% | 0 | 0.0000% |
| TRANSFER | 532,909 | 8.38% | 4,097 | 0.7688% |
| DEBIT | 41,432 | 0.65% | 0 | 0.0000% |

All observed fraud cases belong to CASH_OUT and TRANSFER.

This is a dataset-specific synthetic pattern, not a general rule about financial fraud.

## 3. Temporal distribution shift

Fraud prevalence by period:

| Steps | Transactions | Fraud cases | Fraud rate |
|---|---:|---:|---:|
| 1–100 | 1,059,516 | 1,190 | 0.112% |
| 101–200 | 1,322,666 | 1,064 | 0.080% |
| 201–300 | 1,691,468 | 1,147 | 0.068% |
| 301–400 | 1,713,380 | 1,076 | 0.063% |
| 401–500 | 274,777 | 1,084 | 0.395% |
| 501–600 | 197,240 | 1,052 | 0.533% |
| 601–700 | 92,152 | 1,110 | 1.205% |
| 701–743 | 11,421 | 490 | 4.290% |

Fraud prevalence increases substantially after step 400, while transaction volume decreases.

Fraud-rate increases occur in both CASH_OUT and TRANSFER.

## 4. Balance consistency observations

Balance-change analysis identified substantial differences between legitimate and fraudulent transactions.

Examples:

- Legitimate CASH_OUT origin balance movement matched the transaction amount in 10.883% of cases.
- Fraudulent CASH_OUT origin balance movement matched the amount in 99.417% of cases.
- Legitimate TRANSFER destination balance remained unchanged in 0.020% of cases.
- Fraudulent TRANSFER destination balance remained unchanged in 99.292% of cases.

These patterns suggest that post-transaction balance fields encode strong simulator-specific fraud signals.

They must not be treated as ordinary authorization-time features.

The balance equations and mismatch percentages are descriptive diagnostics, not universal financial-accounting rules.

## 5. Proposed chronological partitions

| Partition | Steps | Transactions | Fraud cases | Fraud rate |
|---|---:|---:|---:|---:|
| Training | 1–400 | 5,787,030 | 4,477 | 0.0774% |
| Validation | 401–600 | 472,017 | 2,136 | 0.4525% |
| Testing | 601–743 | 103,573 | 1,600 | 1.5448% |

The test fraud prevalence is approximately 20 times the training prevalence.

These partitions are accepted for the primary chronological experiment.

The test partition represents an out-of-time stress test under substantial distribution shift. Metrics must be interpreted in that context.

## 6. Initial feature policy

| Feature | Decision |
|---|---|
| `type` | Include |
| `amount` | Include |
| `step` | Include experimentally |
| `oldbalanceOrg` | Include conditionally |
| `oldbalanceDest` | Investigate availability |
| `newbalanceOrig` | Exclude |
| `newbalanceDest` | Exclude |
| `nameOrig` | Exclude |
| `nameDest` | Exclude |
| `isFlaggedFraud` | Exclude |
| `isFraud` | Target only |

Raw account identifiers are excluded from the initial model.

Post-transaction balances and features derived from them are excluded to avoid temporal leakage.

## 7. Evaluation policy

Primary metrics:

- Average precision / PR-AUC
- Precision
- Recall
- F1-score
- ROC-AUC
- Precision at fixed review capacity

Accuracy will not be used as the primary metric because fraud prevalence is extremely low.

Report the prevalence and number of fraud cases in every evaluation partition.

## 8. Limitations

1. PaySim is synthetic.
2. Fraud labels are concentrated in two transaction types.
3. Fraud prevalence changes sharply over time.
4. Balance-update patterns may expose simulator-specific label-generation behavior.
5. The availability of destination balance information at authorization time requires further investigation.
6. Results must not be interpreted as evidence of real-world production performance.