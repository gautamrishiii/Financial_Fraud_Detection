# MoMTSim V1 vs V2 — Comparative Analysis

## 1. Purpose

This report compares the MoMTSim V1 and V2 synthetic transaction datasets, their fraud-label distributions, exploratory findings, modelling experiments, and evaluation limitations.

The objective is to understand which findings are specific to each simulator version and whether the experiments provide evidence of generalizable fraud-detection capability.

## 2. Dataset comparison

| Property | MoMTSim V1 | MoMTSim V2 |
|---|---:|---:|
| Total transactions | 1,720,181 | 4,225,958 |
| Legitimate transactions | 1,544,663 | 1,992,840 |
| Fraudulent transactions | 175,518 | 2,233,118 |
| Fraud prevalence | 10.20% | 52.84% |
| Maximum step | 143 | 192 |
| Transaction types | 5 | 5 |
| Fraud transaction type | TRANSFER only | TRANSFER only |

Both versions assign all fraud labels to TRANSFER transactions. However, their TRANSFER fraud prevalence differs substantially.

- V1: approximately 30.83% of transfers are fraudulent.
- V2: approximately 88.94% of transfers are fraudulent.

Consequently, model metrics must be interpreted relative to each version's own population and baseline prevalence.

## 3. Transaction-type shortcut

Both datasets contain a strong relationship between transaction type and fraud labels.

All observed fraud cases occur in TRANSFER transactions. A classifier operating on all transaction types could exploit this relationship by flagging every transfer.

This would produce perfect recall on the observed data but would not distinguish fraudulent from legitimate transfers.

For this reason, the primary modelling experiments were restricted to TRANSFER transactions.

## 4. Important differences in feature patterns

### MoMTSim V1

V1 showed weak discrimination within the TRANSFER population.

The final selected model used:

- Histogram Gradient Boosting
- `amount`
- `oldBalInitiator`
- `oldBalRecipient`

Final test results:

| Metric | Result |
|---|---:|
| PR-AUC | 0.3363 |
| ROC-AUC | 0.5464 |
| Test fraud prevalence | 29.48% |

The model offered modest prioritization:

| Review capacity | Expected/observed precision | Recall |
|---:|---:|---:|
| 1% | 57.03% | 1.94% |
| 5% | 40.09% | 6.80% |
| 10% | 35.57% | 12.07% |

This model is not production-ready.

### MoMTSim V2

V2 showed substantially stronger relationships between pre-transaction balance features and fraud labels.

For TRANSFER transactions, the fraudulent and legitimate distributions differed markedly, particularly for `oldBalInitiator`.

A depth-6 decision tree using only `oldBalInitiator` achieved:

| Metric | Training | Validation |
|---|---:|---:|
| PR-AUC | 0.9969 | 0.9916 |
| ROC-AUC | Not reported here | 0.9613 |

The depth-2 tree achieved a slightly higher validation F1-score (0.9607) than depth 6 (0.9606), while depth 6 provided better validation PR-AUC.

However, constrained review analysis exposed a major limitation: the depth-6 tree's highest-scoring leaf contains approximately 43.16% of validation transfers.

At 10% review capacity, the expected recall was only 11.56%, despite expected precision close to 100%.

Further investigation found that training-derived exception patterns did not generalize reliably to validation.

## 5. Chronological evaluation

Both versions were divided chronologically rather than randomly.

### V1

| Split | Steps | Transactions | Fraud prevalence |
|---|---|---:|---:|
| Train | 0–92 | 1,161,938 | 9.80% |
| Validation | 93–116 | 215,320 | 13.47% |
| Test | 117–143 | 342,923 | 9.54% |

### V2

| Split | Steps | Transactions | Fraud prevalence |
|---|---|---:|---:|
| Train | 0–127 | 2,736,398 | 53.90% |
| Validation | 128–159 | 716,871 | 51.83% |
| Test | 160–192 | 772,689 | 50.04% |

The chronological approach is more appropriate than random splitting for studying temporal generalization.

However, the V2 test period has already been examined during exploratory analysis and rule evaluation. It must not be described as an untouched independent holdout.

## 6. Feature availability and leakage

The experiments use a pre-transaction fraud-detection framing.

The principal candidate features are:

- `amount`
- `oldBalInitiator`
- `oldBalRecipient`

The use of pre-transaction balances relies on an explicit assumption that these values are available before transaction authorization.

Post-transaction balances and balance changes are excluded from the intended pre-transaction feature set.

Raw account identifiers are also excluded from the initial models.

These assumptions require confirmation before any real-world deployment.

## 7. Why V1 and V2 metrics are not directly comparable

The datasets have different fraud prevalence, simulator behaviour, feature distributions, and temporal patterns.

V1's final model is a Histogram Gradient Boosting classifier evaluated on its test split.

V2's strongest reported tree metrics are from validation experiments.

Therefore, comparing V1 test PR-AUC directly with V2 validation PR-AUC would be methodologically misleading.

The differences in results demonstrate that synthetic simulator versions can encode very different label-generation mechanisms.

## 8. Main findings

1. Both datasets assign all observed fraud labels to TRANSFER transactions.
2. V1 has substantially weaker within-TRANSFER predictive signal than V2.
3. V2 has an unusually strong relationship between initiator balance and fraud labels.
4. V2's strong tree metrics do not automatically translate into useful low-capacity transaction ranking.
5. Exception rules derived from a small number of training examples failed to generalize to validation.
6. Chronological validation is essential because feature and fraud distributions change over time.
7. Pre-transaction balance availability remains an explicit modelling assumption.
8. Neither dataset alone establishes real-world fraud-detection effectiveness.

## 9. Current project status

| Area | Status |
|---|---|
| Dataset inventory | Completed |
| V1 data audit | Completed |
| V2 data audit | Completed |
| V1 chronological modelling | Completed |
| V1 final test evaluation | Completed |
| V2 chronological modelling | Exploratory experiments completed |
| V2 constrained review analysis | Completed |
| V2 exception-rule investigation | Completed |
| Cross-version comparison | Documented |
| Independent external validation | Not completed |
| Production readiness assessment | Not established |

## 10. Recommended research direction

The next research phase should investigate whether the observed V2 relationships are specific to its synthetic label-generation mechanism.

Priority areas:

1. Examine simulator documentation and label-generation logic, if available.
2. Investigate the provenance and meaning of balance features.
3. Compare feature distributions across independent datasets.
4. Evaluate models on a genuinely independent dataset or a new simulation period.
5. Assess review-capacity performance rather than relying only on aggregate classification metrics.
6. Investigate whether useful discrimination remains after controlling for simulator-specific shortcuts.

## 11. Final conclusion

MoMTSim V1 and V2 demonstrate two very different modelling environments.

V1 provides a challenging within-TRANSFER classification problem with weak predictive performance.

V2 provides apparently strong classification results, but its unusually high fraud prevalence and strong balance-feature relationships raise concerns about simulator-specific label patterns.

The project therefore treats predictive metrics as evidence about these synthetic datasets, not proof of real-world fraud-detection capability.

Further independent validation and investigation of the simulator's label-generation mechanism are necessary before making stronger claims.
