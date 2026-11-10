# MoMTSim V1 — TRANSFER-Only Final Model Evaluation

## 1. Objective

Evaluate a frozen Histogram Gradient Boosting candidate for pre-transaction fraud-risk prioritization among TRANSFER transactions in MoMTSim V1.

This experiment evaluates a chronological holdout period that was not used for model selection or hyperparameter tuning.

## 2. Dataset and chronological split

| Partition | TRANSFER transactions | Fraud rate |
|---|---:|---:|
| Training | 363,447 | 31.3157% |
| Validation | 94,936 | 30.5448% |
| Final test | 110,945 | 29.4777% |

The final test contains 32,704 fraud cases.

Only TRANSFER transactions were evaluated because all fraud-labelled transactions in this simulator dataset belong to the TRANSFER category.

## 3. Frozen model configuration

Model: HistGradientBoostingClassifier

Features:
- amount
- oldBalInitiator
- oldBalRecipient

Configuration:
- max_iter = 150
- learning_rate = 0.1
- max_leaf_nodes = 15
- l2_regularization = 1.0
- early_stopping = False
- random_state = 42
- class balancing through training sample weights

Missing numerical values are imputed using the training median.

The model was fitted using the chronological training partition only.

## 4. Final test performance

| Metric | Result |
|---|---:|
| Test fraud prevalence / PR-AUC baseline | 0.2948 |
| Model PR-AUC | 0.3363 |
| Model ROC-AUC | 0.5464 |

The model's PR-AUC exceeds the prevalence baseline, but its ROC-AUC indicates weak ranking discrimination.

Validation PR-AUC was 0.3401, compared with final test PR-AUC of 0.3363. This is a relatively small difference, although it does not establish long-term or real-world stability.

## 5. Review-capacity analysis

| Review capacity | Reviewed | True positives | False positives | False negatives | Precision | Recall | Lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1% | 1,110 | 633 | 477 | 32,071 | 0.5703 | 0.0194 | 1.9346 |
| 5% | 5,548 | 2,224 | 3,324 | 30,480 | 0.4009 | 0.0680 | 1.3599 |
| 10% | 11,095 | 3,947 | 7,148 | 28,757 | 0.3557 | 0.1207 | 1.2068 |

At 1% review capacity, precision is 57.03%, but recall is only 1.94%. The model therefore prioritizes a relatively enriched subset of transactions while missing most fraud cases.

## 6. Feature investigation

Permutation importance on validation data:

| Feature | Mean importance | Standard deviation |
|---|---:|---:|
| oldBalInitiator | 0.030931 | 0.001281 |
| amount | 0.005355 | 0.000486 |
| oldBalRecipient | -0.000018 | 0.000451 |

A recipient-balance ablation experiment compared:

| Feature set | PR-AUC | ROC-AUC |
|---|---:|---:|
| amount + oldBalInitiator + oldBalRecipient | 0.340129 | 0.520398 |
| amount + oldBalInitiator | 0.339148 | 0.519503 |

The recipient balance made a small positive contribution on validation. The three-feature version was retained for the final holdout evaluation.

Permutation importance and ablation results are dataset- and model-specific; they do not establish causation.

## 7. Important assumptions and limitations

1. The scenario is pre-transaction fraud-risk scoring.
2. oldBalInitiator and oldBalRecipient are included under the explicit assumption that these balances are available before authorization. Their availability in a real production system has not been established.
3. Post-transaction balances and derived post-event balance changes were excluded.
4. Raw account identifiers were excluded.
5. The evaluation is restricted to TRANSFER transactions.
6. Results are specific to this synthetic MoMTSim V1 simulator.
7. Synthetic simulator performance is not evidence of real-world bank performance.
8. The model has weak within-TRANSFER discrimination and is not production-ready.
9. The final test set must not be reused for further feature selection or hyperparameter tuning.

## 8. Conclusion

The frozen Gradient Boosting candidate achieved a test PR-AUC of 0.3363 against a test fraud prevalence of 0.2948, with ROC-AUC of 0.5464.

The model showed modest prioritization value at limited review capacities, but recall remained low. The results support further research into transaction-level features and stronger fraud-detection methods, not production deployment.

The experiment provides a reproducible baseline for future work on TRANSFER-only fraud-risk prioritization in MoMTSim V1.
EOF