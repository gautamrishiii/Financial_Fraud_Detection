# Multi-Channel Financial Fraud Detection System
## Project Experiment Summary

**Project status:** MoMTSim V1 final holdout evaluation completed
**Current modelling scenario:** Pre-transaction, real-time fraud detection
**Current scope:** MoMTSim V1 TRANSFER-only experiment

---

## 1. Project objective

Develop a multi-channel financial fraud detection system using multiple transaction datasets, with a focus on identifying fraudulent transactions while accounting for class imbalance, data leakage, temporal behaviour, and operational alert capacity.

The broader project includes MoMTSim, PaySim, and IEEE-CIS datasets. Each dataset requires its own data preparation and modelling decisions because their schemas, fraud rates, and transaction characteristics differ.

## 2. Dataset inventory

| Dataset | Records | Fraud records | Fraud rate |
|---|---:|---:|---:|
| MoMTSim V1 | 1,720,181 | 175,518 | 10.20% |
| MoMTSim V2 | 4,225,958 | 2,233,118 | 52.84% |
| IEEE-CIS training | 590,540 | 20,663 | 3.50% |
| PaySim | 6,362,620 | 8,213 | 0.13% |

These datasets are not directly interchangeable. Their labels, schemas, class distributions, and simulation characteristics must be considered separately.

## 3. MoMTSim V1 data audit

- 1,720,181 records and 10 original columns.
- No missing or blank values were found in the audited columns.
- No exact duplicate rows were found.
- Transaction types include DEBIT, DEPOSIT, PAYMENT, TRANSFER, and WITHDRAWAL.
- All labelled fraud records belong to the TRANSFER category.
- Account identifiers have high cardinality and are excluded from the initial model.
- Negative balance values were observed and retained rather than automatically removed.
- Balance discrepancies were investigated without modifying source values.

**Important limitation:** The concentration of all fraud labels in TRANSFER transactions creates a strong opportunity for shortcut learning. Overall model scores can look impressive without demonstrating useful discrimination among transfers.

## 4. Chronological data splitting

MoMTSim V1 was divided by its simulator `step` variable.

| Split | Steps | Records | Fraud records | Fraud rate |
|---|---|---:|---:|---:|
| Training | 0–92 | 1,161,938 | 113,816 | 9.80% |
| Validation | 93–116 | 215,320 | 28,998 | 13.47% |
| Test | 117–143 | 342,923 | 32,704 | 9.54% |

The chronological split preserves temporal ordering. The test partition was reserved for final evaluation rather than model selection.

## 5. Feature availability policy

The selected scenario is **pre-transaction real-time fraud detection**.

| Feature | Decision | Reason |
|---|---|---|
| amount | Include | Known from the transaction request |
| transactionType | Include in general experiments | Known from the transaction request |
| step | Experimental | Simulator time index; not necessarily a production feature |
| oldBalInitiator | Conditional | Requires a balance lookup before authorization |
| oldBalRecipient | Conditional | Requires a balance lookup before authorization |
| newBalInitiator | Exclude | Post-transaction information |
| newBalRecipient | Exclude | Post-transaction information |
| Raw account identifiers | Exclude initially | High cardinality and potential memorization |
| isFraud | Target only | Must never be an input feature |

The balance fields are included only under the explicit assumption that these balances are available before transaction authorization. This availability has not been established for a real payment system.

## 6. Baseline modelling

Logistic Regression experiments compared feature groups A–D.

| Feature group | Features | PR-AUC |
|---|---|---:|
| A | amount, transactionType | 0.3093 |
| B | A + step | 0.3095 |
| C | amount, transactionType, old balances | 0.3312 |
| D | All five selected fields | 0.3315 |

The full-dataset baseline benefited substantially from transaction type. Since all fraud labels belong to TRANSFER, this result does not establish strong fraud discrimination within transfers.

## 7. TRANSFER-only experiments

To assess discrimination without relying on transaction type, modelling was restricted to TRANSFER transactions.

### Logistic Regression feature ablation

| Feature group | Features | PR-AUC |
|---|---|---:|
| A | amount | 0.3093 |
| B | amount, step | 0.3095 |
| C | amount, oldBalInitiator, oldBalRecipient | 0.3317 |
| D | amount, step, old balances | 0.3320 |

The improvement from old balance information was modest. Adding `step` contributed almost no additional value.

### Model comparison

| Model | Validation PR-AUC | Validation ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.3317 | 0.5213 |
| Histogram Gradient Boosting | 0.3393 | 0.5190 |

Histogram Gradient Boosting achieved a modestly higher PR-AUC, while Logistic Regression achieved a slightly higher ROC-AUC. The difference was not sufficient to establish a universally superior model.

### Temporal stability

The validation period was divided into six chronological windows. Histogram Gradient Boosting achieved higher PR-AUC than Logistic Regression in all six windows, but ROC-AUC remained close to 0.51–0.55.

This indicates some consistency in the modest precision-recall improvement, not strong predictive discrimination.

### Feature importance

Permutation importance for the selected gradient boosting model:

| Feature | PR-AUC importance |
|---|---:|
| oldBalInitiator | 0.030931 |
| amount | 0.005355 |
| oldBalRecipient | -0.000018 |

The initiator's old balance contributed the most measurable predictive value in this experiment. The recipient's old balance had no measurable positive contribution in this importance analysis.

Removing the recipient balance slightly reduced validation PR-AUC, so the three-feature model was retained as the final candidate based on validation results.

## 8. Final holdout evaluation

**Selected model:** Histogram Gradient Boosting Classifier

**Features:**
- amount
- oldBalInitiator
- oldBalRecipient

**Configuration:**
- max_iter = 150
- learning_rate = 0.1
- max_leaf_nodes = 15
- l2_regularization = 1.0
- early_stopping = False
- random_state = 42
- Balanced training sample weights
- Median imputation

The model was trained on training-period TRANSFER transactions and evaluated on the untouched test-period TRANSFER transactions.

### Test results

| Metric | Result |
|---|---:|
| Test transactions | 110,945 |
| Fraud transactions | 32,704 |
| Test fraud rate | 29.4777% |
| Dummy PR-AUC baseline | 0.2948 |
| Model PR-AUC | 0.3363 |
| Model ROC-AUC | 0.5464 |

### Review-capacity analysis

| Review capacity | Reviewed | True positives | False positives | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| 1% | 1,110 | 633 | 477 | 0.5703 | 0.0194 |
| 5% | 5,548 | 2,224 | 3,324 | 0.4009 | 0.0680 |
| 10% | 11,095 | 3,947 | 7,148 | 0.3557 | 0.1207 |

The model prioritizes fraud somewhat better than random selection, but captures only a small fraction of fraud at limited review capacity.

## 9. Final interpretation

The final model demonstrates modest ranking ability on the MoMTSim V1 TRANSFER-only holdout.

It is not production-ready.

The results should be interpreted within the limits of this synthetic dataset. They do not establish performance on real financial transactions or on other datasets.

The experiment also depends on the assumption that the two old-balance features are available before authorization.

The final test set must not be reused for further feature selection, threshold tuning, or model comparison. Any future model changes should be evaluated using a new, appropriately separated evaluation period or dataset.

## 10. Important project files

### Main scripts

- `src/inspect_datasets.py`
- `src/analyze_labels.py`
- `src/audit_momtsim.py`
- `src/prepare_momtsim_splits.py`
- `src/prepare_momtsim_preprocessing.py`
- `src/train_momtsim_baseline.py`
- `src/train_momtsim_transfer_only.py`
- `src/compare_transfer_models.py`
- `src/analyze_transfer_temporal_stability.py`
- `src/analyze_transfer_feature_importance.py`
- `src/evaluate_momtsim_transfer_final.py`

### Main reports

- `reports/momtsim_v1_model_input_policy.md`
- `reports/momtsim_v1_feature_engineering_plan.md`
- `reports/momtsim_v1_transfer_final_evaluation.md`

### Main evaluation outputs

- `eda_outputs/momtsim_v1_train.csv`
- `eda_outputs/momtsim_v1_validation.csv`
- `eda_outputs/momtsim_v1_test.csv`
- `eda_outputs/momtsim_v1_transfer_final_metrics.csv`
- `eda_outputs/momtsim_v1_transfer_final_capacity.csv`

## 11. Next project stages

1. Review and consolidate the MoMTSim V1 experiment.
2. Design a separate data audit and feature policy for MoMTSim V2.
3. Develop a separate PaySim modelling pipeline.
4. Investigate IEEE-CIS transaction and identity data, including the train/test schema differences.
5. Compare dataset-specific model results without assuming their scores are directly comparable.
6. Plan a unified application architecture only after the individual dataset experiments are understood.

---

**Current milestone:** MoMTSim V1 TRANSFER-only final holdout evaluation and documentation completed.
**Overall project status:** Dataset exploration and one dataset-specific modelling experiment completed; multi-dataset modelling and application development remain.