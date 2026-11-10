# MoMTSim V2 — Modeling Findings

## 1. Objective

Investigate whether pre-transaction features can distinguish fraudulent
from legitimate TRANSFER transactions in MoMTSim V2.

This is an exploratory study of a synthetic dataset, not a production
fraud-detection evaluation.

## 2. Modeling population

Only TRANSFER transactions are included.

All labeled fraud transactions in MoMTSim V2 occur in the TRANSFER
category. Restricting the modeling population avoids allowing
transactionType to act as a trivial classifier.

## 3. Feature policy

| Feature | Treatment |
|---|---|
| amount | Include |
| oldBalInitiator | Include, subject to balance-availability assumption |
| oldBalRecipient | Include, subject to balance-availability assumption |
| step | Exclude from primary model; retain for temporal analysis |
| transactionType | Exclude; constant within TRANSFER population |
| newBalInitiator | Exclude; post-transaction |
| newBalRecipient | Exclude; post-transaction |
| initiator | Exclude initially; raw identifier |
| recipient | Exclude initially; raw identifier |
| isFraud | Target only |

Balance fields are treated as available before authorization solely
under an explicit modeling assumption. This availability must not be
presented as a verified production capability.

## 4. Chronological splits

| Split | Steps | Transactions | Transfer transactions | Transfer fraud rate |
|---|---|---:|---:|---:|
| Train | 0–127 | 2,736,398 | 1,613,966 | 91.38% |
| Validation | 128–159 | 716,871 | 429,545 | 86.50% |
| Test | 160–192 | 772,689 | 467,436 | 82.72% |

Fraud prevalence declines across the chronological splits.

## 5. Rule-pattern findings

Among TRANSFER transactions:

- Amount below 1,000 has an observed fraud rate of approximately
  100% across the three splits.
- Initiator opening balance below 1,000 also has an observed fraud
  rate above 99%.
- The combination of amount below 1,000 and initiator balance below
  1,000 produced no observed legitimate cases in the evaluated splits.
- Fraud prevalence changes substantially across amount and balance
  combinations.
- These relationships are likely associated with synthetic
  label-generation mechanisms and must not be generalized to
  real financial transactions.

Eight legitimate transfers had amounts below 1,000:
seven in training, none in validation, and one in test. Their
initiator opening balances were all above 73 million.

## 6. Decision-tree experiment

A shallow decision tree was trained using:

- amount
- oldBalInitiator
- oldBalRecipient

Configuration:

- max_depth = 4
- min_samples_leaf = 500
- random_state = 42

Validation results:

- PR-AUC: 0.9881
- ROC-AUC: 0.9544
- Fraud precision: 0.9816
- Fraud recall: 0.9402

Feature importance:

| Feature | Importance |
|---|---:|
| oldBalInitiator | 0.9953 |
| amount | 0.0043 |
| oldBalRecipient | 0.0004 |

## 7. Feature ablation

| Features | PR-AUC | ROC-AUC |
|---|---:|---:|
| oldBalInitiator | 0.9884 | 0.9539 |
| amount | 0.9708 | 0.8481 |
| amount + oldBalInitiator | 0.9881 | 0.9544 |
| All three features | 0.9881 | 0.9544 |

Initiator opening balance alone reproduces nearly all of the
three-feature model's validation performance.

## 8. Tree depth experiment

Balance-only decision trees:

| Depth | Leaves | PR-AUC | ROC-AUC |
|---:|---:|---:|---:|
| 1 | 2 | 0.9728 | 0.9076 |
| 2 | 4 | 0.9766 | 0.9191 |
| 3 | 8 | 0.9834 | 0.9399 |
| 4 | 15 | 0.9884 | 0.9539 |
| 5 | 26 | 0.9901 | 0.9584 |
| 6 | 38 | 0.9916 | 0.9613 |
| 8 | 76 | 0.9918 | 0.9612 |

Depth 6 is a promising candidate. Depth 8 adds very little improvement.

## 9. Temporal validation of depth 6

| Validation steps | Fraud prevalence | PR-AUC | ROC-AUC |
|---|---:|---:|---:|
| 128–135 | 88.71% | 0.9939 | 0.9678 |
| 136–143 | 86.21% | 0.9920 | 0.9634 |
| 144–151 | 85.88% | 0.9909 | 0.9592 |
| 152–159 | 85.32% | 0.9893 | 0.9553 |

Depth 6 outperformed depth 4 in every validation window.
Performance nevertheless declines gradually over time.

## 10. Experimental limitations

1. The dataset is synthetic.
2. The extreme fraud prevalence is unlike many operational
   transaction-monitoring settings.
3. The initiator opening balance dominates model predictions.
4. The underlying label-generation mechanism has not been verified.
5. Balance availability before authorization is an assumption.
6. The validation set has been reused across exploratory experiments
   and model comparisons.
7. The test set has already been examined in exploratory distribution
   and rule analyses. It is not an untouched independent holdout.
8. These results do not establish real-world fraud-detection
   effectiveness.

## 11. Current conclusion

MoMTSim V2 contains strong nonlinear associations between fraud labels
and initiator opening balance.

A balance-only decision tree with depth 6 achieved validation
PR-AUC of 0.9916 and ROC-AUC of 0.9613.

The result is promising for understanding the synthetic dataset,
but it must not be interpreted as evidence of production readiness.

Further work should investigate the label-generation mechanism,
compare model complexity, document feature availability assumptions,
and seek an independent dataset or future time period for external
evaluation.
## 12. Threshold baseline versus depth-6 tree

A median-balance threshold baseline was compared with a depth-6
decision tree on the chronological validation split.

The threshold was calculated from the training-set median of
oldBalInitiator: 2,515.80.

| Metric | Median threshold | Depth-6 tree |
|---|---:|---:|
| Precision | 0.9935 | 0.9806 |
| Recall | 0.5407 | 0.9414 |
| F1 | 0.7003 | 0.9606 |
| True positives | 200,891 | 349,781 |
| False positives | 1,306 | 6,908 |
| True negatives | 56,681 | 51,079 |
| False negatives | 170,667 | 21,777 |

The depth-6 tree detected 148,890 more fraudulent transfers, with
5,602 additional false positives.

The median threshold achieved higher precision but substantially
lower recall.

### Metric caveat

The threshold baseline produces binary scores rather than continuous
risk scores. Consequently, its PR-AUC and ROC-AUC are not directly
comparable to the ranking metrics of the probability-producing tree.

Precision, recall, F1 and confusion matrices provide a more useful
comparison of these particular classification decisions.

### Conclusion

The depth-6 tree is the stronger candidate when detecting more
fraudulent transactions is the primary objective.

The median threshold is more conservative and produces fewer
false alerts.

Neither approach is production-ready based on synthetic validation
results alone.

## 13. Decision tree complexity comparison

Decision trees with maximum depths from 1 to 8 were compared using the same chronological training and validation splits.

All models used:
- Feature: `oldBalInitiator`
- Minimum samples per leaf: 500
- Random state: 42

| Depth | Leaves | Validation PR-AUC | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0.9728 | 0.9798 | 0.9394 | 0.9592 |
| 2 | 4 | 0.9766 | 0.9785 | 0.9434 | 0.9607 |
| 3 | 8 | 0.9834 | 0.9816 | 0.9402 | 0.9604 |
| 4 | 15 | 0.9884 | 0.9816 | 0.9402 | 0.9604 |
| 5 | 26 | 0.9901 | 0.9803 | 0.9414 | 0.9605 |
| 6 | 38 | 0.9916 | 0.9806 | 0.9414 | 0.9606 |
| 7 | 54 | 0.9917 | 0.9806 | 0.9414 | 0.9606 |
| 8 | 76 | 0.9918 | 0.9806 | 0.9414 | 0.9606 |

### Findings

- Depth 2 achieved the highest validation F1 (0.9607).
- Depth 6 achieved validation PR-AUC of 0.9916, compared with 0.9766 at depth 2.
- Depths 5 through 8 produced nearly identical classification decisions.
- Increasing depth from 6 to 8 added 38 leaves but provided negligible PR-AUC improvement.
- Greater tree complexity primarily refined probability estimates rather than changing the predicted classes.

### Candidate selection

Retain depth 2 and depth 6 as competing candidates:

- Depth 2: simpler structure and slightly higher validation F1.
- Depth 6: stronger validation ranking performance.

No final production model is selected. These results are based on synthetic data, and the simulator's label-generation mechanism remains an important limitation.

The test data has already been examined during exploratory analysis, so it should not be described as an untouched independent holdout.

## 14. Depth-2 decision tree leaf stability

The depth-2 decision tree was fitted on training TRANSFER transactions and evaluated on the chronological validation period.

The tree produced four leaves.

| Leaf | Training transactions | Training fraud rate | Validation transactions | Validation fraud rate |
|---:|---:|---:|---:|---:|
| 2 | 3,426 | 29.101% | 1,763 | 29.155% |
| 3 | 1,423,251 | 99.596% | 354,503 | 98.319% |
| 5 | 5,852 | 52.119% | 3,719 | 53.455% |
| 6 | 181,437 | 29.403% | 69,560 | 29.488% |

### Learned balance ranges

| Leaf | Initiator balance range | Predicted class |
|---:|---|---|
| 2 | ≤ -42.10 | Legitimate |
| 3 | > -42.10 and ≤ 2,645,072.62 | Fraud |
| 5 | > 2,645,072.62 and ≤ 4,889,601.25 | Fraud |
| 6 | > 4,889,601.25 | Legitimate |

### Findings

- Leaves 2 and 6 showed very similar fraud rates across the two periods.
- Leaf 3 decreased from 99.596% to 98.319%.
- Leaf 5 increased from 52.119% to 53.455%.
- The learned decision boundaries were applied unchanged to validation data.

The four-leaf model therefore shows reasonably consistent leaf-level fraud prevalence across these periods, although some drift is present.

### Limitations

These fraud rates describe the synthetic MoMTSim V2 dataset. They do not establish equivalent relationships in real financial transactions.

The balance-based decision rules also depend on the assumption that `oldBalInitiator` is available before transaction authorization.

## 15. Three-level risk categorization

The four leaves of the depth-2 decision tree were consolidated into three risk categories.

- LOW: leaves 2 and 6
- MEDIUM: leaf 5
- HIGH: leaf 3

### Training and validation results

| Split | Risk category | Transactions | Fraud | Legitimate | Fraud rate |
|---|---|---:|---:|---:|---:|
| TRAIN | LOW | 184,863 | 54,345 | 130,518 | 29.397% |
| TRAIN | MEDIUM | 5,852 | 3,050 | 2,802 | 52.119% |
| TRAIN | HIGH | 1,423,251 | 1,417,508 | 5,743 | 99.596% |
| VALIDATION | LOW | 71,323 | 21,026 | 50,297 | 29.480% |
| VALIDATION | MEDIUM | 3,719 | 1,988 | 1,731 | 53.455% |
| VALIDATION | HIGH | 354,503 | 348,544 | 5,959 | 98.319% |

### Findings

- The ordering of observed fraud rates is consistent across training and validation.
- LOW has approximately 29.5% fraud prevalence.
- MEDIUM has approximately 52–53% fraud prevalence.
- HIGH has approximately 98–100% fraud prevalence.
- The HIGH category contains most transactions, so it is not a selective review queue.

### Interpretation and limitations

These categories provide an understandable summary of the depth-2 tree's predictions.

The observed fraud rates are historical synthetic-data statistics, not calibrated probabilities of real-world fraud. Operational use would require validation on representative real transaction data, assessment of review capacity, and confirmation that the balance feature is available before authorization.

## 16. Review capacity analysis

Two review policies were evaluated using the depth-2 decision tree.

- HIGH_ONLY: review transactions assigned to the HIGH category.
- MEDIUM_AND_HIGH: review transactions assigned to either MEDIUM or HIGH.

### Validation results

| Metric | HIGH only | MEDIUM + HIGH |
|---|---:|---:|
| Transactions reviewed | 354,503 | 358,222 |
| Review workload | 82.530% | 83.396% |
| True positives | 348,544 | 350,532 |
| False positives | 5,959 | 7,690 |
| False negatives | 23,014 | 21,026 |
| True negatives | 52,028 | 50,297 |
| Precision | 0.9832 | 0.9785 |
| Recall | 0.9381 | 0.9434 |

### Findings

Including MEDIUM alongside HIGH:
- Adds 3,719 transactions to the review queue.
- Detects 1,988 additional fraudulent transfers.
- Adds 1,731 false alerts.
- Improves recall by approximately 0.46 percentage points.
- Reduces precision by approximately 0.47 percentage points.

The HIGH-only policy is more selective, but still requires reviewing 82.530% of validation transfers.

### Operational interpretation

The model demonstrates strong classification performance on the synthetic validation data, but its review workload is substantial.

Neither policy currently provides a sufficiently selective review queue for a low-capacity fraud investigation team.

These results do not establish real-world operational effectiveness. Additional validation using representative real transaction data and realistic review-capacity constraints is required.

## 17. Constrained review capacity and score ties

The depth-2 tree was evaluated at review capacities of 1%, 2%, 5%, and 10% of validation TRANSFER transactions.

### Validation score distribution

| Predicted score | Transactions | Fraud | Legitimate |
|---:|---:|---:|---:|
| 0.995965 | 354,503 | 348,544 | 5,959 |
| 0.521189 | 3,719 | 1,988 | 1,731 |
| 0.294030 | 69,560 | 20,512 | 49,048 |
| 0.291010 | 1,763 | 514 | 1,249 |

The highest-scoring group contains 82.53% of validation transfers.

### Expected review results

| Review capacity | Reviewed | Expected fraud detected | Expected precision | Expected recall |
|---:|---:|---:|---:|---:|
| 1% | 4,296 | 4,223.8 | 0.9832 | 0.0114 |
| 2% | 8,591 | 8,446.6 | 0.9832 | 0.0227 |
| 5% | 21,478 | 21,117.0 | 0.9832 | 0.0568 |
| 10% | 42,955 | 42,233.0 | 0.9832 | 0.1137 |

### Tie-handling methodology

The tree produces only four distinct predicted scores. The review capacities fall inside the largest score group.

Expected results were therefore calculated by assuming random selection among transactions with identical scores. These are expected counts, not deterministic transaction-level outcomes.

### Findings

- The tree achieves high precision within its highest-scoring group.
- That group contains 82.53% of validation transfers.
- At 5% review capacity, expected recall is only 5.68%.
- The tree cannot distinguish individual transactions within a leaf because they receive identical scores.

### Conclusion

The depth-2 tree separates broad risk groups effectively on this synthetic dataset, but its limited score resolution makes it unsuitable for fine-grained prioritization under constrained review capacity.

Classification performance and individual transaction ranking must be evaluated separately.

## 18. Depth-6 constrained review capacity

The depth-6 decision tree was evaluated at review capacities of 1%, 2%, 5%, and 10% using the chronological validation TRANSFER population.

### Expected review results

| Review capacity | Reviewed | Expected fraud detected | Expected precision | Expected recall |
|---:|---:|---:|---:|---:|
| 1% | 4,296 | 4,295.8 | 1.0000 | 0.0116 |
| 2% | 8,591 | 8,590.6 | 1.0000 | 0.0231 |
| 5% | 21,478 | 21,477.1 | 1.0000 | 0.0578 |
| 10% | 42,955 | 42,953.1 | 1.0000 | 0.1156 |

### Highest-scoring leaf

The highest-scoring leaf contained:

- Transactions: 185,388
- Fraud: 185,380
- Legitimate: 8
- Observed fraud rate: approximately 99.996%

This leaf represents approximately 43.16% of validation transfers.

All four review capacities fall entirely inside this leaf. Consequently, the model cannot distinguish individual transactions within the selected group.

### Comparison with depth 2

Depth 6 improves expected precision and recall slightly at the tested capacities.

However, expected recall remains low, ranging from approximately 1.16% at 1% capacity to 11.56% at 10% capacity.

### Tie-handling limitation

The expected counts assume random selection among transactions with identical predicted scores. They are not deterministic transaction-level results.

### Conclusion

The depth-6 tree provides more score levels than depth 2, but its largest high-scoring leaf still prevents effective individual transaction ranking at low review capacities.

Its high expected precision should not be interpreted as evidence of strong individual ranking ability or real-world fraud detection effectiveness.

## 19. High-risk leaf exception-rule investigation

The depth-6 tree's highest-scoring training leaf contained 740,763 transactions:

- Fraud: 740,751
- Legitimate: 12

Three candidate features were investigated within this leaf: `amount`, `oldBalRecipient`, and `step`.

### Training observations

All 12 legitimate transactions had `oldBalRecipient = 0`.

Their amounts ranged from 18,533.84 to 1,191,453.73, whereas the maximum fraudulent amount in this leaf was 143,656.73.

A training-derived amount condition (`amount > 143656.73`) selected five legitimate transactions and no fraudulent transactions. However, this apparent separation was based on only five examples.

### Validation evaluation

The corresponding validation leaf contained 185,388 transactions:

- Fraud: 185,380
- Legitimate: 8

| Condition | Selected | Legitimate captured | Fraud included | Legitimate precision | Legitimate recall |
|---|---:|---:|---:|---:|---:|
| Recipient balance = 0 | 419 | 3 | 416 | 0.7160% | 37.50% |
| Amount > 100,000 | 0 | 0 | 0 | N/A | 0% |
| Amount > 143,656.73 | 0 | 0 | 0 | N/A | 0% |
| Combined condition | 419 | 3 | 416 | 0.7160% | 37.50% |

The combined condition produced the same result as recipient balance equal to zero.

### Interpretation

The training-derived amount pattern did not recur in validation. Recipient balance equal to zero captured only three of eight legitimate transactions and also selected 416 fraudulent transactions.

These findings demonstrate that the apparent exception patterns in the training leaf do not generalize reliably.

### Conclusion

No additional exception rule is adopted.

The small number of legitimate training examples is insufficient to establish a reliable distinction. The depth-6 model's high-risk leaf remains highly fraud-heavy, but it does not provide effective individual ranking within that leaf.

Validation results are diagnostic rather than independent final evidence because this period has already been used in earlier experiments.
