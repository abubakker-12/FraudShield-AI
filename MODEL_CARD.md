# FraudShield AI model card

## Active model

Depth-4 XGBoost, version `fs1-6430f15b1824`, review threshold `0.640148937702179`. The 0–1 output is a class-weighted, **uncalibrated model score**. The binary flag is true at or above the threshold. The target is the source dataset's recorded `is_fraud` label.

## Intended use

Portfolio demonstration, fraud triage research, and human investigation training with synthetic records. A flag prioritizes a review. A reviewer determines case outcomes from independent evidence. The model has no authority to deny a payment or confirm fraud.

## Data and input contract

The supplied source contains 1,000,000 transactions, 198,662 customer identifiers and a 5.5255% recorded fraud rate. Its production lineage and snapshot semantics are undocumented. The scorer uses nine allowlisted transaction fields, including hour, amount, authentication activity, payment, merchant and device categories. Standard scaling and one-hot encoding were fitted on training data only. Derived inputs include overnight, repeated failures, short intervals, log amount and interactions. Account balance, frequency, raw identifiers, raw timestamps and `fraud_type` are excluded. The merchant-by-period interaction was investigated offline.

International status, retry count, prior transaction interval and recent PIN change must refer to observations available at scoring time. No production feature store or external validation was supplied.

## Evaluation

The initial splits preserved chronology and disjoint customer buckets. The baseline fit 100,000 sampled training records, ranked models on 28,950 validation rows, and selected its threshold by validation F1. The test set comprised 32,465 rows.

| Held-out metric | Result |
|---|---:|
| Average precision | .12918 |
| ROC AUC | .72549 |
| Accuracy | 82.21% |
| Precision | 13.25% |
| Recall | 40.21% |
| F1 | .19936 |
| False-positive rate | 15.34% |
| False-negative rate | 59.79% |
| TN / FP / FN / TP | 25,971 / 4,706 / 1,069 / 719 |

During 2.0, the retained model reached .14163 AP on the later selection subpartition. The strongest challenger reached .14010. Offline calibration used an earlier, distinct calibration cohort. Deployment keeps the established, uncalibrated score contract. The original test partition was not rescored for this release. See [development results](reports/experiment-v2.json) and [active artifact metadata](models/metadata.json).

## Explanations and limitations

Permutation SHAP explains class-1 scores with 64 synthetic background transactions. The reference and signed changes add to the displayed score. Contributions are model associations and may distribute across correlated features. Availability and subgroup performance have not been verified in an external banking system. Internal precision is modest, and many recorded fraud cases were missed. Changes to data, score calibration, handling costs or review capacity require fresh evaluation. Use trusted model artifacts and human interpretation.
