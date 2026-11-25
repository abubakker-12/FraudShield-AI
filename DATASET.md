# Dataset and model provenance

| Field | Details |
|---|---|
| Original dataset | Bank Transaction Fraud Detection Dataset |
| Publisher | Nafiul Islam (`nafiulislam490`) |
| Original source | `https://www.kaggle.com/datasets/nafiulislam490/bank-transaction-fraud-detection-dataset?resource=download` |
| Dedication | [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) |

The project owner confirmed the original Kaggle URL and verified the CC0 dedication. A [public snapshot of Kaggle’s dataset metadata](https://github.com/zzfcharlie/kaggle-fetch/blob/master/out_fraud_50/datasets/nafiulislam490__bank-transaction-fraud-detection-dataset.md) independently lists the same publisher, source, schema and CC0 designation. The original page and direct metadata API returned HTTP 404 during verification. The local CSV differs in byte size from the catalogue entry, so provenance relies on the owner’s confirmation.

## Usage and preprocessing

The published description characterizes the source as synthetic banking data. The local CSV contains 1,000,000 rows spanning 2020–2024, with 55,255 recorded fraud labels. The application uses the data for offline EDA, training and evaluation. Training parses day-first dates, uses chronological and customer-disjoint splits, and fits scaling and one-hot encoding on the training sample only. XGBoost uses class weighting. Customer and transaction IDs are excluded from prediction inputs. `fraud_type` is excluded because it identifies the target.

The supplied CSV and notebook remain local. The public app loads `data/demo.csv`, which contains 500 independently sampled synthetic feature records with fictional `DEMO-` identifiers. The generator uses the training categories and synthetic numeric draws. The public reports show aggregates and evaluation, and the screenshots show the app using synthetic demo records. The original visual reference stays local.

## Trained artifact

The shipped model is the existing depth-4 XGBoost and its fitted preprocessing, version `fs1-6430f15b1824`, at threshold `0.640148937702179`. Its SHAP reference matrix is derived from 64 synthetic demo records. The artifact and evaluation metadata matched the recovery baseline, and the release audit did not refit them. The held-out results describe this dataset and an uncalibrated model score. See the [model card](MODEL_CARD.md) and [evaluation metadata](models/metadata.json).

The dataset’s CC0 dedication covers the upstream dataset. The repository does not assign a separate software licence by implication.
