# BirdCLEF+ 2026

Kaggle-only workflow. The 16 GB competition dataset stays on Kaggle; all
training and inference run inside Kaggle notebooks.

## Layout

```
configs/                 YAML experiment configs (reference; notebooks set config inline)
data_prep/               Helpers: folds (site-grouped, segment-balanced), pseudo-labels
src/                     Library: audio, datasets, models, losses, train, metrics, taxonomy
notebooks/
  eda.ipynb                  EDA + class coverage + fold inspection
  kaggle_train_cnn.ipynb     Train Model A - timm EfficientNet-B0 on log-mel (Kaggle GPU)
  train_model_b_perch.ipynb  Train Model B - MLP head over frozen Perch v2 embeddings (Kaggle GPU)
  evaluate_best.ipynb        True full-soundscape macro AUC for any saved best.pt (Kaggle GPU)
  submission.ipynb           Submission notebook (Kaggle CPU, <= 90 min)
output/                  Local-only weights cache (gitignored)
scripts/                 Local utilities (path patches, etc.)
```

## Models

- **Model A** - timm `tf_efficientnet_b0.ns_jft_in1k`, 1-channel log-mel,
  234 classes. Saved as `best.pt`.
- **Model B** - frozen Google Perch v2 (Bird Vocalization Classifier) ->
  1280-d embedding -> 3-layer MLP head (1280 -> 512 -> 234). Saved as
  `best_emb.pt`. Training oversamples soundscapes 50/50 vs focal to avoid
  domain shift on validation.

## Required Kaggle inputs

| Notebook                      | Datasets / Models to attach                                         |
| ----------------------------- | ------------------------------------------------------------------- |
| `eda.ipynb`                   | `birdclef-2026`, `birdclef2026-code`                                |
| `kaggle_train_cnn.ipynb`      | `birdclef-2026`, `birdclef2026-code`                                |
| `train_model_b_perch.ipynb`   | `birdclef-2026`, `birdclef2026-code`, `google/bird-vocalization-classifier` |
| `evaluate_best.ipynb`         | `birdclef-2026`, `birdclef2026-code`, dataset containing `best.pt`  |
| `submission.ipynb`            | `birdclef-2026`, dataset with `best_emb.pt`, `google/bird-vocalization-classifier` |

The `birdclef2026-code` dataset is this repo's `src/` + `data_prep/` packaged
as a private Kaggle Dataset. Bump its version after each local code change.

## Validation

`data_prep/make_folds.build_folds` does a site-grouped 5-fold split balanced
by segment count over labeled soundscapes. Whole sites stay in one fold.
`evaluate_best.ipynb` reports macro AUC over all ~840 labeled soundscape
segments (much larger and more reliable than the per-epoch val window).

## Submission constraints

CPU only, no internet, <= 90 min, output `submission.csv` with `row_id` plus
234 species columns in `taxonomy.csv` order.
