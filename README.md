# BirdCLEF+ 2026

End-to-end solution for the [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)
Kaggle competition: identify which of **234 bird, amphibian, mammal and insect species**
are vocalizing in 5-second windows of unlabeled tropical soundscape recordings.

The competition provides ~30k focal recordings (single-species, weak labels)
plus a smaller set of labeled soundscape segments for validation. The hidden
test set is multi-species soundscapes, so the central challenge is bridging
the **focal -> soundscape domain gap**.

This repo runs **entirely on Kaggle** (the 16 GB dataset is never downloaded
locally). Submissions must run on **CPU only, no internet, within 90 minutes**.

## Approach

Two complementary models, ensembled at submission time:

### Model A - log-mel CNN (from scratch)
- Backbone: timm `tf_efficientnet_b0.ns_jft_in1k` (5.3 M params, ImageNet-pretrained).
- Input: single-channel log-mel spectrogram, 128 mels, 5 s @ 32 kHz, n_fft=1024, hop=320.
- Head: linear (1280 -> 234) with dropout.
- Training: BCE with class-balanced `pos_weight`, mixup, label smoothing,
  cosine LR with warmup, on-the-fly ogg decoding (`OggOnTheFlyDataset`) so
  no mel cache hits the 20 GB Kaggle output cap.
- Validation: site-grouped, segment-balanced 5-fold over labeled soundscapes.
- Output: `best.pt`. Current LB: **0.804**.

### Model B - Perch v2 frozen embedder + MLP head
- Embedder: Google's [Bird Vocalization Classifier (Perch v2)](https://www.kaggle.com/models/google/bird-vocalization-classifier),
  used as a **frozen** feature extractor (1280-d embedding per 5 s @ 32 kHz clip).
- Head: 3-layer MLP `LayerNorm(1280) -> Linear(1280, 512) -> GELU -> Dropout
  -> Linear(512, 512) -> GELU -> Dropout -> Linear(512, 234)` (~0.9 M params).
- Training: BCE + mixup, AdamW, cosine LR. Embeddings are pre-extracted once
  and trained over - one epoch is ~0.7 s.
- **Key trick:** the train pool is ~95 % focal but validation is 100 %
  soundscape. Without resampling the head overfits focal style and val AUC
  collapses after epoch 1 (0.87 -> 0.62). A `WeightedRandomSampler` that
  oversamples soundscape rows to ~50 % of every batch fixes this entirely.
- Output: `best_emb.pt`. Full-soundscape macro AUC: **0.83**.

### Ensemble
At submission time, both models score every 5 s window; their sigmoid
probabilities are averaged (`probs = w * model_A + (1 - w) * model_B`).
The two models make uncorrelated errors (one learns mel patterns from
scratch, the other inherits Perch's pretrained bird-acoustic prior), so the
average lifts LB above either model alone.

## Validation strategy

`data_prep/make_folds.build_folds` builds **site-grouped, segment-balanced
5-fold splits** over labeled soundscapes:

- Whole recording sites (`_S\d+_` in the filename) stay in one fold - never
  split a site across train/val (otherwise nearby segments leak across).
- Folds are balanced by total **segment count**, not file count, since
  files vary in length.
- Per-epoch validation uses one fold (~150 segments). For final model
  comparison, `evaluate_best.ipynb` runs over **all ~840 labeled segments**,
  which is the only val number stable enough to track <1 % LB-relevant
  changes.

## Repo layout

```
configs/                 YAML experiment configs (reference; notebooks set config inline)
data_prep/               Folds (site-grouped, segment-balanced), pseudo-labels
src/                     Audio I/O, datasets, models, losses, training loop, metrics
notebooks/
  eda.ipynb                  Class coverage + fold inspection
  kaggle_train_cnn.ipynb     Train Model A (Kaggle GPU)
  train_model_b_perch.ipynb  Train Model B (Kaggle GPU; needs Perch v2 model)
  evaluate_best.ipynb        Full-soundscape macro AUC for any saved best.pt
  submission.ipynb           Submission notebook (Kaggle CPU)
output/                  Local-only weights cache (gitignored)
scripts/                 Local utilities
```

## Required Kaggle inputs per notebook

| Notebook                      | Datasets / Models to attach                                         |
| ----------------------------- | ------------------------------------------------------------------- |
| `eda.ipynb`                   | `birdclef-2026`, `birdclef2026-code`                                |
| `kaggle_train_cnn.ipynb`      | `birdclef-2026`, `birdclef2026-code`                                |
| `train_model_b_perch.ipynb`   | `birdclef-2026`, `birdclef2026-code`, `google/bird-vocalization-classifier` |
| `evaluate_best.ipynb`         | `birdclef-2026`, `birdclef2026-code`, dataset containing `best.pt`  |
| `submission.ipynb`            | `birdclef-2026`, dataset with `best_emb.pt`, `google/bird-vocalization-classifier` |

`birdclef2026-code` is this repo's `src/` + `data_prep/` packaged as a
private Kaggle Dataset; bump its version after each local code change.

## Submission constraints

CPU only, no internet, <= 90 min, output `submission.csv` with `row_id`
plus 234 species columns in `taxonomy.csv` order.
