# BirdCLEF+ 2026 — Kaggle-only workflow

Everything runs in Kaggle notebooks. The 16 GB competition dataset is never
downloaded locally — all training and inference happens inside Kaggle.

See [PLAN.md](PLAN.md) for strategy.

## Layout
```
configs/           YAML experiment configs (reference only; Kaggle notebooks set config inline)
data_prep/         Helpers: folds, pseudo-labels, embedding extraction
src/               Library: audio, datasets (cache + on-the-fly), models, losses, train, export
notebooks/
  eda.ipynb                Kaggle EDA
  kaggle_train_cnn.ipynb   Kaggle GPU training for Model A (on-the-fly ogg -> mel, no cache)
  submission.ipynb         Kaggle CPU submission (<= 90 min)
```

## Kaggle workflow

### 1. Upload this repo as a private utility Kaggle Dataset
Create a Kaggle Dataset named **`birdclef2026-code`** containing the `src/` and
`data_prep/` folders. Bump its version whenever you edit the code locally.

### 2. EDA
Open `notebooks/eda.ipynb` as a new Kaggle notebook. Attach `birdclef-2026` and
`birdclef2026-code`. Run all — verifies class coverage + site-based folds.

### 3. Train Model A (Kaggle GPU)
Open `notebooks/kaggle_train_cnn.ipynb`. Accelerator = **P100 / T4 x2**.
Attach `birdclef-2026` + `birdclef2026-code`. Run all — uses
`src.ogg_dataset.OggOnTheFlyDataset` so no mel cache is written (avoids the
20 GB notebook output cap). Outputs `best.pt` + `effnet_b0_fold0.int8.onnx` to
`/kaggle/working/`. Save Version → *Output → New Dataset* named
**`birdclef2026-weights`**. For extra folds, duplicate the notebook, set
`val_fold=1,2,...`, and save each into a new version of the same weights dataset.

### 4. (Optional) Model B — external embedding head
Attach a Perch v2 / BirdNET Kaggle Model; adapt `data_prep/extract_embeddings.py`
+ `src/train_emb.py` into a new Kaggle training notebook (same pattern as
`kaggle_train_cnn.ipynb`). Export via `src.export_onnx.export_embhead` +
`quantize_int8`; save into `birdclef2026-weights`.

### 5. Submit
Open `notebooks/submission.ipynb`. **CPU runtime, internet OFF.** Attach
`birdclef-2026` + `birdclef2026-weights` (+ Perch Model if using Model B). Set
`USE_MODEL_B = True` only when `embhead_fold*.int8.onnx` exists in the weights
dataset. Run all, confirm dry-batch latency, then Save Version to submit.

## Design notes
- **No local mel cache.** `OggOnTheFlyDataset` decodes ogg + computes log-mel in
  each DataLoader worker. This sidesteps the ~20 GB Kaggle output cap that an
  offline shard store would blow through on 46k files.
- **Site-based validation.** `data_prep/make_folds.build_folds` groups labeled
  soundscape files by recording site (`_S\d+_` in the filename) and assigns
  whole sites to folds — never split a site across train/val.
- **Submission constraints.** CPU ≤ 90 min, no internet, 234 species columns
  matching `taxonomy.csv` order. Enforced in `notebooks/submission.ipynb`.
