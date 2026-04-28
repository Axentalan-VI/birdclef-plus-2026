# BirdCLEF+ 2026 — Kaggle Solution Plan

## 1. Problem recap
- **Task:** Multi-label acoustic species ID (birds, amphibians, mammals, reptiles, insects) in Pantanal soundscapes.
- **Classes:** 234 species/sonotypes. Not all appear in the hidden test set.
- **Input at test time:** ~600 one‑minute `.ogg` soundscapes @ 32 kHz in `test_soundscapes/`. Predict probability per 5‑sec window → ~7,200 rows × 234 cols.
- **Metric:** Macro‑averaged ROC‑AUC, skipping classes with no positives.
- **Kaggle submission constraints (hard):**
  - Notebook‑only, **CPU only**, ≤ **90 min** runtime.
  - **No internet**. External pre‑trained models must be attached as Kaggle Datasets/Models.
  - Output: `submission.csv` with `row_id` + 234 species columns.

Implication: this is primarily an **engineering problem** of running compact, fast models on CPU within 90 minutes, backed by heavy offline GPU training.

---

## 2. Data strategy

### 2.1 Sources
- `train_audio/` — focal XC + iNat recordings (weakly labeled, one `primary_label` + `secondary_labels`).
- `train_soundscapes/` — unlabeled + **labeled subset** (`train_soundscapes_labels.csv`, 5‑sec segments). Same sites as test.
- `taxonomy.csv` — defines 234 submission columns.

### 2.2 Label handling
- Build a 234‑dim multi‑hot target per training clip.
- From `train_audio`: set `primary_label = 1`, `secondary_labels = 1` (lower weight, e.g. 0.5 during BCE).
- From labeled `train_soundscapes`: treat each 5‑sec segment as a strongly labeled sample (highest weight). **These are gold** — same domain as test.
- Classes absent from `train_audio` but present in labeled soundscapes: rely on soundscape labels (key for sonotypes/non‑bird classes).

### 2.3 Pre‑processing (offline, cache to dataset)
- Resample to 32 kHz mono.
- Convert to **mel‑spectrogram**: `n_fft=1024`, `hop=320` (10 ms), `n_mels=128`, `fmin=20`, `fmax=16000`, log‑mel (power→dB), per‑image mean/std norm.
- Crop/pad to fixed 5‑sec windows (target length 500 frames).
- Save as `.npy` / `.npz` shards on a Kaggle Dataset to avoid re‑decoding 16 GB of ogg in the CPU notebook.
- **Clip selection from focal audio:** take top‑energy 5‑sec crops (signal‑power ranked) + center crop; yields 2–5 crops per XC file.

### 2.4 Train/val split
- **Validation = held‑out labeled `train_soundscapes` files** (by filename, not segment) → mimics the test distribution exactly.
- Secondary validation on stratified `train_audio` to monitor focal→soundscape generalization.

---

## 3. Modeling (offline GPU training, CPU inference)

### 3.1 Backbone choices (CPU‑friendly)
Prioritize fast, quantizable CNNs:
1. **EfficientNet‑B0 / B1** (timm) on mel‑spec — canonical BirdCLEF workhorse.
2. **MobileNetV3‑Large** — smaller, very fast on CPU.
3. **TF‑EfficientNetV2‑S** — optional, only if 90‑min budget allows.
4. **Embedding path (recommended):** use **Perch v2 / BirdNET** embeddings (both available as Kaggle Models, license‑clean for non‑commercial). Run the frozen embedder once per 5‑sec window, train a small MLP / linear head on top. This is extremely fast at inference and competitive.

Plan to train **2 families in parallel**:
- A. EfficientNet‑B0 on log‑mel (fine‑tuned end‑to‑end).
- B. Linear/MLP head on Perch v2 embeddings.

Ensemble (A+B) at inference.

### 3.2 Training recipe (GPU, offline)
- Loss: `BCEWithLogitsLoss` with class‑frequency‑based **pos_weight** (cap at ~50) or **focal BCE**.
- Sample weights: soundscape‑labeled > primary label > secondary label.
- Augmentations (spec‑level): SpecAugment (time+freq masking), random time shift, Gaussian noise, mixup (α=0.5), **random background mixing** with unlabeled `train_soundscapes` segments (strong domain adaptation trick).
- Optimizer: AdamW, cosine schedule, 25–40 epochs, AMP.
- **Pseudo‑labeling round 2:** after v1, predict on unlabeled `train_soundscapes`, threshold → add as soft labels, retrain. This has been the dominant trick in prior BirdCLEF editions.
- Train 5‑fold (by soundscape site) → keep best 2–3 folds for ensemble to fit time budget.

### 3.3 Export for Kaggle
- Convert PyTorch → **ONNX** (opset 17) → **ONNX Runtime CPU** with graph optimization + **int8 dynamic quantization** (`onnxruntime.quantization`).
- Alternatively OpenVINO IR for Intel CPUs (Kaggle uses Xeon). Benchmark both; OpenVINO often 2–3× faster.
- Upload weights + ONNX files as a **private Kaggle Dataset** attached to the submission notebook.

---

## 4. Inference pipeline (the Kaggle notebook)

Target budget: ~600 files × 12 windows = 7,200 inferences in ≤ 80 min (leave 10 min headroom).

### 4.1 Pipeline
```
for each ogg in test_soundscapes/ (parallel load with soundfile + ThreadPool):
    y = load_32k_mono(ogg)               # 60 s
    segs = split_into_5s(y, step=5s)     # 12 segments
    mels = batch_logmel(segs)            # vectorized on CPU
    # Model A: CNN on mels (ONNX INT8)
    logits_a = ort_session_a.run(mels_batch12)
    # Model B: embeddings head
    emb = perch_embed(segs)              # may itself be TFLite/ONNX
    logits_b = head_b.run(emb)
    probs = sigmoid( w_a*logits_a + w_b*logits_b )
    write rows [filename_05, filename_10, ..., filename_60] × 234 cols
```
- Batch all 12 segments of a file in a single ORT call.
- Use `num_threads = os.cpu_count()`, set `sess_options.intra_op_num_threads` accordingly.
- Pre‑allocate numpy arrays for the submission to avoid pandas append overhead. Build as dict → single `pd.DataFrame.to_csv`.

### 4.2 Post‑processing
- **Temporal smoothing:** for each (file, class), blend current prob with neighbors: `p_t ← 0.6*p_t + 0.2*p_{t-1} + 0.2*p_{t+1}`. Historically +0.005–0.01 on LB.
- **File‑level prior:** `p_t ← p_t * (0.8 + 0.2 * max_t_in_file(p))` to amplify classes that look confidently present somewhere in the file.
- **Rank normalization per class** across the test set before sigmoid blending (optional, helps ROC‑AUC macro).
- **Do NOT threshold** — metric is ROC‑AUC, keep continuous scores.

### 4.3 Time budget table (target)
| Stage | Time |
|---|---|
| Import + model load | 1 min |
| Audio load (threaded) | 4–6 min |
| Mel computation | 3 min |
| Model A (EffNet‑B0 INT8 ONNX) | 25–35 min |
| Model B (Perch emb + MLP) | 15–20 min |
| Post‑process + CSV | 1 min |
| **Total** | **50–65 min** |

---

## 5. Step‑by‑step execution plan

1. **Env setup (local/GPU box):** Python 3.11, PyTorch, timm, torchaudio, librosa, onnx, onnxruntime, openvino.
2. **EDA notebook:** class distribution, clip durations, labeled‑soundscape coverage per class, site map.
3. **Cache builder:** turn `train_audio` + labeled `train_soundscapes` into 5‑sec mel `.npy` shards → upload as Kaggle Dataset.
4. **Baseline (Model A v1):** EfficientNet‑B0, 20 epochs, no pseudo‑labels. Validate on held‑out labeled soundscape files.
5. **Embedding baseline (Model B):** extract Perch v2 embeddings, train MLP head. Confirm ensemble > single models.
6. **Pseudo‑label round:** predict on unlabeled `train_soundscapes`, keep top‑confidence positives (per‑class thresholds). Retrain Model A.
7. **Export + quantize:** ONNX INT8 + OpenVINO. Benchmark on a Kaggle CPU notebook with a dummy 60‑sec clip to verify throughput.
8. **Inference notebook v1:** single‑model submission; confirm runtime < 70 min and format valid.
9. **Ensemble + TTA:** add Model B, temporal smoothing, file‑level prior. Submit.
10. **Iterate:** try MobileNetV3 / EffNetV2‑S, extra folds, weighted ensemble tuned on the held‑out soundscape fold.

---

## 6. Repository layout (suggested)

```
birdclef-2026/
├── configs/                 # YAML configs per experiment
├── data_prep/
│   ├── build_mel_cache.py
│   ├── make_folds.py
│   └── pseudo_label.py
├── src/
│   ├── datasets.py          # clip sampling, mixup, bg mixing
│   ├── models.py            # timm CNN + Perch head
│   ├── losses.py            # weighted BCE / focal
│   ├── train.py
│   └── export_onnx.py
├── notebooks/
│   ├── eda.ipynb
│   └── submission.ipynb     # the CPU inference notebook uploaded to Kaggle
├── kaggle_assets/           # ONNX weights + tiny python utils packaged as dataset
└── PLAN.md
```

---

## 7. Key risks & mitigations
- **Domain gap (focal XC → soundscape):** mitigated by background mixing + pseudo‑labels + soundscape‑weighted loss + soundscape‑based validation.
- **Rare / sonotype classes:** rely on labeled soundscapes + pos_weight; skipped‑class metric means we mostly need to avoid hurting common classes.
- **CPU timeout:** quantize to INT8, OpenVINO, batch all windows per file, prefer EffNet‑B0 / MobileNetV3 over larger nets. Always benchmark before full submit.
- **No internet:** bundle every dependency (ONNX weights, Perch model) as attached Kaggle Dataset; ensure notebook imports only installed packages.
- **Label noise in XC:** lower loss weight for secondary labels; use `rating` as sample weight.

---

## 8. First concrete action
Create `data_prep/build_mel_cache.py` and an `eda.ipynb` that:
- loads `train.csv` + `train_soundscapes_labels.csv`,
- reports class coverage per source,
- materializes the held‑out soundscape validation split file list.

Everything downstream depends on that cache + split.
