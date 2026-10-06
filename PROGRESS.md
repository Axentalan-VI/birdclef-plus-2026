# Progress

## Current status

Finished. The competition closed on 2026-06-03 with a final standing of
**3073 of 4094 teams**; best public score **0.81943** macro ROC-AUC from 6
scored submissions.

The repo holds the full pipeline: on-the-fly OGG to log-mel decoding over ~46k
recordings, a log-mel CNN trained from scratch, a frozen Perch v2 embedder with
an MLP head, an ensemble of the two, and INT8 ONNX export so 234 species score
within the 90-minute CPU-only inference limit.

## Last session (2026-10-06)

- Recovered the submission history into `EXPERIMENTS.md`; it existed nowhere.
- Corrected the README, which claimed the competition was still running four
  months after it closed.
- Stripped a UTF-8 BOM that was swallowing the README's top heading on GitHub.
- Stopped `pytest` collecting `scripts/test_local.py`, a manual sanity script.

## Open issues

- **Ten of 16 submissions returned no public score** and the cause was never
  diagnosed.
- **No cross-validation numbers were ever recorded**, so the 0.797-0.819 spread
  cannot be distinguished from noise.
- **No automated tests.** The only `test_*.py` is a manual script that
  downloads audio and runs a checkpoint.

## Next steps (prioritized)

1. If this is revisited for a later edition: run the documented validation and
   log a CV score per configuration before submitting anything.
2. Add a real test for the log-mel transform and the ONNX export, the two
   places where a silent change breaks inference inside the time limit.

## Decisions & rationale

- Everything runs on Kaggle; the 16 GB dataset is never downloaded locally,
  because the local GPU is a 2 GB MX550.
- INT8 ONNX rather than PyTorch at inference: the submission must score 234
  species on CPU within 90 minutes.
