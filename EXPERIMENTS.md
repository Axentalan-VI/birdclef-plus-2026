# Experiments

Every submission to [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026),
newest first, from this account's submission history. Metric: macro ROC-AUC,
higher is better.

16 submissions were made; **6 returned a public score and 10 returned none** —
those are notebook runs that completed but produced no scored submission file.

| ID | Date | Notebook / version | Public LB | Notes |
|----|------|--------------------|-----------|-------|
| E016 | 2026-05-24 | submission v13 | **0.81943** | **best** |
| E015 | 2026-05-19 | submission v12 | 0.80290 | |
| E014 | 2026-05-19 | submission v11 | — | no score returned |
| E013 | 2026-05-14 | submission v10 | 0.80431 | |
| E012 | 2026-05-14 | submission v9 | 0.74353 | clear regression |
| E009-E011 | 2026-05-11 | submission v6-v8 | — | no score returned |
| E006-E008 | 2026-05-05 | submission v2-v4 | — | no score returned |
| E005 | 2026-05-04 | notebook5404 v7 | 0.79658 | |
| E004 | 2026-04-28 | notebook5404 v6 | 0.80481 | first strong result |
| E001-E003 | 2026-04-28 | notebook5404 v2-v5 | — | no score returned |

## What this table says

- **The whole usable range is 0.797-0.819**, five submissions inside 0.022.
  Only v9 (0.744) is clearly different, and clearly worse.
- **0.80481 on 2026-04-28 was already within 0.015 of the final best**, reached
  on the first scored attempt. Four weeks of further work bought +0.015.
- **Ten submissions produced no score at all.** Each one cost a notebook run
  and a submission slot and returned nothing. Whatever was failing was never
  diagnosed, because nothing was logged.
- **No cross-validation score exists for any row.** The README describes a
  validation strategy; its outputs were never recorded, so there is no way to
  tell whether 0.819 was a real improvement or leaderboard noise.

## Next time

Record the CV score beside every submission, and stop submitting after a run
that returns no score until the cause is known.
