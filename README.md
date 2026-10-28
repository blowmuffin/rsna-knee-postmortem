# RSNA Knee Abnormality Detection (Kaggle, 2026): a post-mortem

Code, full experiment log and post-mortem for [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection): detect 12 knee findings from multimodal knee MRI, scored by macro ROC AUC. Notebook-only submissions; the competition closed on 2026-10-22.

**I did not medal. This repo records what I tried, what the winners did differently, and why I fell short.**

## Result

| | |
|---|---|
| **Private leaderboard rank** | **1,718 / 6,299 teams** (final) |
| Private AUC | 0.942 |
| Best public AUC | 0.958 |
| My own models alone | 0.938 public / 0.924 private |
| Needed for gold | 0.954 private (winners: 0.960) |

## Why it fell short

1. **I scored with a public recipe, not my own models.** Both scored submissions were blends on the public 0.958 notebook. Over a thousand teams submitted the same recipe and all landed on 0.942 private.
2. **I tuned to the public leaderboard.** The public blends dropped 0.016 from public to private (0.958 → 0.942). The winners' own models dropped only about 0.01.
3. **My best work was never submitted.** My strongest own set (OOF AUC 0.908) was built, but the session ended before I used it.
4. **The real gap was labels, not models.** Every top team relabelled the training data with an LLM that read the MRI images as well as the report. Those labels scored 0.935–0.947 against the expert-labelled gold studies. My labels came from the reports alone and plateaued at 0.89–0.91. Single models trained on the winners' labels hit 0.954–0.958 private; ensembling added at most 0.002.

The full analysis, with the top-5 write-ups, is in [`reports/2026-10-25 post-mortem top solutions.md`](reports/2026-10-25%20post-mortem%20top%20solutions.md).

## What I built

- **Pipeline (`knee.py`, `build.py`)**: 2.5D multiple-instance models over MRI slice stacks (EfficientNet, ConvNeXt), with pseudo labels and rubric labels derived from the radiology reports. Everything ran as Kaggle kernels; `build.py` generates the prep, train, infer and blend kernels in `kernels/`.
- **Blending and evaluation**: rank blends with public notebooks, out-of-fold AUC tooling (`combos.py`, `stack.py`, `oof_report.py`).
- **Tests**: `python test_knee.py` runs the CPU self-checks.

## Layout

| Path | What |
|---|---|
| `knee.py`, `build.py`, `test_knee.py` | Model code, kernel builder, self-checks |
| `kernels/` | Generated Kaggle kernels |
| `runs/` | Downloaded kernel outputs: logs, OOF predictions, configs |
| `labels/` | My derived label sets (study IDs and model outputs only) |
| `EXPERIMENTS.md`, `HANDOFF.md` | Every experiment and result, plus session notes |
| `reports/`, `research_notes/` | Post-mortem and research |
| `baseline/`, `ref/` | Public Kaggle notebooks used as bases and references (their authors' work) |

**Data is not included.** The competition data, including the radiology report text, is licensed under the competition rules and can't be redistributed. Get it from the [competition data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data).
