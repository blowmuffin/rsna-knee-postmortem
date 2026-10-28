# Leaderboard research, 2026-10-15 ~19:20 UTC

Sources: the Kaggle forum (topic ids below), the top public notebooks, the efficiency LB output, and the Kaggle API. Numbers are as posted by their authors unless marked *measured*.

## Deadlines (Kaggle API, measured)
- **Team merger and new-entrant deadline: 2026-10-15 23:59 UTC**, about 4.5 h after this was written.
- Final submission deadline: 2026-10-22 23:59 UTC. 5 submissions/day, team size ≤5.

## The leaderboard itself
- The public LB is computed on ~30% of the test set and the private LB on the other 70% (LB header, quoted in 749570). Ties are broken by submission time.
- The public score is printed to 3 decimals, and 0.957 and 0.956 are blocks of 276 and 908 teams *(measured 10-15)*. A change under ~0.002 cannot be seen on the public LB.
- The top public notebooks are inference-only rank blends of public checkpoints, and they top out at 0.9576. Nothing above that is public.

## Eligibility
- **LLM-API labels are explicitly allowed.** Host, topic 733965: "You can use LLM API, such as those from OpenAI, to read the reports to generate the labels." Our Claude rubric labels are fine.
- **OAI is unresolved.** On 9-21 the host said OAI is fine if it is free and needs no institutional review (741819). Topic 749145 asks about the NIH geo restriction (China, Russia, Venezuela cannot register) and has had no host answer. Tucker Arrants (#1) warned that OAI "may not be an allowed external data source" (746792).
  - Our base (goodpjw v21) uses nartaa's Raptor checkpoints, trained partly on OAI, and heliosli's anatomy network.
  - Risk to medals: low to moderate; enforcement is usually checked only for prize winners. Risk to prizes: real.
- Gold-58 labels are themselves noisy. hengck23 found a gold study labelled Effusion = 1 whose report says "No knee effusion is present."

## What strong teams report
Single architectures, no external data:

| Team (rank) | Model | Public LB |
|---|---|---|
| azamat1ch (#4) | ConvNeXt-T, 5-fold | **0.959** |
| azamat1ch (#4) | EfficientNetV2-S, 5-fold | 0.956 |
| azamat1ch (#4) | CoAtNet, 5-fold | 0.952 |
| Musa Tur Farazi (#83) | single model, single fold | 0.956 |
| Tanuki_boosting (#42) | single model, 5-fold | 0.955 |
| CoolinLai (#27) | **ResNet-50, 224 px**, 5-fold | 0.954 |
| colum2131 (#29) | ConvNeXt-T 2.5D (pseudo labels) | 0.953 |
| colum2131 (#29) | DINOv2 ViT-S (pseudo labels) | 0.951 |
| NguyenThanhNhan (#46) | Qwen3.5-2B VLM, LoRA, 384 px, single fold | 0.950 |
| **us** | EffNet-B0 MIL, 288 px, 5-fold | **0.926–0.938** |

(Sources: topic 735304 "Best single-model score"; someone also reported a 0.960 single submission, 748111, method unknown.)

**On like-for-like backbones we are about 0.02 below these teams.** That is the "red flag" gap HANDOFF noted on 10-03, and it is not a prep bug (checked 10-03).

**Every strong team names the same lever: labels.**
- hengck23: "model is not the key. it is your labels." His numbers: an LLM label set at 0.89 on gold gives ~0.95 train AUC, ~0.92 on gold and **~0.930 public LB**. That is exactly where our own models land.
- Raymond Yuen (#42): multiple weak-label sources (public, API, local) → teacher models → pseudo labels → final model. "Different teachers seem to help different targets." Label API cost was under $5.
- Chris Deotte: with pseudo labels, "every single model can match our best ensemble CV score".
- Prateek Grover: train only on cells where several LLM labellers agree with a strong signal (>0.9 or <0.1), otherwise ignore the cell. About 25% of cells were dropped, and both CV and public LB improved.
- Kirderf: read every report several times, fuse the readers with their spread and confidence, re-read the cells most likely to be wrong, and fill silent cells with measured priors rather than 0.
- Danial Zakaria's 0.949 label recipe (746792): their own Gemini-3-Flash labels averaged with steven v4_blend (where v4 = 0.25, the "not addressed" marker, keep their own), then averaged with riadmohamed42's JEV HYBRID scores. JEV was tuned on gold by its author, so gold is not independent for it.

## Public recipes worth copying
- **S6 retrained on other labels = 0.950 single model** (749359). nartaa retrained heliosli's S6 pipeline on their own labels: all data 0.950, folds 1–4 0.948, fold-0 OOF 0.907 (our best is 0.896).
  - Recipe: convnext_small.fb_in22k_ft_in1k, per-finding spatial attention plus window-attention MIL. Full field of view **plus 100 mm knee-centred crops** placed by the anatomy V-Net. 320 px cache cropped to 288, 3 slices per window at 3.5 mm, 64 windows per study, 16 epochs, AdamW (1e-4 encoder / 1e-3 head, wd 0.05), batch 2, cosine with 5% warm-up.
  - Weights are public: `nartaa/rsna-knee-s6fall-weights` (`E_prior_s42_fall.pt`, CC BY-NC). Our base does not include this model.
  - Training code: they cite heliosli's `compact_train.py`, which is **not** in `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` under that name. Still to locate.
- Raptor CoAtNet 0.949 (746792): 384 px, 12 random windows per step, 16 epochs, BCE pos_weight (1−p)/p clipped to [1, 10]. OAI added +0.005. Anatomical mirror at test time added +0.001; a plain width flip on every plane did not help. 16 epochs take ~22 h on an 8 GB laptop GPU.
- Knee-centred crops show up in both S6 (100 mm) and bliverrigpdoge's add-on (+0.001 at weight 0.2). That supports our zoomed-cache idea.

## Execution risks
- A hidden-test rerun can fail even when the visible run succeeds (749623). Kaggle's note says the hidden data "can be larger/smaller/different". Any unhandled exception is a failed submission, so anything we put before the base has to be wrapped so it can never raise.

## Implications for us
1. **Labels are the biggest lever we have not pulled.** We have 8+ label sets locally (`labels/`), including per-cell weights in `rsna-knee-2026-calibrated-soft-targets/train_targets_weights.csv`. A consensus label with a confidence mask costs CPU only to build and ~0.7 GPU-h to A/B (eff31 at 224 px, folds 0–1, vs eff31-mil).
2. If it closes part of the ~0.02 gap, every own model gets better, and the add-on becomes a ~0.95-class member worth weight 0.3–0.5 rather than 0.15.
3. A rubric-only ConvNeXt (plan step 3a) is weaker than ConvNeXt on the improved labels. Swap it once the label A/B is in.
4. Retraining S6 on our labels is the strongest single option, but its training code still has to be located, and its T4 runtime is unknown (it trained in bf16, so on a non-T4 GPU).
