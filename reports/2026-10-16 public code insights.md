# Public code, models and run logs: 2026-10-16

## Sources
- **All 1,339 public notebooks** were listed with each notebook's best public score, votes and attached inputs, using Kaggle's public web listing.
- **Sources:** 233 were pulled, covering every notebook at ≥ 0.94 plus the training, analysis and high-vote ones; `ref/` already held 45 more.
- **Run logs:** 39 key notebooks, fetched with `list_kernel_session_output`; plain text in `ref/_logs/`.
- **Descriptions:** the main weight datasets.
- **Embedded code:** heliosli's notebooks carry the full source of `compact_train.py`, `compact_infer.py` and `px_cache.py` as string literals. These are the R6/R7/S6 trainers we failed to find on 10-15. They were extracted with `ast.literal_eval`, never executed; they are in `ref/_heliosli_compact_src/`.
- **Catalog of every notebook:** score, model families, private inputs, and notes on 29 key ones, in [research_notes/public_code_catalog.csv](../research_notes/public_code_catalog.csv).

Scores below are public LB numbers the authors measured on their own submissions unless marked otherwise. Many notebooks are AI-written and quote numbers with no submission behind them; those are ignored.

## 1. How the public frontier was built (measured)

| LB | Recipe | Step that moved it |
|---|---|---|
| 0.891 | pilkwang: 20-member DINOv2-S | first public baseline |
| 0.924 → 0.932 | dreaddevelopment Raptor: single CoAtNet-rmlp-2 384, 2.5D windows, per-finding attention MIL | **slice density and window coverage**: 42 → 62 windows +0.003; 44 → 64 → 80 slices per study 0.926 → 0.928 → 0.932; spreading the same 44 slices wider cost 0.009 |
| 0.937 → 0.943 | mattiaangeli / jiweiliu mega-stacks: DINOv2/v3 folds, RadImageNet heads, 4 CoAt arms, per-label weights | adding correlated members plus per-label weights tuned on the public LB; about 200 forks sit at 0.941 |
| **0.949** | **nartaa: one CoAtNet 384** | OAI MOAKS rows (+0.005), JEV-blend labels (+0.002), 96 slices, native 384 px stored and 320 centre crop at inference, 94 windows, top-3 epoch averaging, anatomical mirror TTA (+0.001) |
| 0.950 | matterhorn: CoAtNet + ConvNeXt MIL; nartaa S6fall (single ConvNeXt-S) | **fewer, more diverse models** beat the 75-model stacks |
| 0.951 / 0.954 | heliosli compact S5 / S6 (ConvNeXt-S, 4 folds) + Raptor, 70/30 | a strong, *different* second model |
| 0.956 | heliosli R7 + Raptor, 50/50 | R7 = S6 + crop 352 + fastMRI auxiliary loss + refit Effusion/Baker's stackers |
| 0.956, ranked higher | + nartaa's unused 224 checkpoint at 35% of the Raptor slot | rank 233 → 156 inside the block |
| 0.957 | heliosli R6 + **flip TTA** + Raptor 384/224 (our base957 = goodpjw v21) | flip TTA with medial↔lateral output swap |
| 0.957 | R7 base + bliverrigpdoge's private KCD25 at 0.2 | knee-centred compartment crops (the only comp-data-only add-on measured to help) |
| **0.958** | heliosli (10-15): R6 flip-TTA + Raptor 384/224, **per-finding Raptor weight**: Synovitis 0.8, Lateral OA 0.2, PF OA 0.6, Baker's 0.35, others 0.5 | per-finding weights set from where each model is strong; R7 dropped |

Every notebook at ≥ 0.950 is built from about six model datasets ([catalog](../research_notes/public_code_catalog.csv), `model_families` column). leoprovorov estimates that the public models' errors correlate at ρ ≈ 0.86–0.89, which caps the family at about **0.960**.

## 2. Findings that change our plan

### 2a. Base: move pick B to the 0.958 recipe. It scores higher and is safer.
- Our base957 (goodpjw v21) runs **R7 at 0.4 of the compact slot**. heliosli's own description of R7 is "four folds, crop 352, **fastMRI auxiliary training**".
- fastMRI requires an application naming an institution and a data-sharing agreement. The host's 10-14 rule says datasets that "limit access to users with institutional affiliations clearly do not" meet the external-data criteria.
- R6's own description does not mention fastMRI. The trainer's fastMRI hook (`--fm-aux`, v9.8) dates from 10-11 01:00, around when R6 was submitted, so this is not proven either way.
- heliosli's 0.958 notebook uses **R6 only**. goodpjw's v26 is v21 plus the 0.958 weights but still keeps R7 at 0.4.
- **Action:** pull `heliosli/blend-mine-in-private-lb-luck-to-all-upvoters` into `baseline/helios958/`. Run it through `build.py base` and later `blend base=helios958`. One submission to confirm 0.958.
- The Raptor checkpoints remain OAI-trained. That is unchanged and probably acceptable under the 10-14 "registration only" guidance.

### 2b. Our add-on weight: 0.15 is probably too high. Use ≤ 0.05–0.10 and judge it by rank.
- goodpjw's own measurements with their 2.5D ConvNeXt-T reader (0.929 alone, 3 folds):
  - at full weight on the 0.956 base: **0.954 (a loss)**;
  - at ¼ weight: 0.954, the same as the base;
  - an S6-distilled reader (gold 0.931): 0.953.
- They now ship it at 1/10 and advise "on top of a two-model blend that already covers the reader's strengths, keep it at 0.01 to 0.05 or leave it out".
- mayankraaz saw the same: adding the reader took 0.956 → 0.954.
- Our members (0.938 as an ensemble) are in the same class, so `knee-blend957-eff` at 0.15 is more likely to cost than to gain.
- **Action:** rebuild at `blend=0.95`, i.e. ours 0.05 (0.10 at most), on the new base. Read the result by rank inside the block (forum insight 1).

### 2c. What an add-on needs in order to help: it must be different, not merely good
- The two comp-data-only add-ons that measurably helped are:
  - **knee-centred compartment crops** (bliverrigpdoge KCD25: ConvNeXt-S + EffNetV2-S on medial/lateral compartment crops from the sagittal and coronal series plus axial, ImageNet init, at 0.2: 0.954 → 0.955 and 0.956 → 0.957);
  - **a different view of the same model** (the 224 crop at 35%; mirror TTA).
- A whole-field-of-view MIL reader like ours or goodpjw's mostly duplicates what R6 and Raptor already see.
- **Implication for step 3b:** our zoom cache is image-centred at 110 mm. Its value should come from compartment views (medial and lateral halves at higher pixel density), not from a slightly tighter centre crop. The forum measured image-centred 140 → 110 mm at +0.0005.

### 2d. Mirror augmentation with medial/lateral label swap: the one training trick in the top code we lack
- heliosli's trainer v9.4 `--mirror p`:
  - coronal and axial windows are flipped on width;
  - sagittal windows have their slice order reversed;
  - the targets swap Medial↔Lateral Meniscus and Medial↔Lateral OA;
  - MCL gets no loss on mirrored batches.
- At inference, flip TTA averages the plain and mirrored passes with the outputs swapped back. That is the R7 → R6+flip step (0.956 → 0.957), although R6 and R7 also differ in other ways.
- Our cache is canonical-left, so the same trick applies directly:
  - mirroring = reverse the sagittal slice axis, flip coronal/axial on width;
  - set MCL to NaN, which our `masked_bce` already ignores;
  - swap the 4 columns.
- Cost: about 15 lines in `to_input` / `train_worker` plus a TTA pass in `run_infer`.
- Without training on mirrored inputs, mirror TTA on a canonical-left model would feed it out-of-distribution right-looking knees, so do not add the TTA alone.

### 2e. Training recipe gaps against the strong single models (for any new member)

| | Raptor / nartaa 0.949 | helios S6/R6 | **ours (eff320)** |
|---|---|---|---|
| Backbone | CoAtNet-rmlp-2 384 (73M) | ConvNeXt-S | EfficientNet-B0 (5M) |
| Pixels | native 384 stored, 320 crop | 288 cache, 256/352 crop | 320 cache, 288 |
| Slices | 96/study, every window scored | every 3rd slice × 2 views, ~529 windows/study | 31 per slot × 5 |
| Epochs | 16, top-3 epoch average | 16, EMA | **6** |
| LR | bb 3e-5 / head 1e-3, OneCycle, clip 3 | 1e-4 / 1e-3, cosine, wd 0.05 | 3e-4 single group |
| Loss | BCE + pos_weight | BCE + state heads + pair ranking | masked BCE |
| Aug | intensity jitter only, random windows | crop/shift/rotation/gamma + **mirror + label swap** | rot ±10, scale, gamma |
| Labels | JEV blend (+OAI) | own Opus/state spec | v4b + 0.5 pseudo |

- Dread's gold-58 backbone screen with everything else fixed: CoAtNet-384 0.9025, Swin-B-384 0.8825, CNN-336 0.8833, EffV2-L-480 0.8716, ConvNeXt-B-384 0.8754, MaxViT-384 0.8438. So backbone family matters at a fixed recipe, and the CoAtNet-rmlp-2 is the strongest.
- **We cannot train it on Kaggle:** paiky1995 fine-tuned the public 384 CoAtNet on a Kaggle T4 at **48 min per epoch** on 3,575 studies, and the holdout AUC fell from 0.908 to 0.870.
- ConvNeXt-T at 288 (our running cnx288r) is the closest affordable match to the S6/R6 family. It is also the reason it may add little diversity.

## 3. Confirmed by public code (keep doing)
- Rank averaging per finding. Every top notebook does it.
- Score every window position: 94 → 24 windows cost 0.006 (nartaa).
- Wrap every add-on stage so it can never raise. The top notebooks run each stage in its own process with a time cap and fall back to the base file. Our ours-first layout matches this.
- Report-label sets are interchangeable on gold: 9 public sets lie within each other's confidence intervals on 11/12 findings (karttikjangid: steven_v4 0.893, consensus 0.889, pilkwang 0.870, qwen 0.852; jev_hybrid 0.904 was tuned on gold).

## 4. Measured negatives in public code (don't repeat)
- Fine-tuning the public Raptor CoAtNet on Kaggle T4s: too slow, and the holdout got worse (paiky1995).
- DINOv2/DINOv3 mega-stacks: capped at 0.941–0.943 despite 20+ members. Kaggle-trained DINOv2 runs reach OOF 0.73–0.79 (dehaiwang, stevenleehans, wguesdon).
- Per-label weights tuned on the public LB (mattiaangeli: Lateral Meniscus = 1.00). Treat them as private-LB risk. heliosli's 0.958 weights were set from model strengths "before any run", which is less risky but still unvalidated on the private split.
- Raptor intensity-gain change: 0.940 → 0.922; residual head swaps: no change (prvsiyan's ablation table).
- Narrowing nartaa's inference crop from 320 to 304 or 288: −0.001 / −0.002. A fresh crop-trained model also lost (0.949 → 0.946).
- A third full-FOV reader on a two-model blend: no gain or a loss (goodpjw, mayankraaz).

## 5. Runtime facts from the logs (3 visible studies, so per-stage overhead only)
- Raptor 384 stage ≈ 50 s, 224 stage ≈ 26 s; R6 stage ≈ 6 min (mostly decode, ~529 windows per study, 2 workers).
- The 0.958 notebook finishes in ~8 min on the visible set. Its own guards cap the R6 stage at 8.3 h on the hidden test.
- hengck23's MaViT baseline: 1.5 s/study end to end, 0.935 LB (6 views × 8 slices, 288 px, 130 mm). That is a reference point for the efficiency track, which we are not pursuing.
