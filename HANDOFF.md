# RSNA Knee — handoff (latest: 2026-10-16 20:10 UTC)

**Resuming? Read the LAST section, "ON RESUME (2026-10-17 12:30 UTC)". It is self-contained: state, running kernels, quota, and the next commands in order. Older sections are history.**

**Current plan: the "2026-10-16 plan to the finish" section at the end.**

**Every experiment and its result: [EXPERIMENTS.md](EXPERIMENTS.md). Check it before running anything; add a row the moment a result lands.**

Goal (user /goal): reach #1 on the public LB with as few submissions as possible. Research what top teams do, but also try our own ideas.
User preference: use **Sonnet** (`model: 'sonnet'`) for any multi-agent/Workflow fan-out; never Opus for bulk sub-agents.

## Scores so far
| Submission | LB |
|---|---|
| Public notebook fork (jiweiliu) | **0.943** (best) |
| Own eff31-mil + rad31-mil (MIL head) | 0.936 |
| Public 0.65 + own EffNet v1 | 0.941 |
| Public 0.5 + rad + eff-ps | 0.941 |
| Own EffNet 31-slice pseudo (eff31-ps) alone | 0.926 |
| Own RadImageNet R50 alone | 0.924 |
| Own EffNet v1 alone | 0.916 |

Top 10 needs ~0.958; #1 is 0.961. Our models alone are ~0.925. Blending them into the public notebook has not helped yet.

## OOF (5-fold, vs v4b report labels / vs 58 gold)
- knee-train-eff31-mil: 0.892 / 0.913 (best EffNet: new MIL head, 31-slice cache v1, pseudo 0.5 from knee-train-rad)
- knee-train-rad31-mil: 0.885 / 0.917 (RadImageNet R50, MIL head, pseudo 0.5 from eff31-ps)
- prep v2 cache (knee-prep2a/b/c) runs eff2/rad2: no gain over the v1 31-slice cache
- knee-train-r50 (ImageNet resnet50, lr 1e-4): underfit 0.848 / 0.867. FIXED after the run: knee.py now gives every backbone except RadImageNet and CoAtNet lr 3e-4
- CoAtNet 6 epochs at lr 1e-4: underfit 0.859
- Rank average of eff31-mil + rad31-mil: about 0.895 / 0.918

## Running now
- **knee-train-eff31-rub**: identical to eff31-mil but trained on our rubric labels (`labels=rubric`), a clean label A/B. Compare it to eff31-mil, mainly on gold. y_eval stays v4b, so the "vs labels" number is biased against the rubric run.

## Own ideas tested
- OOF stacker (`stack.py`, cross-label logistic regression): no gain
- DICOM metadata (`runs/knee-meta/meta.csv`): sex present for 95% of studies, NO age; adding sex/scanner/field strength to the stacker gave no gain
- **Rubric-aligned LLM labels** (host label rubric: severity, borderline graded negative):
  - Pilot on the 58 gold studies: 0.910 vs 0.893 for v4b. Mixed 50/50 with OOF predictions: 0.935 vs 0.930.
  - Full labels: dataset `ayushkasare/knee-rubric-labels` (private, `rubric_labels.csv`). Chunks 0-32 were labelled by Opus, the rest by Sonnet (Spearman agreement 0.82-0.97).

## Code (knee.py / build.py / test_knee.py; `python test_knee.py` must print ok)
- `HEAD='mil'`: per-finding attention over all tokens, with slot and slice-position embeddings. Old models load as 'slot' from their config.
- `N_SLICE=31`, `PREP` version: 1 = band 0.1-0.9; 2 = band 0.04-0.96, square crop with zero-padding, same-plane fallback, multi-echo dedupe. run_infer runs one test prep per PREP group.
- `LABEL_SRC` 'v4b' or 'rubric'. config.json records HEAD, N_SLICE, PREP and LABEL_SRC.
- Build commands:
  - `python build.py train name=X backbone=... epochs=N pseudo=a:k1,k2 cache=knee-prep31a,knee-prep31b,knee-prep31c labels=rubric`
  - `python build.py infer|blend trains=k1,k2 [blend=0.5] [nslice=16]`
- Push: `python -m kaggle kernels push -p kernels/<name>`.
- Watcher: `bash wait_kernels.sh k1 k2` (in this folder), with background runs of 25 min or less. OOF comparison: `python oof_report.py runs/a runs/b`.
- Download outputs with `--file-pattern "(\.log|oof\.csv|config\.json)$"`.

## GPU quota
18.4h of 30h used this week; resets 2026-10-10 00:00 UTC. Only 2 GPU sessions can run at once. CPU kernels (prep) are free.

## Next steps
1. When eff31-rub finishes: download it, then run `python oof_report.py runs/knee-train-eff31-mil runs/knee-train-eff31-rub`. Judge the label change on gold.
2. If rubric helps, train `rad31-rub`: `backbone=radimagenet_resnet50 epochs=8 labels=rubric pseudo=0.5:knee-train-eff31-mil cache=knee-prep31a,knee-prep31b,knee-prep31c`, about 4.5h.
3. Final candidate: an infer notebook of the best 2-3 models alone, plus a blend notebook (public 0.5 + ours). Submit only if the ensemble's gold score clearly beats ~0.918. Use one submission at a time.
4. Ideas not yet tried:
   - Retry ImageNet ResNet-50 (it now gets lr 3e-4 automatically). CoAtNet needs more epochs or a higher lr.
   - More epochs with random centre subsets (Raptor-style regularisation).
   - Pseudo weight 0.65-0.7, which the gold pilot favoured.

## Research added at the end of the session (forum topics 745214, 744519)
- Gold-58 is unreliable as an LB proxy. SpeedSci (#118): one label version gave EffNet 0.945 on gold but 0.910 on LB; their EffNet/ResNet at ~0.89 gold reach **0.940 LB**. Tucker (#40): simple Qwen labels; most of his models are in the low 0.95s on LB; label tweaks barely matter. SpeedSci: changing labels moved their DINOv2 from 0.931 to 0.942 LB. So the rubric-label gain must be confirmed on LB, not gold.
- **Red flag:** at equal gold, ours LB-underperforms by ~0.015 (eff31-ps 0.905 gold -> 0.926 LB vs their 0.89 -> 0.940). Suspect something test-specific:
  - Do hidden-test scans miss planes or sequences? (topic 744519, unanswered.) Train-time slot dropout may help.
  - Does our test prep differ from train in a way gold OOF cannot show?
  - Worth adding random whole-slot dropout during training: zero out a random present slot with p ~0.2, so the model is robust to missing series.

## 2026-10-03 21:05 UTC session
- Checked: the train-cache prep code (kernels/knee-prep31a) and the test prep code (knee-infer-eff31 and current knee.py at PREP=1) are identical, apart from one sort tie-break (multi-echo only). The LB gap is **not** a prep bug.
- `SLOT_DROP=0.15` is in knee.py, but no trained run has used it yet (eff31-mil, rad31-mil and eff31-rub predate it).
- Decision (user): calibrate on LB first. When eff31-rub finishes, submit an infer notebook of eff31-mil + rad31-mil (+ eff31-rub if it is not worse on gold) **alone**. The result decides how the GPU gets spent:
  - LB ≥ ~0.94: the pipeline is sound. Run rad31-rub, slot-dropout runs, and a diverse fleet, then blend.
  - LB ~0.926: the recipe is the gap. Retrain a ~0.95 public training notebook (Speedy Raptors CoAtNet / DINOv2 dinosaur-v5) on our labels.
- /goal (money): main-LB top 10 **and** an Efficiency Prize top 3. Efficiency score = AUC/(0.5-maxAUC) + runtime_s/32400, so 0.01 AUC is worth ~12 min. The efficiency LB (host notebook ryanholbrook/rsna-knee-abnormalities-efficiency-lb, output full_leaderboard.csv, stale since ~Oct 1) has a top 3 at **0.958 / 0.957 / 0.952 AUC**. So both tracks need a ~0.955 model.
- **knee-infer-mil v1 (eff31-mil + rad31-mil alone, 10 folds) = 0.936 LB** (sub 56807017). Old head was 0.926 (eff31-ps) / 0.924 (rad). Real progress, but under the 0.94 bar and below the 0.943 public stack, so the recipe gap remains.
- Raptor CoAtNet recipe (ref/nvap2011__coatnet-train) vs ours: coatnet_rmlp_2 at 384px, 16 epochs, backbone LR 3e-5 / head LR 1e-3, BCE pos_weight=(1-prev)/prev clipped to [1,10], labels 0.5 pilkwang report_labels_v2 + 0.5 steven v4_blend, trained on all ~4350 non-gold studies with gold-58 as validation.
- New knee.py knobs: `HEAD_LR`, `POS_W`, and `SLOT_DROP` (build options headlr=, posw=, slotdrop=), all recorded in config.json.
- **Running:** knee-train-eff31-rcp = eff31-mil + headlr 1e-3 + posw 1 + slotdrop 0.15, folds 0 and 1 only (~0.7 GPU-h). Compare it on the same folds: `python oof_report.py runs/knee-train-eff31-mil runs/knee-train-eff31-rcp` (it intersects the studies automatically).
- eff31-rcp (Raptor knobs: head LR, pos_weight, slot dropout) gave **no gain**: 0.887 vs 0.889 OOF on the same folds.
- Submitted knee-blend-mil (public 0.6 + MIL pair 0.4) at ~22:50 UTC: blend calibration, score pending.
- knee.py now supports DINOv2 (`backbone=vit_small_patch14_dinov2.lvd142m`, img_size resampled to IMG, backbone LR 3e-5; use headlr=1e-3). **Running:** knee-train-dino31 (folds 0–1) and eff31-rub.
- Next: when rub finishes, compare it with eff31-mil on gold. If DINOv2 beats eff31-mil OOF on folds 0–1, it becomes the next full 5-fold member (GPU heavy, likely after the Oct 10 reset).
- **Session paused 2026-10-03 ~23:00 UTC by the user.** On resume:
  1. Check eff31-rub and dino31 status.
  2. Download both (`--file-pattern "(\.log|oof\.csv|config\.json)$"`), then run `python oof_report.py runs/knee-train-eff31-mil runs/knee-train-eff31-rub runs/knee-train-dino31`.
  3. Check the LB score of knee-blend-mil.
  4. Log everything in EXPERIMENTS.md.
- eff31-rub done: gold 0.913 = eff31-mil 0.913, so rubric labels give no gain; **don't train rad31-rub**. Still waiting: dino31 and the knee-blend-mil LB score.

## 2026-10-04 07:20 UTC session
- knee-blend-mil = **0.944** (public 0.943 + 0.001, noise). DINOv2-S at backbone LR 3e-5 underfit (0.847 vs 0.889 OOF on folds 0–1), so dropped for now.
- GPU: 22.9h of 30h used, 7.1h until the 2026-10-10 reset.
- Next lever: input resolution. We train at 224 from a 256 cache; public strong models use 384 caches, and the forum says gains stop *above* 288. Our weak gold labels are small structures (lateral meniscus 0.855, fracture 0.887).
- New knobs: `build.py prep size=320` (cache px) and `train img=288` (network px). config.json records SIZE; run_infer preps the test set once per (PREP, SIZE) and feeds each model its own IMG, so 224 and 288 models can be mixed.
- **Running (CPU, free):** knee-prep320a–d (shards i:4). Next: 2-fold A/B `python build.py train name=knee-train-eff320 backbone=efficientnet_b0.ra_in1k epochs=6 folds=0,1 img=288 pseudo=0.5:knee-train-rad cache=knee-prep320a,knee-prep320b,knee-prep320c,knee-prep320d`, then compare with `python oof_report.py runs/knee-train-eff31-mil runs/knee-train-eff320`.
- **Resolution A/B: eff320 (320px cache, img 288) beats eff31-mil 0.895 vs 0.889 OOF on folds 0–1** (gold 0.913 vs 0.905, 11/12 labels up). It is the biggest gain since the MIL head.
- Pushed knee-train-eff320b (folds 2–4). Infer attaches eff320 + eff320b together for 5 folds. GPU: 24.4h used; ~2.6h left after eff320b until the Oct 10 reset.
- Next week: rad31-mil at 288 (RadImageNet ~2.3h/fold at 288, so ~7 GPU-h for 5 folds); try img=320/384 on EffNet; use eff320 OOF as a pseudo teacher.
- Efficiency maths: 1 min of runtime = 0.00086 AUC (32400 s per unit vs 0.46 AUC per unit). Efficiency leaders are at 0.952–0.958 AUC, probably with 5–10 min runtimes. At ~0.945 we would need to be ~15 min faster than them, which is impossible. **Efficiency engineering only pays once our AUC is within ~0.005 of the leaders**, so keep putting the GPU into AUC.
- **Paused 2026-10-04 ~13:05 UTC by the user.** knee-train-eff320b (folds 2–4) was still running on Kaggle, with an ETA of ~16:00–16:30 UTC. On resume:
  1. Check `python -m kaggle kernels status ayushkasare/knee-train-eff320b`.
  2. When it is COMPLETE, download its log/oof/config to runs/knee-train-eff320b, then run `python oof_report.py runs/knee-train-eff31-mil runs/knee-train-eff320b` (and the same with eff320). The two eff320 kernels share no studies, so compare each one against eff31-mil separately.
  3. `python build.py infer name=knee-infer-288 trains=knee-train-eff320,knee-train-eff320b,knee-train-rad31-mil`, push, check the log, then submit with the hypothesis: the 288px EffNet lifts the own ensemble above 0.936.
  4. Log everything in EXPERIMENTS.md.

## 2026-10-04 16:50 UTC session
- eff320b done: full 5-fold eff320 (288px) = **0.896 OOF / 0.915 gold**, our best single model. knee-infer-288 (eff320 + eff31-mil + rad31-mil) = **LB 0.938**, the best own-only score.
- rad320 (RadImageNet at 288, 6 ep) is **worse** (0.876 vs 0.882 OOF), so keep RadImageNet at 224.
- GPU: 28.8h of 30h used; **1.2h left until the 2026-10-10 reset.**
- Leaderboard reality (5126 teams):
  - 0.959 = rank 10 (prize line); 0.950 ≈ rank 96; 0.946 ≈ rank 185;
  - silver (top 5%) ≈ rank 256 ≈ 0.945;
  - **our 0.944 ≈ rank 296**;
  - 0.943 is a huge tie block of public forks (ranks ~500–1000).
- Our experiments add ~+0.002 LB each. A top-10 prize needs +0.015, so it is out of reach on this trajectory; silver is the realistic target.
- Pushed knee-blend-288 (public 0.6 + the 0.938 ensemble 0.4), to submit once it finishes. Hypothesis: ≥ 0.945.
- Submitted knee-blend-288 at ~17:00 UTC (pending; blend scores take hours).
- **GPU plan from the 2026-10-10 reset (30h, then another 30h from Oct 17).** Goal: a more diverse own ensemble at 288px, to strengthen the public-stack blend.
  1. ImageNet ResNet-50 (resnet50.a1_in1k, now lr 3e-4) at img=288 on the 320 cache, folds 0–1 A/B, ~3h. A different family from EffNet, and it was never rerun after the LR fix.
  2. EffNet-B0 img=320 on the same cache, folds 0–1, ~2.5h, to see whether resolution keeps paying past 288.
  3. eff320 with labels=rubric, 5 folds, ~6.5h: diversity (eff31-rub had Spearman 0.941 vs eff31-mil and rank-avg +0.002 OOF).
  4. Complete whichever of 1–2 wins to 5 folds. Then build the final blend: public stack + own ensemble, with the weight picked from the LB calibrations (≤2 subs).
- Finals: an accuracy pick (public + own blend) and a second, robust pick (own-only ensemble or a different blend weight). The Efficiency Prize is not reachable at our AUC (see the efficiency maths above).
- Greedy OOF selection: own ensemble saturates ~0.898 OOF (≈ LB 0.938). Don't resubmit own-only reweightings; see EXPERIMENTS.md.
- **Research report 2026-10-04:** `reports/Knee MRI winning techniques.md` (notes in `research_notes/Knee MRI winning techniques/`). Verdict: no +0.015 technique exists; the big RSNA wins needed masks or keypoints we don't have. Revised plan order:
  1. **Team merge before Oct 15** with a partner whose models are *not* the public stack (biggest lever, 0 GPU).
  2. **ConvNeXt-T member** at lr 1e-4 (replaces the ResNet-50 A/B); blend at 0.10–0.15.
  3. EffNet img=320 screen.
  4. Cheap A/Bs: rotation ±25° (ours is ±10°) and a slice-level auxiliary loss.
  5. Full-data refits once the recipe is fixed.
  6. Fail-safe fallback for any own-only final.
  - Skip: OrthoFoundation (pretrained on OAI/fastMRI, so eligibility risk), DINOv3 (licence plus no evidence), MRI-CORE / MedImageInsight / VLMs (too heavy).
  - Realistic landing: 0.947–0.952.
- 2026-10-04 19:40: new knobs `rot=` (ROT_DEG, default 10), `aux=` (AUX_W: max-over-tokens MIL aux loss), and ConvNeXt LR 1e-4; test_knee.py has test_aux_loss. Pushed knee-train-cnx-smoke for ConvNeXt-T timing. The knee-blend-288 score is still pending.
- ConvNeXt-T at 288 needs bs 2 (OOM at 4), ~2.1 h/fold. **First jobs after the Oct 10 reset**, two GPU sessions in parallel:
  - (a) `python build.py train name=knee-train-cnx288 backbone=convnext_tiny.fb_in22k_ft_in1k epochs=6 folds=0,1 img=288 slotdrop=0 pseudo=0.5:knee-train-rad cache=knee-prep320a,knee-prep320b,knee-prep320c,knee-prep320d`
  - (b) EffNet img=320 screen: `... name=knee-train-eff320i backbone=efficientnet_b0.ra_in1k epochs=6 folds=0,1 img=320 slotdrop=0 pseudo=0.5:knee-train-rad cache=<320 cache>`
  - Compare each against eff320 folds 0–1 (runs/knee-train-eff320). Then rot=25 and aux=0.3 arms at 224 on the 31 cache vs eff31-mil (~0.7 h each).

## 2026-10-16 plan to the finish (deadline 2026-10-22 23:59 UTC) — supersedes the plans above
> Dates in this plan were written in local time (IST). The UTC date at writing was **10-15**, so "10-16" below means the first full day. Research behind the 10-15 revisions: `reports/2026-10-15 leaderboard research.md`.
**Where we stand.** Best is 0.944, rank 2085 of 6044. No knee kernels have run since 10-05. The public frontier is now **0.957**: a per-finding rank blend of nartaa's Raptor CoAtNet 384 + 224 (mirror view) and heliosli's compact ConvNeXt-small R6 (flip TTA) / R7, plus small add-ons at 0.1–0.2 weight (knee-centred ConvNeXt, a 2.5D reader). leoprovorov's "0.957 explained" notebook estimates the public models' errors correlate at ρ≈0.86–0.89, so adding more of the same family tops out at ~0.960.
- LB on 10-16: 0.967 is #1 · 0.965 rank 7 · 0.963 rank 15 · 0.960 rank 49 · 0.958 rank 94 · **0.957 = ranks 143–418 (276 teams)** · **0.956 = ranks 419–1326 (908 teams)**.
- Medal lines for 6044 teams: gold ≈ top 22 (~0.963); silver = top 302, inside the 0.957 block; bronze = top 604, inside the 0.956 block.
- Public-half noise SD ≈ 0.003, more than the whole spread of the top 400. The private LB will reshuffle the blocks.

**Verdict.** #1 (0.967) is out of reach in 6 days: it is +0.010 over the best public recipe, and our own pipeline is 0.019 below that recipe. **Target: silver** (top ~300 on private); stretch: 0.959–0.960. The ~1200 teams running the same public blend will all get the same private score. The only way past them is a member whose errors the public family does not share. Ours qualify: own architectures (EffNet MIL, RadImageNet), own prep, v4b + pseudo labels, and our rubric labels.

**Steps**
1. **10-16, secure 0.957.**
   - Pull the current best public notebook into `baseline/` as the new blend base. Candidates: goodpjw2008 `rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954` v21 (R6 + R7 + 224 mirror + reader 1/10), or heliosli `this-is-new-public-dataset-version-0-9576`.
   - Run it unchanged and submit. Expected 0.957, i.e. rank ~150–420.
   - Its log gives the base runtime, which is the headroom left for our add-on under the 9 h limit.
   - Check the GPU quota too: anything left before the 10-17 00:00 UTC reset is lost.
1b. **10-16, labels A/B: top priority** (added 10-15 after the research).
   - Strong solo teams get 0.953–0.959 from one ConvNeXt-T / ResNet-50 / EffNetV2-S without external data. We get 0.926–0.938 from comparable backbones. Every one of them names labels as the lever, and hengck23's numbers (0.89 gold label → ~0.930 LB) match ours exactly.
   - Build a **consensus label set** on CPU from what is in `labels/`: steven v4b, pilkwang, our rubric, qwen3-8b, sol56, hpo v1 blend, calibrated soft targets (which ship per-cell weights in `train_targets_weights.csv`), and the 4-source merge.
     - Target = per-cell mean over the sources.
     - **Confidence mask**: zero loss weight where the sources disagree or the mean is mid-range. One team dropped ~25% of cells this way and gained on both CV and LB.
     - v4's 0.25 marks "not addressed": treat it like SILENT_FIX, not as a positive.
   - Check the label set against gold-58 first (CPU, free; noisy, so look for big moves only).
   - Code: `LABEL_SRC='consensus'` plus a per-cell weight mask in the train loss. Masked cells already happen for missing labels, so reuse that path if there is one.
   - A/B: eff31 at 224 px on the 31 cache, folds 0–1, `pseudo=0.5:knee-train-rad`, against eff31-mil (~0.7 GPU-h). Judge on gold and on OOF vs v4b (biased against the new labels, as with rubric).
   - If it wins: retrain eff320 on it (5 folds, ~6.5 h) and run 3a on it. A ~0.95 own model is worth blend weight 0.3–0.5, where the current 0.938 is only worth ~0.15.
2. **10-16/17, own add-on, no training: run ours FIRST, blend LAST.**
   - Why: the base fills most of the 9 h clock (Raptor 384 ≤2 h, 224 ≤2.5 h, R6 deadline 8.3 h, R7 only if started before 5 h, reader skipped after 8.2 h). Appended last, our part would hit knee.py's 8.2 h guard on the hidden test and drop out without an error, while the 3 visible studies still look fine.
   - Layout: cell 0 = our run_infer writing our ranks to `/kaggle/working/ours.csv` (not submission.csv); then the base, unchanged; then a final cell that rank-blends ours.csv into the base's submission.csv. If ours.csv is missing or broken, the base file stays.
   - The base's clock must count from cell 0. Patch its single clock start (`NB_T0 = time.time()` in cell 4 of v21) to keep an existing value (`globals().get('NB_T0', time.time())`), so its own guards still fit 9 h. This is the only base edit.
   - Cell 0 and the final blend cell must **never raise**: wrap each whole cell in try/except and log the error. Hidden reruns fail on any unhandled exception, even when the visible run passed (forum 749623), and the hidden test "can be larger/smaller/different".
   - **Members: eff320 alone** (5 folds, 0.896 OOF vs 0.897 for the 3-model mix). That is one test prep (PREP 1, 320 px) instead of two, ~20 min instead of ~40 on ~1,300 hidden studies (estimate: ~0.8 s/study per prep). `trains=knee-train-eff320,knee-train-eff320b`.
   - Weight 0.15 (`blend=0.85`) unless optional step 2b says otherwise. Submit once. Keep it if LB ≥ base − 0.001; the LB cannot resolve anything finer.
   - 2b (optional): **gold-58 weight check.** Run the base's models on the 58 gold train studies (~15 GPU-min, with test paths pointed at them). R6/R7 never trained on gold; Raptor used gold for selection, so its numbers are optimistic. Compare per-finding prediction correlation and AUC with our OOF on the same 58, and pick the add-on weight from that rather than 0.15. Do it only if pointing their scripts at train studies is a small patch.
3. **10-17 → 10-20, new members** (≈30 GPU-h after the reset; start right after it, since this is the long pole). Two GPU sessions at a time:
   - **(a) ConvNeXt-T, main job.** ConvNeXt-T at 288 on the 320 cache, a different architecture. **Use the 1b consensus labels if 1b wins**, else `labels=rubric` (different label noise). #4 on the LB reports ConvNeXt-T 5-fold at 0.959. 5 folds ≈ 10.5 GPU-h, run as two kernels:
     `python build.py train name=knee-train-cnx288r backbone=convnext_tiny.fb_in22k_ft_in1k epochs=6 folds=0,1 img=288 slotdrop=0 labels=rubric pseudo=0.5:knee-train-rad cache=knee-prep320a,knee-prep320b,knee-prep320c,knee-prep320d`
     plus a `knee-train-cnx288r-b` twin with `folds=2,3,4`.
   - **(b) Zoomed "knee-centred" cache, cheap screen.** CPU prep (free) at CROP_MM ≈ 110 instead of 150, 320 px, PREP 1, 31 slices, 4 shards: ~1.4× more pixels per mm on menisci, ligaments and cartilage. Resolution was our biggest recent lever (+0.006 OOF from 256 → 320 px), and bliverrigpdoge's knee-centred add-on took the public blend 0.956 → 0.957 at weight 0.2. Screen: eff320 setup, folds 0–1 (~1.5 GPU-h), compared per finding against runs/knee-train-eff320. If it wins on the joint-line findings, run folds 2–4 (~4.5 h).
     - Code needed: a `cropmm=` build option; `CROP_MM` in config.json (missing ⇒ 150); run_infer grouping test preps by (PREP, SIZE, CROP_MM).
     - Risk: the tighter crop can cut suprapatellar effusion and Baker's cysts, so use this model **only for the joint-line findings** (ACL, MCL, both menisci, medial/lateral OA). That needs per-finding member weights in the blend.
     - Check before prep: where the knee sits in the field of view; the crop is centred on the image.
   - Keep a member if its rank average with our ensemble raises OOF (`oof_report.py`).
   - If GPU is left over: eff320 with `labels=rubric`, 5 folds, ~6.5 h.
   - Not doing: a Raptor CoAtNet-384 retrain (~3 h on a 4090, so 12 h+ on a T4, plus its own data pipeline).
   - Maybe: **retrain heliosli's S6 on our labels.** nartaa did exactly this and got 0.950 single-model (fold-0 OOF 0.907; forum 749359). But the `compact_train.py` they cite is not in `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` under that name, and they trained in bf16, i.e. not on a T4. Only if the code turns up and a smoke run works on T4 by 10-17.
4. **10-21, final blend.**
   - Same ours-first layout as step 2. Public base + own members (eff320, plus whichever of 3a/3b pass) at a fixed weight of 0.15–0.2 (or the 2b weight), chosen before seeing the LB. Keep our test preps to ≤2 groups so our part stays under ~45 min.
   - Submit on 10-21 so the score comes back before the deadline. A blend run takes up to ~8.5 h.
5. **10-22, pick 2 finals.**
   - A = public base + own members: the diversity bet.
   - B = public base unchanged: the floor, landing wherever the 0.957 block lands.

**Not doing:** tuning blend weights on the public LB (noise 0.003), own-only submissions, the efficiency prize, DINOv2, prep v2, Raptor knobs, extra pseudo rounds.

**Check (updated 10-15 from the forum and the Kaggle API):**
- **Team merger and new-entrant deadline: 2026-10-15 23:59 UTC** (API `mergerDeadline`). After that we are solo for good. **Decision (user, 10-15): stay solo.**
- LLM-API labels (our Claude rubric labels) are explicitly allowed (host, forum 733965).
- OAI is unresolved. The host said on 9-21 it is fine if free with no institutional review; the geo-restriction question (749145) is unanswered, and #1 Tucker Arrants warned it may not be allowed. The base's Raptor checkpoints were trained partly on OAI. Low-to-moderate risk for medals, real risk for prizes.

### 10-16 step 1 progress
- heliosli's 0.9576 notebook **cannot run unchanged any more**: its `heliosli/rsna-knee-r6-frozen-fe055406` dataset was overwritten on 10-14 18:12 with R7's files (manifest/stacker sizes identical to `rsna-knee-compact-r7cnxfsss`), and the notebook asserts R6's bundle id `fe055406fad6a12f`.
- Base used instead: goodpjw2008 `rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954` **v21** (16 Oct). It is heliosli's recipe (Raptor 384 + 224 mirror, R6 flip TTA via rostigor's pinned copy `rostigor/rsna-knee-r6-fe055406`), plus R7 at 0.4 of the compact slot and goodpjw's 2.5D reader at 1/10. The pull lives in `baseline/goodpjw-v21/`.
- build.py has a new `base` mode (the public notebook unchanged, as our private kernel; drops the pulled `id_no`), and `blend` takes `base=<subdir>`. Step 2 needs no extra datasets: inference builds models with `pretrained=False`, so the RadImageNet weights the jiweiliu base attached are not read.
- Pushed `ayushkasare/knee-base957` v1 (unchanged v21) on 10-16 and verified it cell for cell against the pull. Not submitted yet.
- knee-base957 v1 **COMPLETE** (10-15 ~19:10 UTC; outputs in `runs/knee-base957/`). Every stage passed on the 3 visible studies: Raptor 384 (48 s), 224 (25 s), R6 (345 s), R7 (195 s, share 0.4), and goodpjw's reader (weight 0.15, mean rank correlation 0.781 with the stack). It wrote a 3×12 submission.csv in ~11.5 min. Most of the R6/R7 time on 3 studies is startup, so the hidden-test runtime can't be extrapolated from it.

### 10-15 step 1b progress (~19:35 UTC)
- knee-base957 v1 **submitted** (sub 57242025, 19:21 UTC), score pending.
- Consensus labels built: `consensus.py` → `labels/knee-rubric-labels/consensus_labels.csv`, uploaded as a new version of `ayushkasare/knee-rubric-labels`. Target = mean of v4b(+SILENT_FIX), rubric, pilkwang v2, gpt56sol; cells with mean in (0.3, 0.7) are NaN = no loss (10.3% of cells; Effusion 31%, Synovitis 1%). Gold check in EXPERIMENTS.md.
- knee.py: `LABEL_SRC='consensus'`; `masked_bce` (mean over non-NaN targets, identical to plain BCE without NaN; `test_masked_bce`); POS_W prevalence uses nanmean. build.py mounts the label dataset for `labels=consensus`.
- Windows quirk: `kaggle datasets version -p labels/knee-rubric-labels` fails (upload cache path); run it from inside the folder with `-p .`.
- **Running:** knee-train-eff31-cons (folds 0–1). When done: download, then `python oof_report.py runs/knee-train-eff31-mil runs/knee-train-eff31-cons` (it intersects studies). Judge on gold first and on OOF vs v4b second, which is biased against new labels. If it wins: eff320 + ConvNeXt-T on consensus (steps 1b/3a).

### On resume (user paused 2026-10-15 ~19:45 UTC; decision: stay solo)
1. `python -m kaggle competitions submissions -c rsna-knee-abnormality-detection` → log the knee-base957 score (sub 57242025) in EXPERIMENTS.md.
2. `python -m kaggle kernels status ayushkasare/knee-train-eff31-cons`; when COMPLETE: `python -m kaggle kernels output ayushkasare/knee-train-eff31-cons -p runs/knee-train-eff31-cons --file-pattern "(\.log|oof\.csv|config\.json)$"`, then `python oof_report.py runs/knee-train-eff31-mil runs/knee-train-eff31-cons`. Log it; take the step 1b fork (consensus wins → eff320 + ConvNeXt-T on consensus; loses → ConvNeXt-T rubric).
3. Build step 2 (ours-first blend), push, check the visible-test run, then submit once.
4. Step 3b zoom prep code + CPU prep kernels (free) when there is a gap.
5. GPU quota resets 2026-10-17 00:00 UTC.

## 2026-10-16 05:00 UTC session (resume)
- **knee-base957 = LB 0.957** (sub 57242025), as predicted. It is the blend base and final pick B.
- **eff31-cons (consensus labels) = no gain**: 0.886 vs 0.889 OOF, gold 0.906 vs 0.905 on folds 0–1; rank-avg 0.891 / 0.908, Spearman 0.942. Step 1b fork → ConvNeXt-T on **rubric**.
- **Running (pushed 05:05 UTC on this week's quota, which is lost at the 10-17 reset anyway):** knee-train-cnx288r (folds 0–1, ETA ~07:30) and knee-train-cnx288r-b (folds 2–4, ETA ~09:45). Kaggle MCP was not connected, so the exact quota is unchecked; the dataverse-* kernels of 10-10 also drew on it.
- **Step 2 built:** `build.py blend` now does ours-first: cell 0 `%%writefile`s knee.py (MODE infer, BLEND 0), cell 1 runs it as a subprocess (timeout `ours_min`=60 min) and renames its submission.csv to ours.csv; then the base with `NB_T0 = globals().get('NB_T0', time.time())` (asserted exactly once); a last cell rank-blends ours.csv at base weight `blend=` and keeps the base file on any problem (tested locally: happy path, missing study, missing file, NaN). The old append-last jiweiliu layout is gone.
  - Built `kernels/knee-blend957-eff` (eff320 + eff320b, base 0.85). **Push it as soon as cnx288r frees a GPU slot**, check the log for KNEE_OURS_DONE / KNEE_MIX_DONE and the 3×12 output, then submit once (hypothesis: ≥ 0.957, keep if ≥ base − 0.001).
- **Step 3b zoom cache started (CPU, free):** knee-prep320za–zd pushed 05:10 UTC = `python build.py prep name=knee-prep320z<a..d> nslice=31 size=320 cropmm=110 prep=1 shard=<0..3>:4` (~15–20 min each). New code: `cropmm=` build knob; train reads `CROP_MM` from prep_done.json (shards must agree) and writes it to config.json; run_infer groups test preps by (PREP, SIZE, CROP_MM), and old configs default to 150 mm, so existing models are unaffected. test_knee.py ok.
- **Kaggle MCP added** for this folder (local scope, token from `~/.kaggle/access_token`; `claude mcp list` → Connected). It loads only in sessions started after 10-16 05:15 UTC. Use `get_accelerator_quota` for the real GPU quota.

### On resume (session ended 2026-10-16 05:22 UTC; all 6 kernels RUNNING, nothing submitted since base957)
1. **Status:** `for k in knee-train-cnx288r knee-train-cnx288r-b knee-prep320za knee-prep320zb knee-prep320zc knee-prep320zd; do python -m kaggle kernels status ayushkasare/$k; done`. Check the GPU quota with the Kaggle MCP. Use `bash wait_kernels.sh <kernels>` (background) to wait.
2. **Zoom cache centring check.** knee-prep320zd was already COMPLETE at 05:23 UTC (the other shards were still running), so it can use zd's shard-3 studies `1.2.826.0.1.3680043.8.498.10009639203170750274174707434356622764` and `1.2.826.0.1.3680043.8.498.10021179701366615563817491713738306606` with `ayushkasare/knee-prep320zd` instead of the za ids below. Original note, once knee-prep320za is COMPLETE (before spending GPU on it): download 2–3 studies, e.g. `python -m kaggle kernels output ayushkasare/knee-prep320za -p <scratch dir> --file-pattern "1\.2\.826\.0\.1\.3680043\.8\.498\.1000487322"` (first shard-0 studies: `...10004873229099053869093324292195817260`, `...10013663742400736029415768216599146542`), load with `np.load(allow_pickle=False)`, and render the middle slice of each slot. The femoral condyles, tibial plateau and both menisci must sit inside the 110 mm frame. If the knee is cut off, drop 3b. If it's fine, the GPU screen is `python build.py train name=knee-train-eff320z backbone=efficientnet_b0.ra_in1k epochs=6 folds=0,1 img=288 slotdrop=0 pseudo=0.5:knee-train-rad cache=knee-prep320za,knee-prep320zb,knee-prep320zc,knee-prep320zd`, compared per finding with `python oof_report.py runs/knee-train-eff320 runs/knee-train-eff320z`.
3. **REVISED 10-16 from the public-code research (`reports/2026-10-16 public code insights.md` §2a–2b). Do these before pushing knee-blend957-eff:**
   - (a) **New base = heliosli's 0.958 recipe.** It is R6 flip-TTA + Raptor 384/224 with per-finding Raptor weights, and it drops **R7**, which is "fastMRI auxiliary training": a host-rule risk (institution-gated data), and our v21 base carries it at 0.4.
     - `python -m kaggle kernels pull heliosli/blend-mine-in-private-lb-luck-to-all-upvoters -p baseline/helios958 -m`, then `python build.py base name=knee-base958 base=helios958`.
     - Check that it has exactly one `NB_T0 = time.time()` (`build.py blend` asserts this).
     - Push, check the visible run, submit once. It becomes pick B if it scores ≥ 0.957.
   - (b) **Our add-on weight 0.05, not 0.15.** goodpjw measured their 0.929 reader: full weight cut 0.956 → 0.954, ¼ weight no gain, so they keep it at 1/10. `python build.py blend name=knee-blend958-eff blend=0.95 base=helios958 trains=knee-train-eff320,knee-train-eff320b`. Judge it by rank inside the block, not by the displayed score.
   - Original step 3, now superseded: `python -m kaggle kernels push -p kernels/knee-blend957-eff` (already built; rebuild with `python build.py blend name=knee-blend957-eff blend=0.85 base=goodpjw-v21 trains=knee-train-eff320,knee-train-eff320b` if knee.py changed). When it is COMPLETE, the log must show `KNEE_OURS_DONE` and `KNEE_MIX_DONE` and a 3×12 submission.csv. Then submit once: `python -m kaggle competitions submit -c rsna-knee-abnormality-detection -k ayushkasare/knee-blend957-eff -v 1 -f submission.csv -m "base957 0.85 + eff320 5f 0.15, ours-first layout. Hypothesis: >= 0.957 (keep if >= 0.956)"`.
4. **The GPU slot order after that (revised 10-16 from the forum, see `reports/2026-10-16 forum insights.md`):** two 2-fold screens in parallel, both eff31 at 224 on the 31 cache, folds 0–1, against eff31-mil:
   - (a) stronger, heavier teacher, no code change: `python build.py train name=knee-train-eff31-pt backbone=efficientnet_b0.ra_in1k epochs=6 folds=0,1 slotdrop=0 pseudo=0.7:knee-train-eff320,knee-train-eff320b,knee-train-eff31-mil,knee-train-rad31-mil cache=knee-prep31a,knee-prep31b,knee-prep31c` (~0.7 GPU-h);
   - (b) the same plus `epochs=12 rot=25` as `knee-train-eff31-ptl` (~1.4 GPU-h).
   Judge on gold as well as OOF, since heavier pseudo-labels move the targets away from v4b. The winning recipe becomes the setup for the remaining members. The eff320z screen comes after these, and only if the centring check passes. The forum puts image-centred 110 mm crops at noise level (+0.0005).
   - **Read the knee-blend957-eff result by rank, not score.** We were #477 in the 0.957 block (182–584) at 10-16 05:43 UTC. A jump inside the block after the blend scores means our members add signal; no jump means they don't.
5. **When cnx288r and cnx288r-b are COMPLETE:** download both (`--file-pattern "(\.log|oof\.csv|config\.json)$"`), then `python oof_report.py runs/knee-train-eff320 runs/knee-train-cnx288r runs/knee-train-cnx288r-b` (compare each cnx run with eff320 on its own folds; the two cnx runs share no studies). Keep ConvNeXt as a blend member if its rank-average with eff320 raises OOF. Log in EXPERIMENTS.md.
7. **Public code/model research: DONE 10-16.**
   - Covered: 1,339 notebooks listed, 278 sources read, 39 run logs. Catalog in `research_notes/public_code_catalog.csv`; report in `reports/2026-10-16 public code insights.md`.
   - New candidates for any member trained from here:
     - **mirror augmentation with a medial↔lateral label swap** (heliosli trainer `--mirror`: flip coronal/axial on width, reverse the sagittal slice order, swap Medial/Lateral Meniscus and Medial/Lateral OA, MCL → NaN), then flip TTA at inference;
     - **knee-centred compartment views**: the only comp-data-only add-on measured to help, +0.001 at weight 0.2, twice.
   - Retraining a 384 CoAtNet on Kaggle T4s is not viable (48 min per epoch, holdout got worse).
   - heliosli's R6/R7/S6 trainer `compact_train.py` (3,072 lines) is extracted to `ref/_heliosli_compact_src/`; run logs of 39 key notebooks are in `ref/_logs/`, and all pulled notebooks are in `ref/`.
6. **Forum research: DONE 10-16 ~06:00 UTC.** All 197 topics were read; findings are in `reports/2026-10-16 forum insights.md` and folded into step 4 above. The host ruled on 10-14 that public weights are prize-eligible and that registration-only datasets are allowed. Original brief: read **all** competition discussion topics for insights. Use the Kaggle MCP if it has discussion tools; otherwise use the built-in browser on `https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion?sort=published` (it loads, but `get_page_text` returned empty at first: scroll or wait, or use `read_page`). `reports/2026-10-15 leaderboard research.md` already covers everything up to 10-15 19:20 UTC, so focus on newer topics, high-vote topics, and anything about hidden-test runtime/failures, labels, blending, OAI eligibility, and the final-selection rules. Write the findings to `reports/2026-10-16 forum insights.md`, then fold the actionable points into this plan.

## 2026-10-16 06:25 UTC session: research only (no kernels pushed, nothing submitted)
This session read the whole forum and the whole public Code tab. Nothing was trained, built, pushed or submitted.
- **Forum:** all 197 topics → `reports/2026-10-16 forum insights.md`. Raw dump, one file per topic: `research_notes/forum_2026-10-16/`.
- **Public code:**
  - all 1,339 notebooks → catalog `research_notes/public_code_catalog.csv` (best public score, votes, model families, notes on 29 key ones);
  - findings → `reports/2026-10-16 public code insights.md`;
  - 272 notebooks in `ref/`; 39 run logs in `ref/_logs/`;
  - heliosli's R6/R7/S6 trainer `compact_train.py` + `compact_infer.py` in `ref/_heliosli_compact_src/`;
  - score tables and the helper scripts (forum/kernel fetchers via kagglesdk, catalog builder) in `research_notes/public_code_2026-10-16/`.
- **Decisions this research supports** (detail in the two reports):
  1. **New base = heliosli's 0.958 notebook** (`heliosli/blend-mine-in-private-lb-luck-to-all-upvoters`): R6 flip-TTA + Raptor 384/224 with per-finding Raptor weights (Synovitis 0.8, Lateral OA 0.2, PF OA 0.6, Baker's 0.35, else 0.5).
     - Our current base957 (goodpjw v21) runs **R7 = "fastMRI auxiliary training"**. Under the host's 10-14 rule that institution-gated data "clearly do not" qualify, that is a risk; the 0.958 recipe drops R7.
     - Checked: the notebook's 3 inputs are all public datasets and it has exactly one `NB_T0 = time.time()`, so `build.py base` / `blend base=helios958` work as-is.
  2. **Our add-on at weight 0.05 (`blend=0.95`), not 0.15.** goodpjw's 0.929 reader at full weight cut 0.956 → 0.954, at ¼ weight it did nothing, so they keep it at 1/10. Read any blend by **rank inside the 0.957 block** (we were #477 in 182–584 at 10-16 05:43 UTC), not by the displayed score.
  3. Forum-driven GPU screens: (a) `pseudo=0.7` from the 4-kernel teacher; (b) the same plus `epochs=12 rot=25`. Commands are in step 4 of the section above.
  4. New candidates for members:
     - mirror augmentation + medial↔lateral label swap (MCL → NaN), then flip TTA (~15 lines in `to_input`/`train_worker`/`run_infer`);
     - a knee-centred **compartment-view** member: the only comp-data-only add-on measured at +0.001 (weight 0.2, twice).
  5. Not viable: retraining a 384 CoAtNet on Kaggle T4s (48 min/epoch, holdout got worse).

### On resume (session ended 2026-10-16 06:25 UTC)
State at 06:20 UTC:
- **knee-train-cnx288r** and **knee-train-cnx288r-b**: RUNNING (ETA ~07:30 / ~09:45 UTC).
- **knee-prep320za–zd**: COMPLETE (110 mm zoom cache; centring check not done yet).
- **knee-blend957-eff**: built locally, **never pushed**.
- Last submission: knee-base957 = 0.957 (sub 57242025).
- **GPU quota (Kaggle MCP):** 6.3 h used + 10.6 h reserved by the two cnx runs, out of 30 h. That leaves ~13 h this week, **lost at the 2026-10-17 00:00 UTC reset**, then a fresh 30 h. Only 2 GPU sessions at a time.
- Final submission deadline 2026-10-22 23:59 UTC; 5 submissions/day.

Do in this order:
1. **Base 0.958 (CPU build; GPU at push):**
   - `python -m kaggle kernels pull heliosli/blend-mine-in-private-lb-luck-to-all-upvoters -p baseline/helios958 -m`
   - `python build.py base name=knee-base958 base=helios958`
   - When a GPU slot frees, `python -m kaggle kernels push -p kernels/knee-base958`. Check that the log reaches `NOTEBOOK_COMPLETE` with a 3×12 submission.csv.
   - Submit once: `-m "heliosli 0.958 recipe unchanged (R6 flip-TTA + Raptor 384/224, per-finding weights; no R7/fastMRI). Hypothesis: 0.958; becomes pick B"`.
2. **Our add-on at 0.05:**
   - `python build.py blend name=knee-blend958-eff blend=0.95 base=helios958 trains=knee-train-eff320,knee-train-eff320b`, push, check `KNEE_OURS_DONE` / `KNEE_MIX_DONE`.
   - Submit after the base958 score is in, so the two can be compared by rank.
   - Delete the stale `kernels/knee-blend957-eff` plan (step 3 of the section above).
3. **When cnx288r (folds 0–1) completes:**
   - download `--file-pattern "(\.log|oof\.csv|config\.json)$"` to runs/;
   - `python oof_report.py runs/knee-train-eff320 runs/knee-train-cnx288r`;
   - log it in EXPERIMENTS.md;
   - push screen (a) `knee-train-eff31-pt` in the freed slot (command in step 4 above). Use the second slot for (b) `knee-train-eff31-ptl` or for base958/blend958, whichever is pending.
4. When cnx288r-b completes: same download and report for folds 2–4. Keep ConvNeXt as a member only if its rank-average with eff320 raises OOF.
5. Zoom cache: do the centring check (step 2 above) before any GPU on it. Prefer turning it into a compartment-view member (medial/lateral halves) over a plain centre zoom.
6. Use GPU left before the 10-17 reset for whichever screen wins. After the reset, retrain the best recipe on 5 folds (with mirror aug if implemented).
7. Final picks (10-22): A = base958 + our members at ≤ 0.10 (keep only if they rank above base958); B = base958 alone.

## 2026-10-16 12:30 UTC: ConvNeXt results (steps 3–4 of "On resume" done)
- **knee-train-cnx288r and -b: COMPLETE.** Outputs in `runs/knee-train-cnx288r*/`; merged 5-fold OOF in `runs/cnx288r-5f/oof.csv`. Full row in EXPERIMENTS.md.
  - 5-fold OOF 0.895 vs eff320 0.896 (OOF is scored vs v4b, which is biased against rubric-trained models); gold 0.928 vs 0.915, higher in both fold halves.
  - Rank-avg eff320 + cnx = **0.899 / 0.925**, the best own OOF so far, as good as adding eff31-mil and rad31-mil too. Spearman 0.936.
  - **Keep ConvNeXt as a member.** Our add-on is now `trains=knee-train-eff320,knee-train-eff320b,knee-train-cnx288r,knee-train-cnx288r-b`. Both use the 320 px / 150 mm cache, so there is still one test prep.
- **This changes step 2 of "On resume":** build the add-on as `python build.py blend name=knee-blend958-ec blend=0.95 base=helios958 trains=knee-train-eff320,knee-train-eff320b,knee-train-cnx288r,knee-train-cnx288r-b`. It has 10 fold models instead of 5, and ConvNeXt is heavier, so our part takes longer on the hidden test. The 3-study visible run can't show that, so check the per-group time in the visible log before trusting `ours_min=60`.
- **Code review of today's edits (10-16):** no bugs found; `python test_knee.py` ok. Two things to fix:
  - a zoom (110 mm) member adds a second full test-DICOM pass (~17–20 min on ~1,300 hidden studies), so raise `ours_min` before such a blend;
  - `build.py` silently ignores misspelled options (e.g. `label=` for `labels=`); a one-line assert against the known keys would catch it.
- Nothing pushed or submitted this session. Both GPU slots are free. The GPU quota is unchecked since 06:20; the week's remainder is lost at the 10-17 00:00 UTC reset.
- **12:37 UTC pushed:** knee-base958 v1 (heliosli 0.958 notebook unchanged: `baseline/helios958/`, 3 public inputs, one `NB_T0`; its R6 input is the `-w1` copy without R7, and fastMRI appears only in the embedded trainer source) and knee-train-eff31-pt (screen a). knee-train-eff31-ptl (screen b) is built in `kernels/` and goes into the slot base958 frees. GPU quota at 12:30 UTC: 9.6 h of 30 h used, nothing reserved.
- **12:46 UTC:** knee-base958 v1 **COMPLETE** in ~8 min on the 3 visible studies (`runs/knee-base958/`): Raptor 384 44 s, Raptor 224 23 s, R6 stage PASS (340 s, no fallback studies), `NOTEBOOK_COMPLETE` with per-finding Raptor weights (Lateral OA 0.2, PF OA 0.6, Synovitis 0.8, Baker's 0.35, else 0.5; a224 0.35), 3×13 submission.csv with no NaN. **Not submitted yet.** Submit: `python -m kaggle competitions submit -c rsna-knee-abnormality-detection -k ayushkasare/knee-base958 -v 1 -f submission.csv -m "heliosli 0.958 recipe unchanged (R6 flip-TTA + Raptor 384/224, per-finding weights; no R7/fastMRI). Hypothesis: 0.958; becomes pick B"`. knee-train-eff31-ptl pushed into the freed slot at 12:46 UTC, so both screens are running.
- **12:50 UTC: knee-base958 v1 submitted** (ref 57269563, PENDING; 4 submissions left today). Leaderboard at 12:50 UTC (6,085 teams): top 0.968; gold line (rank 22) 0.963; 0.958 block = ranks 113–446; 0.957 block = 447–791. **We are #707 at 0.957, below the bronze line** (rank 608 is in the 0.957 block). Silver line (rank 304) sits inside the 0.958 block. The top moved a lot since 05:43 UTC, so there may be new public work above 0.958: check notebooks and topics newer than 10-16 06:00 UTC.
- **13:10 UTC, CPU work while the screens train:**
  - **Mirror augmentation is in knee.py:** `MIRROR` knob, build option `mirror=0.5`. Per study with probability p, sagittal slices are reversed and coronal/axial images flipped on width; Medial↔Lateral Meniscus and OA labels swap and MCL becomes NaN. Models with `MIRROR` in config.json get flip TTA in both OOF `predict` and `run_infer` (logits averaged, MCL from the plain pass). test_knee.py `test_mirror` passes, and a CPU emulation of the run_infer loop is fine. Not yet run on a GPU.
  - **build.py now rejects unknown options** (review finding 2).
  - **knee-blend958-ec is built** (`kernels/knee-blend958-ec`; one test-prep group, `ours_min` 60) and waits for a free GPU slot.
  - **Zoom-cache centring check passed** (see EXPERIMENTS.md). The cache is image-centred, so a true compartment member would still need knee-centred crops; lower priority than the recipe and mirror members.
- **Forum check 10-16 13:20 UTC** (topics after 06:00 UTC):
  - **749936, goodpjw ledger 0.944 → 0.958:**
    - with public parts only: R6 + flip TTA, then heliosli's per-finding weights, reached 0.958 at rank 195;
    - no change from R7 as a second compact model or from the 224 checkpoint at 50%;
    - their reader at full weight hurt (0.956 → 0.954), and an S6-distilled reader (rank correlation 0.90 with S6) gave 0.953;
    - **diversity beats strength in the third slot**: the only third-slot gain was bliverrigpdoge's knee-centred crops;
    - ties in the score are ordered by submission time, so an exact public copy lands behind earlier copies; a real small gain shows only as rank movement.
  - **743416, OAI under review:** a 10-16 06:11 comment asks whether OAI (Login.gov rejects China, error LG22; NIH NOT-OD-25-083) is "equally accessible", and whether that also covers weights trained on it. **Unanswered.** Raptor in base958 is OAI-trained. For now the 10-14 ruling stands (registration-only datasets OK).
  - **735304** (10-09): one team reports 5-fold single architectures at LB ConvNeXt-T 0.959, EffNetV2-S 0.956, CoAtNet 0.952. Their own pipeline, not public.
  - **Public notebooks:** none above 0.958. The top three are goodpjw's new 0.958 (R6 + their reader), degnonguidi 0.958, and heliosli's 0.958 (our base958). No base switch.
- **13:40 UTC:** knee-train-eff31-pt COMPLETE: OOF 0.891 vs eff31-mil 0.889 (folds 0–1), gold 0.910 vs 0.905. Small gain (EXPERIMENTS.md). **knee-blend958-ec pushed** into the freed slot (visible run). knee-train-eff31-ptl is still running. Hidden test ≈ 1,322 studies (forum 744230), so our blend part is estimated at ~25–35 min against `ours_min` 60.
  - Built and waiting for a slot: `kernels/knee-train-eff320m` = eff320 recipe + `mirror=0.5`, folds 0–1 (compare with `python oof_report.py runs/knee-train-eff320 runs/knee-train-eff320m`).
  - Next after that: the winning pseudo recipe at 288 on the 320 cache (pt: 6 ep; ptl: 12 ep, rot 25).
- **13:46 UTC:** knee-blend958-ec visible run passed (our 10 fold models 60 s, KNEE_OURS_DONE, KNEE_MIX_DONE, 3×13 with no NaN, same ids and columns as base958) and was **submitted (ref 57271112)**. Submissions today: 2 (cap 4). **knee-train-eff320m (mirror screen) pushed 13:46** into the freed slot. Both submissions are pending.
- **14:30 UTC:** **knee-train-eff31-ptl = winning recipe**: OOF 0.896 vs 0.889 (eff31-mil) and 0.891 (pt); gold 0.916. 11/12 findings up. **knee-train-eff320ptl pushed** (ptl recipe at 288 on the 320 cache, folds 0–1, ~3 h). `kernels/knee-train-eff320ptl-b` (folds 2–4, ~6 h) is built and goes into the slot the mirror screen frees (~15:30). Compare per finding with `python oof_report.py runs/knee-train-eff320 runs/knee-train-eff320ptl`. Stale build, not to push: `kernels/knee-train-eff320pt` (6-epoch pt recipe at 288).
- **15:15 UTC:** knee-train-eff320m (mirror screen) **no gain**: 0.892 vs 0.895 OOF, gold 0.902 vs 0.913; mirror dropped from plans. **knee-train-eff320ptl-b pushed** (folds 2–4, ~6 h, ETA ~21:15). Built for the slot eff320ptl frees (~17:30): `kernels/knee-train-eff320z` = ptl recipe at 288 on the **110 mm zoom cache**, folds 0–1. Compare it with eff320ptl on the same folds. A zoom member in a blend adds a second test-prep group, so raise `ours_min` and check the base's time budget first.
- **17:20 UTC:** knee-train-eff320ptl (folds 0–1) COMPLETE: **OOF 0.899 vs eff320 0.895**, gold 0.920. Folds 0–1 rank-avg with cnx: 0.899 (vs 0.897 for eff320 + cnx), so it qualifies for the blend. Decide on 5 folds when eff320ptl-b lands (~21:15). **knee-train-eff320z (zoom, folds 0–1) pushed 17:17** (~3 h).
  - Next blend candidate: `python build.py blend name=knee-blend958-pc blend=0.95 base=helios958 trains=knee-train-eff320ptl,knee-train-eff320ptl-b,knee-train-cnx288r,knee-train-cnx288r-b` (plus eff320/eff320b only if the 3-way rank-avg beats the 2-way by ≥ 0.001). Submit it only after base958 and blend958-ec are scored and the add-on showed a rank gain.
- **17:30 UTC quota:** 19.0 h used of 30 h (resets 10-17 00:00 UTC). The user wants this week's quota fully used. Two slots: eff320ptl-b (ETA ~21:15) and eff320z (ETA ~20:15). With both slots full until midnight the quota runs out ~23:00. Kaggle saves no output from a session that is stopped, so refills must **finish by ~23:00**:
  - slot freed by eff320z (~20:15): if zoom adds ≥ 0.001 to rank-avg, push `kernels/knee-train-eff320z-b` (zoom folds 2–3, ~2.6 h); else push `kernels/knee-train-eff31-ptl-b` (ptl at 224, folds 2–3, ~1.7 h);
  - slot freed by eff320ptl-b (~21:15): `kernels/knee-train-eff31-ptl-c` (fold 4, ~1.7 h) to complete eff31-ptl as a 5-fold 224 member.
  - Next week's 30 h (10-17 → 10-22): cnx with the ptl recipe (12 ep, ~4 h/fold), zoom folds 4 if it helps, then final blends.
- **18:01 UTC: knee-base958 = 0.958, rank 339/6100.** The 0.958 block covers ranks 124–585, so we are in bronze; the silver line (rank 304) is in the same block. The hidden run took ~5 h 10 min, so our add-on must stay well under the ~3.5 h of headroom (we expect ~25–35 min). Base = base958 (goal step 1). knee-blend958-ec (57271112) is still pending; judge it by its rank against 339 inside the 0.958 block.
- **18:41 UTC: knee-blend958-ec = 0.957**, one block below base958 (0.958). **Add-on rejected** under the keep rule. Base958 alone stays our best (rank 339). Weight 0.90 (goal step 6) is not to be tried.
  - What can still help: a much stronger or more different own member. At most one more add-on test, only if the 5-fold eff320ptl + cnx (+ zoom) rank-avg clearly beats 0.899, e.g. ≥ 0.903. Otherwise final picks = base958 (B) and the best other rule-safe candidate (A).
- **19:55 UTC:** **eff320ptl 5-fold = 0.903 OOF** (vs eff320 0.896); rank-avg **eff320ptl + cnx = 0.902** (vs 0.899). New own pair: `trains=knee-train-eff320ptl,knee-train-eff320ptl-b,knee-train-cnx288r,knee-train-cnx288r-b`. Below the ≥ 0.903 bar for another add-on submission; cnx retrained with the ptl recipe (next week) is the likely way past it. **knee-train-eff31-ptl-b (folds 2–3, 224) pushed 19:55**; quota 23.9 h used at 19:55.
- **20:02 UTC:** **knee-train-eff320z (zoom, folds 0–1) = 0.902 vs eff320ptl 0.899**; folds 0–1 rank-avg with eff320ptl + cnx 0.899 → 0.902, so it joins. **knee-train-eff320z-b (folds 2–3) pushed 20:02** (~2.6 h). Quota plan for the rest of the week: at ~21:40, when eff31-ptl-b ends, push `kernels/knee-train-eff31-ptl-c` (fold 4, ~1.4 h alone), reaching ~29.8 h by ~23:05. **After the 00:00 UTC reset:** build and push zoom fold 4 (`python build.py train name=knee-train-eff320z-c backbone=efficientnet_b0.ra_in1k epochs=12 rot=25 folds=4 img=288 slotdrop=0 pseudo=0.7:knee-train-eff320,knee-train-eff320b,knee-train-eff31-mil,knee-train-rad31-mil cache=knee-prep320za,knee-prep320zb,knee-prep320zc,knee-prep320zd`), then cnx with the ptl recipe.

## (history) ON RESUME (2026-10-16 22:50 UTC), superseded
Active `/goal`: finish with the strongest rule-safe final pair by 2026-10-22 (last submission by 10-22 12:00 UTC, at most 4 per UTC day keeping 1 spare). Ask the user before final selection, deleting, publishing, or any dataset with unclear licence.

**Leaderboard and submissions**
- **knee-base958** (ref 57269563) = **0.958**, team rank **339/6100** (10-16 18:01 UTC). This is the base and pick B.
- **knee-blend958-ec** (ref 57271112: base958 0.95 + eff320 + cnx288r at 0.05) = **0.957**. Add-on **rejected** (lower block). Weight 0.90 not to be tried.
- 10-16 UTC submissions used: 2. Blocks at 18:01: 0.959 = 79–123, **0.958 = 124–585**, 0.957 = 586–904; gold line rank 22 = 0.963, silver 304 = 0.958, bronze 608 = 0.957.
- Base hidden runtime ≈ 5 h 10 min (visible 3 studies: 441 s), so our add-on has ~3.8 h headroom (one prep group ≈ 25–35 min; a second crop group adds ~20–30 min, so `ours_min=90` for two groups). A blend's visible run takes ~8–10 min (blend958-ec: 494 s).

**Own models, OOF vs v4b / gold-58** (all in EXPERIMENTS.md; `python combos.py runs/...` scores every subset)
- Winning recipe "ptl": pseudo 0.7 from 4 teachers (eff320, eff320b, eff31-mil, rad31-mil) + 12 epochs + rot 25.
- eff320ptl 5-fold (288, 150 mm cache) = 0.903 / 0.924. Kernels knee-train-eff320ptl (f0–1) + -ptl-b (f2–4); merged `runs/eff320ptl-5f/`.
- **Zoom eff320z (ptl, 110 mm cache) folds 0–3 = 0.905 / 0.928, the best single model.** Kernels knee-train-eff320z (f0–1) + -z-b (f2–3) + -z-c (f4, running); merged f0–3 `runs/eff320z-4f/`.
- cnx288r 5-fold (ConvNeXt-T, rubric labels, old recipe) = 0.895 / 0.928; `runs/cnx288r-5f/`.
- On the folds 0–3 studies: eff320ptl + cnx 0.9032; **eff320ptl + zoom 0.9060**; all three 0.9055. **Own set = eff320ptl + zoom** (cnx288r drops out under the +0.001 rule; it only helps the noisy gold-58).
- Dropped: eff31-ptl (0.899 alone, +0.0003 in the ensemble, second prep group), mirror aug, consensus labels.

**Running** (wait with `bash wait_kernels.sh knee-train-cnxzptl-c knee-train-cnx288ptl-c`)
- knee-train-eff320z-c: zoom fold 4: **COMPLETE 02:01**.
- knee-blend958-z: visible run passed (two prep passes, 10 models, KNEE_OURS_DONE 44 s, KNEE_MIX_DONE, 3×13); **submitted 02:13 as ref 57291505, PENDING** (wait with `bash wait_sub.sh 57291505`; scoring ~5–6 h, so ~08:00).
- knee-train-cnxzptl: **COMPLETE 06:08**, f0–1 0.904 (best single), joins.
- knee-train-cnxzptl-b: **COMPLETE 10:14**, f2–3 0.911; merged f0–3 `runs/cnxzptl-4f/`.
- knee-train-cnxzptl-c: fold 4, pushed 10:14, RUNNING; ETA ~15:00.
- knee-train-cnx288ptl: **COMPLETE 04:05**, f0–1 0.902, joins.
- knee-train-cnx288ptl-b: **COMPLETE 08:47**, f2–3 0.910; merged f0–3 `runs/cnx288ptl-4f/`.
- knee-base958v2: visible run passed (473 s; -w1, bundle fe055406, own leg blended, 3×13); **submitted 08:58 as ref 57304295, PENDING** (wait with `bash wait_sub.sh 57304295`; scoring ~5–6 h, so ~14:30). 10-17 submissions used: 2.
- knee-train-cnx288ptl-c: fold 4, pushed 08:58; queued 08:58–~11:00, then **RUNNING by 11:07 while base958v2 was still PENDING**. So the queue was most likely Kaggle-side T4 capacity, not our session limit (the "scoring holds a slot" idea is not supported). ETA ~15:50.

**Quota:** 28.2 h of 30 h at 22:45 (one session until the reset → ~29.5 h). Resets 10-17 00:00 UTC to a fresh 30 h that lasts to the deadline. Kaggle saves nothing from a stopped session.

**10-17 07:45 milestone: blend958-z = 0.957 (add-on rejected again), and our rank slipped to 553/6123.** The 0.958 block is now ranks 136–855 (720 teams); we are 418th inside it. Exact copies submitted after ours rank below us, so the ~200 teams that moved above us run other 0.958 notebooks with a higher unrounded score. The bronze line (~rank 612) sits inside the block. 0.959 = ranks 90–135; rank 22 (gold line) = 0.963.
- **Base candidate knee-base958v2 (built 08:00, not yet pushed):** heliosli's "this-is-best-public-dataset-new-version-0-9576" V2 (rerun 10-17 04:21, now 0.958) = base958 + retuned per-finding Raptor weights (Lateral OA 0.15, PF OA 0.65, Synovitis 0.85, Baker's 0.30) + **Part B: goodpjw's 2.5D ConvNeXt reader** (dataset goodpjw2008/rsna-knee-2-5d-convnext-reader, Apache 2.0, trained on report-derived labels from competition data only, 3 folds) at per-finding weights 0.004–0.035, with an 8.2 h guard and fallback. Pulled to `ref/_new_2026-10-17/`, copied to `baseline/helios958v2/` with **one change: C_SLUG and the dataset source use base958's `-w1` R6 copy** (the un-suffixed dataset's latest version holds R7; the bundle_id fe055406 assertion stays). Rule check: no R7/fastMRI (bundle asserted), external data = OAI (nartaa Raptor) + SKM-TEA (anatomy.pt), the same as base958. **Plan:** push `kernels/knee-base958v2` when a GPU slot frees (~08:10, before cnxzptl-c), check the visible log (ASSETS_READY with -w1, bundle fe055406, "own leg: blended 3 checkpoints", NOTEBOOK_COMPLETE, 3×13), then submit with the hypothesis "same family with the V2 reweight + goodpjw reader ranks above base958 (553) inside the 0.958 block". If it ranks above, it replaces base958 as pick B and as the base.
- **lavinwins/rsna-gold-medal: user approved 11:08** ("test it unchanged as a pick-A hedge candidate (one submission); I accept the CC BY-NC-SA 4.0 heads dataset and the undocumented training data for that test"). Built unchanged as `kernels/knee-baselav958` (cells identical; private; internet off; 14 datasets + sofiaanjenje e11/e13 kernel outputs + DINOv2 base/small Kaggle models). **Push it when the first GPU slot frees (~15:00, cnxzptl-c)**, check the visible log (it wants 2 GPUs; DINOv2 20/20, A5, RadImageNet, Raptor, CoAt family, submission.csv 3×13), then submit with a hypothesis. Earlier vetting 07:50 (unclear licence/data): "v6 combination layer on top of the 0.943 public pipeline, no new checkpoints" = 20 DINOv2-S members + A5 + RadImageNet heads + Raptor ×4 + a 4-member CoAt family; **no heliosli R6/R7, no nartaa OAI weights**, so it is a different family from base958 and might hedge the OAI question. But `antoinegg1/rsna-knee-e11-diverse-heads-v20` is **CC BY-NC-SA 4.0**, and the mattiaangeli/pilkwang weight datasets state no training data (one is named "CoAtNet Global96 **Gold** Top3"). Per the goal rules, ask before using it. Its visible run is ~7 min, hidden runtime unknown.
- Other new 0.958 notebooks (pulled): lavinwins/rsna-gold-medal (a different family: dread Raptor, mattiaangeli CoAtNets, DINOv2, RadImageNet heads, pilkwang; no heliosli R6) is a diversity candidate for pick A if rule-safe; kiyoshiohno/rsna-knee-template (mattiaangeli weights + nartaa + orthofoundation).

**10-17 11:40: union plan (user-approved via the side session "btw: evaluate the whole situation"; verbatim: "Yes. I approve using lavinwins inside a blend for a final pick. Send the union plan (base958 + lavinwins rank blend with a time guard and fallback) to the main session to build and submit.")**
- Why (side-session evaluation): prize tracks are out of reach; the real fight is silver (≤ ~306) vs bronze (≤ ~612), both in or near the 0.958 block. Most of the ~720 teams at 0.958 are exact copies, and private ties go to the earlier submission, so our 10-16 copy stays behind earlier copies on private too. A unique, marginally better submission escapes the pack. Own add-ons are 0/2, so stop own-model add-on submissions and spend no new GPU on own members (the two running fold-4 kernels finish).
- **Built `kernels/knee-union958lav`** with the new `python build.py union name=knee-union958lav base=helios958 base2=lavin958` (blend 0.5, union_h 8.6): base958's 9 cells unchanged → save base.csv to /tmp/union and remove base958's /kaggle/temp/probe scratch → lavinwins written byte-identical via %%writefile and run unchanged by `python -m nbconvert --execute` in its own kernel, with timeout 8.6 h − elapsed since NB_T0 (psutil kills the whole tree on timeout) → base.csv always restored → 50/50 per-finding rank blend with id/column/finite checks, else base958 kept. Checks: base cells identical, lavinwins round-trips byte for byte, new cells parse, same docker image, 17 datasets + 2 kernels + 2 models, private, internet off. Only shared file name: submission.csv (lavinwins deletes it at start, line in its cell 10). lavinwins has its own TIME_BUDGET 8 h from its start, which our outer timeout overrides.
- **Review fixes (side-session code review 12:10, user-approved verbatim: "Yes, send findings 1-3 (runtime fallback, timm/pydicom downgrade, lavinwins time budget) to the main session before it pushes the union, with the suggested fixes."), applied 12:30:**
  1. Silent fallback risk: lavinwins gets only ~8.6 h − base958 ~5.2 h ≈ 3.4 h and is all-or-nothing (rsna_deadline raises; submission.csv only at the end). If it overruns, the union submits base958's exact predictions, and the LB cannot show that. **New order: submit knee-baselav958 FIRST and time its hidden scoring (submit → COMPLETE). Submit the union only if 5.2 h + lavinwins + 0.3 h < 8.6 h.**
  2. base958's cell 6 pip-installs timm 1.0.25 / pydicom 3.0.1 over the docker's 1.0.29 / 3.0.2 (its own log), and lavinwins' main kernel imports both. Fix: new union cell 0 snapshots the docker's timm + pydicom (+ dist-info) to /tmp/union/site before the base cells; nbconvert runs with PYTHONPATH=/tmp/union/site (lavinwins' CoAt subprocesses still use their own timm 1.0.22 --target dirs).
  3. lavinwins' TIME_BUDGET = 8.0 * 3600 (cells 10 and 46) counts from its own start. In the embedded copy only, both lines read `float(os.environ.get('UNION_BUDGET_S', 8.0 * 3600))`, and the run cell exports budget − 600 s (unchanged when run alone; the build asserts 2 lines). Its flip-TTA budget guard is already off (arm_flip_tta=False), so this mainly makes it fail cleanly instead of being killed.
  Lower priority, also done: the build asserts exactly one NB_T0 in the base; the mix asserts unique ids and computes rho before os.replace; combos.py skips one-class findings.
- **Push order when GPU slots free (revised):** ~15:00 (first slot): push `kernels/knee-baselav958`; check its log (DINOv2 20/20, A5, Rad, Raptor, CoAt family, 3×13) and **submit it**, noting the submit time. ~15:50 (second slot): push `kernels/knee-union958lav` (14 cells: snapshot + base958 ×9 + save + lavinwins + run + mix). **Gate before submitting the union:** markers UNION_SNAPSHOT ×2, UNION_BASE_SAVED, UNION_LAV_DONE, UNION_MIX_DONE; 3×13, no NaN; union_base.csv == runs/knee-base958/submission.csv (visible) and union_lav.csv ≈ knee-baselav958's visible submission.csv (allclose, or per-finding rank corr > 0.999); and the runtime rule from fix 1, once baselav958 has scored. 10-17 submissions: 2 used, so baselav958 = 3rd; the union may wait until 10-18 if baselav958 scores late.
- Final-pick proposal for the user: A = the union if it scores ≥ base958 (higher block or better rank), B = base958 (or base958v2 if it ranks clearly higher, or lavinwins alone as a family hedge). Disclosure: OAI + SKM-TEA (base958) + lavinwins' sources (CC BY-NC-SA 4.0 heads, undocumented training data).

**Next steps, in order**
1. **Done 00:04 10-17 (quota reset to 0 h; zoom f4 survived the reset):** pushed `kernels/knee-train-cnx288ptl` (ConvNeXt-T, ptl recipe, 150 mm cache, folds 0–1, ~4 h; built, differs from eff320ptl only in BACKBONE).
2. **Progress 02:03:** eff320z-c COMPLETE 02:01; **5-fold zoom 0.905 / 0.928**, eff320ptl + zoom **0.9057** (all 4407 studies; all three 0.9049, so cnx288r is out). knee-blend958-z built (diffed vs blend958-ec: only the knee.py mirror/flip code, ours_min 90 and the 5 trains) and **pushed 02:03** for its visible run. Visible log passed; submitted 02:13 (ref 57291505); cnxzptl pushed 02:13. 10-17 submissions used: 1.
   Original plan for this step, **~01:30, when eff320z-c is COMPLETE:**
   - download it, merge `runs/eff320z-5f/oof.csv` (eff320z + -b + -c), run `python combos.py runs/eff320ptl-5f runs/cnx288r-5f runs/eff320z-5f`, and log it;
   - build the add-on test (own rank-avg ≥ 0.903 is met): `python build.py blend name=knee-blend958-z blend=0.95 base=helios958 ours_min=90 trains=knee-train-eff320ptl,knee-train-eff320ptl-b,knee-train-eff320z,knee-train-eff320z-b,knee-train-eff320z-c`, then push it (visible run ~10 min). This is the first real run of **two prep groups** (150 mm + 110 mm), so check the log for two prep passes, 10 models, KNEE_OURS_DONE, KNEE_MIX_DONE and a 3×13 submission.csv;
   - submit once with a hypothesis message, e.g. "base958 0.95 + own eff320ptl+zoom 0.05 (rank-avg OOF 0.906 vs 0.899 last time): add-on lifts rank inside the 0.958 block". Judge by rank vs base958's 339. Wait with `bash wait_sub.sh <ref>`; scoring takes ~5–6 h.
   - then push `kernels/knee-train-cnxzptl` (ConvNeXt-T, ptl recipe, **zoom cache**, folds 0–1, ~4 h; built, differs from eff320z only in BACKBONE). It is screened against cnx288ptl: zoom beat 150 mm for EfficientNet.
3. **Progress 08:50:** cnx288ptl-b COMPLETE 08:47 (f2–3 0.910). Folds 0–3: cnx288ptl **0.9061** (best single), zoom + cnx288ptl **0.9075**, eff320ptl + zoom + cnx288ptl 0.9073 (+0.0013 over eff320ptl + zoom), so **cnx288ptl joins and needs fold 4**. **knee-base958v2 visible run pushed 08:47** into the freed slot. Next: when base958v2 completes (~09:00), check its log, submit it, then push `kernels/knee-train-cnx288ptl-c`; at ~10:15 (cnxzptl-b COMPLETE) push `kernels/knee-train-cnxzptl-c`. Quota 15.2 h at 08:50; the rest of the plan is ~11.6 h, ending ~26.8 h.
   Earlier, **06:09:** cnxzptl f0–1 = **0.904** (best single). Folds 0–1 rank-avg: zoom + cnxz 0.9044, zoom + cnx288ptl + cnxz 0.9047, eff320ptl + zoom + cnxz 0.9041, all four 0.9044. **Pushed cnxzptl-b (f2–3) at 06:09** (ETA ~10:15). Quota 9.9 h at 06:09.
   **Next:** at ~08:10, when cnx288ptl-b completes, push `kernels/knee-train-cnxzptl-c` (f4). At ~10:15, score folds 0–3 of everything with combos.py: push `kernels/knee-train-cnx288ptl-c` only if cnx288ptl still adds ≥ 0.001 on top of zoom + cnxz (else leave cnx288ptl out). If the final set is zoom + cnxz (both 110 mm), the blend needs **one** prep group.
   Earlier, **04:06:** cnx288ptl f0–1 = **0.902** (cnx288r 0.893); on folds 0–1, eff320ptl + zoom 0.9023 → + cnx288ptl **0.9035 (joins)**; zoom + cnx288ptl 0.9038. **Pushed cnx288ptl-b (f2–3) at 04:06** (ETA ~08:10). Next: at ~06:20 score cnxzptl; push cnx288ptl-c (f4) into the free slot (and the cnxzptl-b/-c pair later if cnxz also joins). Quota 5.8 h used at 04:06; finishing both ConvNeXt variants costs ~16 h, so it fits.
   Finishing kernels are prebuilt (02:30) and diffed (only FOLD_LIST differs):** `kernels/knee-train-cnx288ptl-b` (f2–3), `-c` (f4), `kernels/knee-train-cnxzptl-b` (f2–3), `-c` (f4). Push the winner's two in parallel.
   ~04:10 / ~06:20: score cnx288ptl and cnxzptl folds 0–1 with combos.py (vs eff320ptl + zoom on folds 0–1). Finish **only the better one** (folds 2–3 and 4 as two kernels in parallel, ~4 h each), and only if it lifts the rank-avg by ≥ 0.001. Quota: 2 × 4 h screens + 8 h finish + ~1.5 h zoom tail + blend runs ≈ 19 h of 30.
4. **Next blend (with ConvNeXt, after 5-fold scoring):** use `ours_min=120`. Base time check (10-17 04:10): base958 counts from our start (NB_T0), its R6 stage has `DEADLINE_HOURS = 8.3` from that start (later studies fall back), and the base takes ~5 h 10 min hidden, so our part must stay under ~3.1 h. 120 min leaves 1.1 h margin. 5 ConvNeXt-T folds at 288 add maybe 15–25 min hidden to the ~50–70 min for 2 prep groups + 10 EfficientNet folds. Calibrate from blend958-z's hidden runtime (submitted 02:13; completion time minus that, minus base ~5 h 10 min).
   If the blend958-z submission ranks above base958 in the 0.958 block (or lands higher), it is the pick-A candidate, and a later blend with the ConvNeXt member is one more test. If it ranks below, stop add-on tests: pick A becomes a hedge (see step 6).
5. Daily: forum topics and public notebooks newer than the last check (10-16 20:25 UTC; tooling in `research_notes/public_code_2026-10-16/scores.py`, MCP `list_competition_topics` sortBy Recent). Switch base only if a public notebook beats 0.958 and passes the rule checks (no R7/fastMRI).
6. **Final picks (by 10-22):** B = base958 (57269563). A = the best other candidate: a scored add-on blend that ranks above base958, else blend958-ec (57271112) as a hedge. **Ask the user.** Raise the open OAI question first: Raptor in base958 is OAI-trained, topic 743416 asks whether OAI is "equally accessible" (unanswered at 10-16 20:25), and an R6-only pick would hedge it at a large score cost.
   - The goal is DONE only when the transcript shows `date -u` ≥ 10-22 16:00, `tail -40 HANDOFF.md` ending in a "FINAL PICKS" section (refs, scores, ranks, one line of evidence each), and the submissions list with nothing PENDING.

**Forum / public check 10-17 06:40 UTC** (previous 10-16 20:25)
- Still **no public notebook above 0.958**, so no base switch. The 0.958 block grew: kiyoshiohno template (23:08 rerun), lavinwins rsna-gold-medal (01:28), and heliosli "0.9576" (04:21 rerun, now 0.958) joined goodpjw coatnet-r6 and heliosli blend-mine (our base). goodpjw posted "does S6 help next to R6" = 0.957. More 0.958 copies only crowd the block; our rank inside it follows the unrounded score.
- Forum: no new reply in 743416 (OAI) since 10-16 09:02. New activity is minor (749359 S6 ConvNeXt comment, 743698 GPU question).

**Forum / public check 10-16 20:25 UTC** (previous 13:20)
- No public notebook above 0.958. The 0.958 block is still goodpjw coatnet-r6 (12:48 rerun) and heliosli blend-mine (our base). heliosli "0.9576" was rerun at 19:05 and still scores 0.957. **No base switch.**
- 743416 (OAI "equally accessible"): no new reply since 09:02.
- **New topic 749858 (edteoh), an external-data table for the disclosure ask in 743416.** The author says the host (Chris Carr) asked everyone to *list* the external datasets they used, and that OAI, fastMRI and SKM-TEA (free registration + click-through) are "the kind of access Chris said is fine". That is a paraphrase; verify against 743416 before the final picks. The table: nartaa CoAtNet 384/224 = OAI; heliosli R7 = fastMRI (fm_aux); R6 bundle fe055406 has no fm_aux; `anatomy.pt` (crop placement for R6/R7/S6, from heliosli/rsna-orthofoundation-fold0-epoch7-20261004) = **SKM-TEA**; Raptor (dreaddevelopment) has none.
  - The thread warns that `heliosli/rsna-knee-r6-frozen-fe055406` (no suffix) now holds the R7 bundle. **Our base958 is unaffected**: it mounts the `-w1` copy and asserts `manifest['bundle_id'] == 'fe055406fad6a12f'` plus every weight file's SHA-256. Our run passed (ASSETS_READY), and `runs/knee-base958/submission_receipt.json` has bundle_id fe055406fad6a12f = R6.
  - **Disclosure list for any pick built on base958: OAI (nartaa Raptor weights) and SKM-TEA (anatomy.pt).** Give it to the user with the final picks (write-up / disclosure).

**Tools:** `combos.py runs/A runs/B ...` scores the rank-avg of every subset of members on their common studies (use it for the +0.001 join rule). Merge a member's per-kernel OOFs first with `python -c "import pandas as pd,sys; pd.concat([pd.read_csv(f,index_col=0) for f in sys.argv[2:]]).to_csv(sys.argv[1])" runs/<m>-5f/oof.csv runs/<k1>/oof.csv runs/<k2>/oof.csv`. `wait_kernels.sh` (kernels) and `wait_sub.sh <ref...>` (submissions) both exit within 25 min and are re-armed in the background. To get our rank, download the leaderboard: `python -m kaggle competitions leaderboard -c rsna-knee-abnormality-detection --download -p <scratch>`, then read our row (TeamMemberUserNames contains ayushkasare).

## ON RESUME (2026-10-17 12:30 UTC)
Self-contained; everything above is history. Active `/goal`: the strongest rule-safe final pair by **2026-10-22 23:59 UTC** (last submission by 10-22 12:00 UTC so it scores; at most 4 per UTC day, keeping 1 spare; one change each, with a hypothesis message). **Ask the user before selecting final submissions, deleting anything, publishing, or any dataset with unclear licence/access.** The goal is DONE only when the transcript shows `date -u` ≥ 10-22 16:00, `tail -40 HANDOFF.md` ending in a "FINAL PICKS" section (refs, public scores, ranks, one line of evidence each), and the submissions list with nothing PENDING and every scored submission in EXPERIMENTS.md. Then tell the user to select A and B on Kaggle.

**Leaderboard (snapshot 10-17 07:50, 6,123 teams):** team rank **553**, best = base958 0.958. The 0.958 block is ranks 136–855 (720 teams; we are 418th inside it). 0.959 = 90–135, gold line (rank 22) = 0.963. Silver ≈ rank 306 and bronze ≈ rank 612 both fall inside the 0.958 block. Most of the block are exact copies, and private ties go to the earlier submission, so a *unique* submission that is marginally better is the way out.

**Submissions** (all in EXPERIMENTS.md):
| ref | kernel | public | note |
|---|---|---|---|
| 57269563 | knee-base958 (heliosli blend-mine, R6 -w1 + Raptor OAI) | 0.958 | current best and pick B unless beaten |
| 57271112 | knee-blend958-ec (base958 0.95 + own eff320+cnx288r) | 0.957 | add-on rejected |
| 57291505 | knee-blend958-z (base958 0.95 + own eff320ptl+zoom) | 0.957 | add-on rejected again: **stop own add-on submissions** |
| 57304295 | knee-base958v2 (heliosli V2: retuned Raptor weights + goodpjw reader, -w1 slug) | **PENDING** (submitted 08:58, expect ~14:30) | if it lifts team rank above 553 it becomes the base / pick B |
10-17 submissions used: **2** (02:13, 08:58).

**Running GPU kernels** (wait with `bash wait_kernels.sh knee-train-cnxzptl-c knee-train-cnx288ptl-c`):
- knee-train-cnxzptl-c (ConvNeXt ptl zoom, fold 4), ETA ~15:00.
- knee-train-cnx288ptl-c (ConvNeXt ptl 150 mm, fold 4), started ~11:00 after ~2 h in Kaggle's queue, ETA ~15:50.
These only complete the own members for the record (no more own add-on submissions). **Do not push any new training.**

**Quota:** 20.2 h of 30 h at 12:20 (weekly reset 10-24, after the deadline). The two running kernels end near 26.4 h, which leaves ~3 h for visible runs of baselav958 (~5–10 min) and the union (~15 min), plus any rebuilt union. Check with the Kaggle MCP `get_accelerator_quota`.

**Built, waiting for a GPU slot** (details and gates in EXPERIMENTS.md "Built, not yet run"):
1. `kernels/knee-baselav958` = lavinwins/rsna-gold-medal unchanged (a different family, 0.958; CC BY-NC-SA heads + undocumented training data, **user-approved for one test submission**).
2. `kernels/knee-union958lav` = `python build.py union name=knee-union958lav base=helios958 base2=lavin958` (base958 unchanged first, then lavinwins in its own kernel, then a guarded 50/50 rank blend). **User-approved** via the side session ("Yes. I approve using lavinwins inside a blend for a final pick..."). The side-session code review fixes are in: timm/pydicom snapshot, the TIME_BUDGET → UNION_BUDGET_S patch, the NB_T0 assert, unique-id assert, rho before replace.

**Next steps, in order**
1. ~14:30: when 57304295 scores, log its score in EXPERIMENTS.md, download the leaderboard and log the team rank (`python -m kaggle competitions leaderboard -c rsna-knee-abnormality-detection --download -p <scratch>`, then read the row whose TeamMemberUserNames contains ayushkasare; do not print team names, cp1252). If the rank is above 553, base958v2 is the better base.
2. ~15:00 (cnxzptl-c COMPLETE): `python -m kaggle kernels push -p kernels/knee-baselav958`. When it completes, download it (`python -m kaggle kernels output ayushkasare/knee-baselav958 -p runs/knee-baselav958`) and check the log (2 GPUs, DINOv2 20/20, A5, Rad, Raptor, CoAt family, NOTEBOOK/publish markers, 3×13 no NaN). Submit it (3rd of 10-17) with a hypothesis such as "lavinwins rsna-gold-medal unchanged (different family): measures its score and hidden runtime for the union", and **record the submit time**.
3. ~15:50 (cnx288ptl-c COMPLETE): `python -m kaggle kernels push -p kernels/knee-union958lav`. Visible-run gate: UNION_SNAPSHOT ×2, UNION_BASE_SAVED, UNION_LAV_DONE, UNION_MIX_DONE in the log; 3×13 no NaN; `union_base.csv` equal to `runs/knee-base958/submission.csv`; `union_lav.csv` ≈ baselav958's visible `submission.csv` (allclose, or per-finding rank corr > 0.999).
4. When baselav958 scores: hidden runtime L = COMPLETE − submit. **Submit the union only if 5.2 h + L + 0.3 h < 8.6 h** (base958's ~5 h 11 min is an upper bound). Otherwise do not spend the slot (consider union_h or order changes first). If base958v2 beat base958, a union on helios958v2 is an option (`base=helios958v2`; prefer the less public-tuned base958 unless v2 is clearly better).
5. Finish the own members for the record: merge `runs/cnx288ptl-5f/oof.csv` and `runs/cnxzptl-5f/oof.csv` (concat of each member's per-kernel oof.csv), run `python combos.py runs/eff320ptl-5f runs/eff320z-5f runs/cnx288ptl-5f runs/cnxzptl-5f`, and log it. On folds 0–3 the best own set was zoom + cnx288ptl + cnxz = 0.9083 / 0.933.
6. Daily forum and notebook check (last 10-17 06:40; `python research_notes/public_code_2026-10-16/topscores.py` prints the top-40 by public score and the 40 latest runs; MCP `list_competition_topics` sortBy Recent). Switch base only for a rule-safe notebook that scores above ours. Re-check 743416 (OAI) for an answer.
7. **Final picks (propose to the user by 10-21 18:00 UTC; the user selects on Kaggle). The user now aims for GOLD (10-17 12:35)**, about private rank ≤ 22; the public gold line is 0.963 vs our 0.958. Exact copies share one private score, and ties go to the earlier submission, so our 10-16 base958 copy can never reach gold and is not a safe floor either. **Both picks should be UNIQUE:** A = the strongest unique union that scored ≥ base958; B = a second unique submission that differs from A (another union variant or base). Use an exact public copy as B only if no unique candidate scored ≥ 0.958. Gold levers: a third rule-safe family fitted into one 9 h notebook by trimming optional parts; a second union on helios958v2; late public releases (check twice a day from 10-20); NB_T0 / order engineering. Full text in GOAL_NEXT_SESSION.md. Disclosure list: OAI (nartaa Raptor), SKM-TEA (anatomy.pt) for base958; plus lavinwins' sources (antoinegg1 CC BY-NC-SA 4.0 heads, mattiaangeli / pilkwang weights with undocumented training data, DINOv2, RadImageNet) for anything with lavinwins; goodpjw reader (Apache 2.0) for base958v2.

**Key facts to keep**
- Never use a base with R7/fastMRI: `heliosli/rsna-knee-r6-frozen-fe055406` (no suffix) now holds R7; always the `-w1` copy, with bundle_id fe055406fad6a12f asserted.
- Own models: best single cnxz (folds 0–3 0.9072), cnx288ptl 0.9061, zoom 5-fold 0.905, eff320ptl 0.903. Their gold-58 is ~0.93, below the public stacks, which is why add-ons at 0.05 lose ~0.0005–0.001.
- Hidden runtime = submit → COMPLETE (includes any queue wait). base958 ~5 h 11 min. The blend layout with our models costs only ~5–20 min extra.
- Kaggle may queue a kernel ~2 h for T4 capacity (not charged); submissions do not take a GPU session slot.

**Tools:**
- `wait_kernels.sh k1 k2` and `wait_sub.sh ref`: both exit within ~25 min; re-arm them in the background.
- `combos.py`: every subset's rank-avg on common studies.
- `oof_report.py`: per-label.
- `build.py union|blend|base|train|infer|prep`, documented in CLAUDE.md.
- Pulled public notebooks: `ref/_new_2026-10-17/`; bases: `baseline/helios958`, `baseline/helios958v2`, `baseline/lavin958`.

The `/goal` text for the next session is in `GOAL_NEXT_SESSION.md` (written 10-17 12:30).

**Session 2194aef4 stopped at 12:34 UTC on the user's request.** No background waits are running and nothing more was pushed or submitted from it. Still running on Kaggle: knee-train-cnxzptl-c, knee-train-cnx288ptl-c. Pending: submission 57304295 (base958v2). The next session re-arms the waits and starts from this resume section.

**Session 748551ef (started 12:35 UTC, new shorter `/goal`: aim for gold, plan above is a starting point, ask before final picks / deletes / publishing / unclear-licence data).**
- 12:40 quota 20.7 h used of 30. Waits re-armed (kernels + sub 57304295).
- 12:45 daily check done (EXPERIMENTS "Daily check 10-17 12:45"): nothing new usable (new 0.958 goodpjw / kiyoshiohno mount R7; heliosli "0-9586" = our base958v2 code). Forum 750236 asks hosts how external data is checked for medals (unanswered). nartaa (749359) states the base958 Raptor CoAtNet used OAI via NDA Study 3407; lavinwins' Raptor is dreaddevelopment's (competition data), so a lavinwins-only or lavinwins-heavy pick is the OAI-free hedge.
- 12:50 `kernels push knee-baselav958` refused: "Maximum batch GPU session count of 2 reached" (no queueing). Push again when the first -c kernel completes.
- 14:35 cnxzptl-c COMPLETE (f4 0.9044); merged `runs/cnxzptl-5f/` = **0.9065 / 0.934**, best single; zoom + cnxz 0.9074 / 0.933 (EXPERIMENTS). baselav958 auto-pushed into the freed slot.
- 14:41 **base958v2 = 0.958, no team-rank gain** (team rank 662: the 0.958 block grew to 840 teams, ranks 139–978; we are ~50 places below the public bronze line). base958 stays the base.
- 14:47 union958lav pushed (visible run). 14:49:41 **baselav958 submitted (ref 57315208)** after its visible gates passed (350 s, all markers). Its hidden runtime L = COMPLETE − 14:49:41. 10-17 submissions used: 3 (one more allowed today, keep the 5th spare).
- Quota at 14:45: 25.0 h used of 30; cnx288ptl-c still running (ETA ~15:50).
- Lavinwins visible phases (3 studies): dinov2 90 s (mostly loading 20 members), A5 24 s, Rad 15 s, Raptor 44 s, d4 + residual CoAt 64 s, repairv1 + global96 CoAt 75 s. It sizes its DINO cache from free RAM at start ("29.4 GB available, 13.2 GB to the cache") and peaked at 16.6 GB, so in the union check the "memory: X GB available" line (base958 must leave RAM free).
- 15:05 cnx288ptl-c COMPLETE; own models finished: best own set zoom + cnx288ptl + cnxz 0.9078 / 0.934 (record only).
- 15:19 built and pushed **knee-union958lav-lf** (`first=base2`, new build option): lavinwins first, then base958 with NB_T0 patched. v1 (lav_h 3.3) visible run 740 s: all markers; union_base == base958 visible and union_lav == baselav958 visible (max abs diff 0); submission == 50/50 rank mix; 3×13, ids OK. Preferred design: each notebook runs exactly as alone (lavinwins only pip-installs into `--target` dirs). v2 = lav_h 3.6 (pushed 15:35): base958's R6 deadline (8.3 h from NB_T0) caps the total at ~8.4 h whatever lavinwins does, and 3.6 h keeps lavinwins in for L up to ~3.6 h at the cost of ≤ ~12% late studies getting Raptor-only on the base half. Plan: submit v2 as the 4th 10-17 submission once its visible gates pass.
- 15:53:00 **union958lav-lf v2 submitted (ref 57317084)** after its visible gates passed (872 s; identical halves; exact 50/50 mix). 10-17 submissions: 4 used (02:13, 08:58, 14:49, 15:53); the 5th is the spare. Waiting with `wait_sub.sh 57317084 57315208`. Expected: lavinwins alone scores at 14:49 + L; the union at 15:53 + L + ~5.2 h (≤ 8.4 h).
- 15:56 the user forked this session (fork 3bd2c5eb). **Two sessions may be live: before any push or submission, re-read the submissions list and this section so nothing is done twice.** 10-17 is out of submissions anyway (4 used).
- 15:56 a forked session (local_3bd2c5eb, "btw: how was the last two submitions performming") also watches 57315208 / 57317084. Agreed by message: this session (748551ef) owns pushes, submissions and log edits; the fork stays read-only; whoever sees a score first logs it, the other re-reads EXPERIMENTS.md first. Not a user approval of anything.
- ~16:05 on the user's request, 3 read-only research agents (Sonnet) launched: (1) knee competition rules/leaderboard/code/models → `reports/2026-10-17 knee code models leaderboard.md` + `research_notes/public_code_catalog_2026-10-17.csv`; (2) knee forum → `reports/2026-10-17 forum update.md` + `research_notes/forum_2026-10-17/`; (3) the previous RSNA competition → `reports/2026-10-17 prior RSNA competition.md` + `research_notes/prior_rsna_2026-10-17/`. Synthesize them into the plan when they land.
- 16:30 user asked "what haven't we tried / are hesitating on". Analysis in `reports/2026-10-17 how to win.md`: (1) we dropped our own models after two underpowered 0.05 add-ons; our set (OOF 0.9078 / gold 0.934) is at the level of the best public single models (S6: OOF 0.907 / gold 0.931 → LB 0.950) and has never been LB-measured since 10-04; test it as a real third family; (2) OAI hedge: lavinwins ⊕ ours is an OAI-free unique pick; (3) TPU quota (20 h) is unused, the only compute left for training. Proposed to the user; **user (16:35): "wait for the other agents to get their research get over"**, so nothing new is built until the three reports are in.
- 16:40 forum agent done → `reports/2026-10-17 forum update.md`. Key points:
  - No host post since 10-14. 18 asks are unanswered, including 5 new on 10-17 about nartaa/OAI, anatomy.pt, R7/fastMRI and RadImageNet.
  - The OAI geo issue (China / Login.gov) is open. nartaa's OAI use is limited to PF OA, Lateral OA and Synovitis.
  - heliosli's `anatomy.pt` (in R6 and every heliosli stack) is SKM-TEA/KneeXNet-derived with no licence, and KneeXNet comes from OAI. So **base958 carries OAI twice** (nartaa Raptor and anatomy.pt).
  - **A union is no compliance hedge**: one non-compliant member sinks the pick. The real hedge has no nartaa, no heliosli, no R7 and no goodpjw reader, i.e. lavinwins alone (we verified `baseline/lavin958` mounts no heliosli/nartaa) or lavinwins ⊕ ours.
  - Enforcement is stated only for prize winners (rules 2.8, 3.9.b); a medal purge looks impractical (~90% of ≥ 0.95 notebooks use nartaa).
  - Ties go to the earlier submission (rule 3.7.b).
- 16:45 prior-RSNA agent done → `reports/2026-10-17 prior RSNA competition.md` (RSNA 2025 Intracranial Aneurysm, plus 2024 Lumbar). Key points:
  - **The exact-copy block got no medals.** 229 of 246 copy teams shared one private score, at private ranks 245–507 of 1,148; Lumbar was the same.
  - **Public-LB tweakers of one family lost** a median 0.011 privately. Top ranks were stable (Spearman 0.945), and none of the top 12 was a public blend.
  - Heterogeneous ensembles held up on private even when public did not show it.
  - **Hidden runtime varied run to run** (8 h one day, a timeout the next). Five of the top 9 had timeouts. A timed-out run scores nothing, so keep a real margin under 9 h.
  - Compliance rulings came only by thread; there is no evidence that medal positions were audited.
  - Set both picks by hand (otherwise Kaggle takes the best public).
- 16:50 code/leaderboard agent done → `reports/2026-10-17 knee code models leaderboard.md`, `research_notes/public_code_catalog_2026-10-17.csv`. Key points:
  - Nothing above 0.958 is public, and no notebook has been created since 10-15.
  - The 0.958 block now holds ranks 142–992, and we are at 674.
  - Shake-up simulation for a unique entry: at public 0.958 about 1% gold, 20% silver, 39% bronze; at 0.961 about 5% gold, 40–50% silver.
  - **Pick B must differ in composition.**
  - R6 run name `cnx_r6fm`: flip-mirror or fastMRI is unverified. It leans clean (fm_aux appears only in R7's manifest), but disclose it.
  - Synthesis and recommendation are in `reports/2026-10-17 how to win.md`: A = the union (57317084), B = lavinwins ⊕ ours (OAI-free, unique, ~L + 0.5 h). **Asked the user for a go-ahead.**
- 16:40 user: "evaluate the whole situation and act on the best choice". Decision (`reports/2026-10-17 how to win.md`):
  - A = the union (57317084, pending).
  - **B = lavinwins ⊕ ours (base 0.7 / ours 0.3)**: unique, different composition, OAI-free, and it tests our models as a family.
  - Built `kernels/knee-blendlav-ours` (new `nb_t0=0` blend option); visible run pushed 16:39. Submit on 10-18 after the gates, and after L is known (runtime gate: ours ≤ 1.5 h + L < 8.5 h).
  - Quota at 16:37: 25.75 h used, 4.25 h left.
- knee-union958lav (base-first) visible run has been QUEUED since 14:47; now only a backup (could not cancel: the MCP gives no session id).
- (superseded by the -lf build) If L ≥ ~3.1 h the base-first union falls back to base958. Fallback design (not built): lavinwins FIRST via nbconvert, then base958 with NB_T0 patched (as blend mode), so base958's R6 deadline (8.3 h from NB_T0, R6 runs last after the Raptor stages) degrades only its late studies. Only worth it if L ≲ 3.5 h; beyond that something must be trimmed.

### 10-18 (session 748551ef, resumed 07:55 UTC)
- **lavinwins alone (57315208) = 0.943; union A (57317084) = 0.953.** Both plans built on lavinwins are dead.
  - Cause: lavinwins' "0.958" is the notebook's best score over all versions. v1–v5 were heliosli/nartaa copies (3–4 sources). v6 (10-17 01:28, the one we copied, 19 sources) calls itself "v6 combination layer on the **0.943** public pipeline, no new checkpoints". Our visible run was clean (DINOv2 gate PASS, 0 fallbacks), so 0.943 is its real level.
  - **Lesson: a notebook's public badge is its best version, not its current one. Check the version history (`kernels.KernelsService/ListKernelVersions`, web endpoint, kernelId) before trusting it.**
  - No OAI-free family at ≥ 0.95 remains. The OAI hedge is now only "ours alone" (≈ 0.94–0.95, too weak to medal).
  - L was not measured (session ended; > 2.9 h).
- Team rank **846 / 6,165** at 08:00. The 0.958 block is ranks 162–1178. Silver (rank ~308) and bronze (~616) lines are both inside it; the gold line (rank 22) is now 0.964. 0.959 = ranks 104–161.
- Quota 07:58: 26.06 h used, **3.94 h left**. A visible run costs ~6–13 min (blendlav-ours 6 min, union 13 min), so quota is not the limit. The T4 queue is: on 10-17 the pushes sat QUEUED for > 1 h.
- **New plan: our own set as a real third family next to base958.** It is the only lever left with upside (d′ model: +0.0006 to +0.003 at ours 0.25 if ours ≥ 0.943 and corr ≤ 0.87).
  - Pushed 07:59: `knee-infer-own15` (our 15 folds alone, first LB of our set since 10-04) and `knee-blend958-own25` (base958 0.75 + ours 0.25). Rows in EXPERIMENTS "Built".
  - Order: submit own15 first (hidden ~1 h) → its score sets the weight → submit the blend at 0.25 if own15 ≥ 0.943, at ≤ 0.1 or not at all if ≤ 0.940.
  - Then a second weight if the first moves our team rank up.
  - knee-blendlav-ours (visible passed) and knee-union958lav: not to be submitted.
- Picks if nothing beats 0.958: A = base958 (57269563, earliest copy we hold), B = base958v2 (57304295). Both are copies, so they are weak (copy blocks did not medal in prior RSNA).

## COMPETITION OVER (checked 10-25)
- Final: **private 0.942, rank 1,774 / 6,425** (the public-copy block, ranks 616–1937). No submissions after 10-17; own15 / blend958-own25 were never submitted.
- Post-mortem against the top solutions: `reports/2026-10-25 post-mortem top solutions.md`. Write-ups saved in `research_notes/solutions_2026-10-25/`.
- Main lesson: the winners' edge was **image-reading LLM labels** (gold 0.89–0.91 report-only → 0.94–0.956 image + report + teacher), not models or blends.

## FINAL PICKS (resolved after the close; recorded 10-25)
- **No picks were selected or approved in time.** This session stopped on 10-18 and nothing was submitted after 10-17, so the user never approved two picks. With no selection, Kaggle scores the two best public submissions:
  - **57269563** knee-base958 (heliosli 0.958 recipe unchanged): public 0.958, private 0.942.
  - **57304295** knee-base958v2 (heliosli V2, R6 `-w1` copy): public 0.958, private 0.942.
- Team result: **private 0.942, rank 1,774 / 6,425** (public rank ~846 on 10-18). Every base958-family submission scored 0.942 private, so the pick did not change the result.
- Evidence for these two: the only 0.958-public submissions we held, both public copies. The private copy block (ranks 616–1937) took no medal past its first ~27 teams.
- **Goal outcome: not met.** Gold needed 0.954 private. Why: `reports/2026-10-25 post-mortem top solutions.md`.
