# Post-mortem: what the top 5 did and where we went wrong (2026-10-25)

Sources: write-ups saved in `research_notes/solutions_2026-10-25/` (`fetch.py <topic ids>` re-fetches them): 1st (752468 tascj, 752474 sakaku), 2nd (752370), 3rd (752510), 5th (752190), plus 7th (752217) and 8th (752219). **4th place (Ilya Kolb) has not posted.** Private leaderboard paged via the SDK on 10-25.

## Our result
- **Private 0.942, rank 1,774 of 6,425.** Public 0.958.
- Every base958-family submission scored 0.942 private. That includes the 0.05 own add-ons, which were not really different.
- 0.942 is the copy block: 1,322 teams at ranks 616–1937. Bronze (rank 642) took only the first ~27 of them. Silver (321) needed 0.943; gold (22) needed **0.954**.
- Our models alone: 0.938 public / 0.924 private (10-04 set). Our best set (OOF 0.9078, 10-17) was never submitted: `knee-infer-own15` and `knee-blend958-own25` sat built after 10-18 and the session ended before they were used.
- Public → private: the public stacks lost **0.016** (0.958 → 0.942); the winners' own models lost only ~0.009–0.011 (e.g. 0.969 → 0.958). The public blends were tuned on the public split.

## The top 5 (private)

| # | Team | Private / public | Best single model (private) | What it was built on |
|---|---|---|---|---|
| 1 | Team Kangaroo (tascj, sakaku) | 0.960 / 0.969 | 0.955 (MaxViT-v2-B); efficiency-track ConvNeXt-T 256 px alone **0.954** | Agentic GPT-6 Astra labels (reads report + MRI with a slice viewer, ~2.5 min/study), LLM-drawn ROI boxes + a detector, lesion boxes; box-pooling ConvNeXt-T / MaxViT / MambaOut at 384 px |
| 2 | zzzoll, azamat1ch | 0.960 / 0.969 | 0.957 (fastMRI-pretrained MIL reader) | Report read (0.899 on gold) → Astra image read (0.942) → OOF teacher mix (0.956 on gold); 6 2.5D MIL models with knee-centred close-ups; fastMRI and OAI (MOAKS) for two members |
| 3 | DataTech Club | 0.959 / 0.968 | baseline BiT-50 **0.951**, best 0.953 | Astra image + report labels with finding-specific re-reads; OAI MRI mixed into training; 36 checkpoints (BiT / ConvNeXt-S / DINOv3 / CoAtNet, 224–512 px) |
| 4 | Ilya Kolb | 0.958 | – | no write-up yet |
| 5 | RB & Matheus & Tucker & Tony | 0.958 / 0.969 | 0.958 (ConvNeXt-B) | Calibrated public report labels → Opus 5.5 report reads + gpt-6.1-sol on 10 fluid slices → teacher blend; nnU-Net bone masks place 4 anatomy crops; OAI MOAKS pretraining (+0.001) |
| 7 | BarryZhou (solo) | 0.957 / 0.970 | **0.958** (ConvNeXt-S) | **No external data.** 3 Astra reads (structured, image-only, "verify the report against the images") = 0.947–0.952 on gold vs 0.89–0.91 for report-only reads; 13 fixed zoom boxes; ensemble added nothing |

## What every winner did
1. **They treated it as a labelling competition, and the labels came from an LLM that looked at the images.**
   - Report-only reads plateau at 0.89–0.91 on gold-58, the same plateau we hit (ours: v4b 0.893, rubric 0.898, rubric + v4b 0.903, Opus pilot 0.910).
   - Image + report reads reach 0.935–0.947; blends and OOF-teacher mixes reach 0.952–0.956.
   - Measured effect on the leaderboard:
     - 1st: same student 0.922 → 0.935 public.
     - 5th: +0.008 private.
     - 7th: +0.006 private.
     - 3rd's plain baseline on these labels: 0.951 private, which beats the whole copy block.
   - **How:** `claude -p --json-schema` / `codex exec` in headless mode, one session per study, often on subscription quota ("a few hours for a full labeling", 5th). The host had allowed commercial LLMs for labels (thread 733965).
   - **Soft labels, calibrated on gold-58, then a self-distillation round:** OOF predictions of models trained on these labels are mixed back into the labels (1st, 2nd, 5th, 7th).
2. **Higher resolution and more slices:** 384 px on a ~140 mm field of view; 15–48 windows per series; 20–25 epochs; EMA/SWA; higher backbone LR (1st: +0.008 public from the LR alone).
3. **Look at the right anatomy:** ROI boxes, bone-mask crops, or fixed zoom boxes. 5th: +0.003 private. 1st: 0.956 → 0.960 public.
4. **A ruler that matches the test.** Not OOF against report labels: 5th says it "rewards agreeing with the reports, not with the experts". They used:
   - gold-58 for label versions;
   - a CV against the image-read labels, picked for its correlation with the LB (1st);
   - site-grouped folds (5th).
5. **Ensembles mattered little.** Single models reached 0.954–0.958 private, and ensembling added ≤ 0.002. The rank came from labels and inputs.
6. Training ran on rented GPUs (2nd: ~100 GPU h, ~$100); Kaggle ran only the inference.

## Where we went wrong, by cost
1. **Our label ceiling was the report, and we called the gap noise.**
   - EXPERIMENTS: "report labels saturate ~0.90", and every label set was "within gold noise". That reading was right about report labels and wrong about the task: the test is image-read, and image-read labels were the +0.015 step.
   - On 10-16 the forum showed Effusion 0.835 → ~0.92 on gold once an LLM saw the MRI [745949]. We wrote it off as "not feasible in 6 days" because we pictured rendering montages on Kaggle.
   - The winners ran it from a laptop with `claude -p`, the tool we were working in.
   - Our 10-04 research report said "no technique is a +0.015 step". The label source was that step.
2. **We measured with the wrong ruler.**
   - CLAUDE.md said "Trust report-label OOF first". That OOF scores agreement with the reports, so it could never show that the reports were the limit.
   - It also judged our own models against the target the public stacks were not optimised for. Our "own add-on hurts" conclusions came from 0.001-step public reads.
   - Gold-58 alone was too noisy, and we built nothing between the two (no image-label CV, no site folds).
3. **The last week went to public notebooks.**
   - From 10-15, all 7 submissions were copies, unions or 5% add-ons of public stacks, including a day lost on lavinwins (its badge was an older version's score).
   - The prior-RSNA research (10-17) already showed copy blocks do not medal. Here the block dropped 0.016 privately and only its first ~27 teams took bronze.
4. **Model inputs were a step behind:** 224–288 px, 6 epochs, EfficientNet-B0 / ConvNeXt-T on the Kaggle T4 quota (~4 h per ConvNeXt fold), against 384 px and 20–25 epochs. Our own data pointed the same way (224 → 288 px +0.006 OOF; 110 mm zoom +0.003), but compute capped it.
5. **We started late with a weak first plan.** First submission 10-02 (the competition opened in early August; welcome post 08-06). Twelve days (10-04 → 10-15) produced no submission while we iterated on report-label variants. Top teams made 99–238 submissions; we made 18.
6. **Our own set was never measured on the LB after 10-04.** Built 10-18, never submitted. Even if it had scored, the gap (~0.93 vs 0.954 private) came from labels, not from a blend weight.

## What we would do again, and differently
**Kept (it worked or matched the winners):**
- canonical-left orientation;
- 31-slice MIL;
- zoom crops;
- soft labels with NaN masking;
- the visible-run gates;
- logging everything.

**Next medical-imaging competition with a report-only training set and an image-read test:**
1. **Week 1: an agentic image + report labeller.** Use `claude -p --json-schema`, one workspace per study: overview sheets, a slice viewer, the report treated as a draft to verify. Score every label version on the expert subset by AUC, **log loss and confident errors**, with a held-out half.
2. Make the main CV the one that tracks the LB: image-read labels, site-grouped folds. Never report-label OOF.
3. 384 px, ~140 mm field of view, anatomy crops, 20–25 epochs, EMA, one self-distillation round. Rent the training GPUs; keep Kaggle for inference.
4. Treat public notebooks as reading material and blend members, not as picks; check a notebook's version history before trusting its badge.
