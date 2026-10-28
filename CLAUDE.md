# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Kaggle competition code for **RSNA Knee Abnormality Detection** (macro AUC over 12 findings, notebook-only submissions, team `ayushkasare`).

- **Read `HANDOFF.md` first.** It holds the current state, what is running, and the next steps.
- **Check `EXPERIMENTS.md` before running anything.** It is the log of every experiment and result. Add a row the moment a result lands.

## Environment constraints

- Nothing heavy runs locally. Local torch is CPU-only, and the DICOMs (570 GB) are never downloaded; `data/` holds only the competition CSVs. All prep, training and inference run as **Kaggle kernels**. Locally we only build kernels, run tests, and analyse downloaded outputs in `runs/<kernel>/`.
- Use the Kaggle CLI as `python -m kaggle ...` (it is not on PATH).
- Background shell jobs here die after ~30 min. Long waits therefore use `bash wait_kernels.sh k1 k2` (kernels) or `bash wait_sub.sh <ref...>` (submissions leaving PENDING), re-armed as needed: each exits when any listed item finishes or errors, or after ~25 min.
- Kaggle limits:
  - at most 2 GPU sessions at once;
  - weekly GPU quota (check it with the Kaggle MCP `get_accelerator_quota`); CPU kernels are free;
  - 12h per kernel (the training loop budgets 11.3h) and a 9h submission runtime;
  - 5 submissions per day.
- Use fp16 autocast, never bf16, on T4.

## Commands

```bash
python test_knee.py                      # all self-checks (CPU, plain asserts, prints "ok"); no pytest — the last line calls each test_* function, so importing test_knee runs them all
python -c "exec(open('test_knee.py').read().rsplit('\n', 2)[0]); test_mirror()"     # one test: everything but that last line, then call it
python build.py prep  name=knee-prep320a nslice=31 size=320 prep=1 shard=0:4          # CPU cache kernel
python build.py train name=knee-train-X backbone=efficientnet_b0.ra_in1k epochs=6 folds=0,1 \
       pseudo=0.5:knee-train-rad cache=knee-prep31a,knee-prep31b,knee-prep31c [img=288 labels=rubric headlr=1e-3 posw=1 slotdrop=0.15 rot=25 aux=0.3 mirror=0.5 smoke=N]
python build.py infer name=knee-infer-X trains=knee-train-a,knee-train-b              # our models alone
python build.py blend name=knee-blend-X blend=0.85 base=goodpjw-v21 trains=...        # ours first, public base, rank blend (base weight 0.85)
python build.py base  name=knee-base957 base=goodpjw-v21                              # public notebook in baseline/<base>/, unchanged
python build.py union name=knee-union958lav base=helios958 base2=lavin958 [blend=0.5] [union_h=8.6]  # base unchanged, then base2 unchanged in its own kernel, then a guarded rank blend
python -m kaggle kernels push -p kernels/<name>
python -m kaggle kernels status ayushkasare/<name>
python -m kaggle kernels output ayushkasare/<name> -p runs/<name> --file-pattern "(\.log|oof\.csv|config\.json)$"
python oof_report.py runs/knee-train-a runs/knee-train-b   # per-label OOF AUC vs report labels and vs the 58 gold studies, rank-avg, Spearman
python -c "import pandas as pd,sys; pd.concat([pd.read_csv(f,index_col=0) for f in sys.argv[2:]]).to_csv(sys.argv[1])" runs/<m>-5f/oof.csv runs/<k1>/oof.csv runs/<k2>/oof.csv   # merge one recipe's fold-split kernels
python combos.py runs/eff320ptl-5f runs/cnx288r-5f runs/X  # rank-avg macro AUC of every subset of members on their common studies (the +0.001 join rule)
python stack.py runs/... [--meta runs/knee-meta/meta.csv]  # OOF logistic-regression stacker (has not helped so far)
python consensus.py                                        # rewrites labels/knee-rubric-labels/consensus_labels.csv, then:
python -m kaggle datasets version -p labels/knee-rubric-labels -m "<what changed>"
python -m kaggle competitions submit -c rsna-knee-abnormality-detection -k ayushkasare/<name> -v <ver> -f submission.csv -m "<hypothesis>"
python -m kaggle competitions leaderboard -c rsna-knee-abnormality-detection --download -p <scratch>   # our rank: the row whose TeamMemberUserNames has ayushkasare
```

`folds=` omitted trains all 5 folds. Long recipes split folds across kernels named `-b`, `-c` (e.g. knee-train-eff320ptl f0–1 + `-ptl-b` f2–4); their OOFs are merged into `runs/<recipe>-<k>f/oof.csv` before `combos.py`.

`build.py` checks that every kernel in `trains=` is COMPLETE before building an infer or blend kernel. Smoke builds get a `-smoke` suffix (a smoke train runs fold 0 only, 2 epochs, on N cached studies, repeated if the cache is smaller), and infer skips any config with `SMOKE` set *or missing* unless built with `allow_smoke=1`. Other knobs: `cpu=1` (no GPU), `nslice=16` on infer/blend when every attached model is 16-slice, and `nb_t0=0` on blend for a base with its own clock instead of `NB_T0` (e.g. `base=lavin958`; then check by hand that our part plus the base's runtime fits in 9 h). An unknown option fails the build (typos used to be ignored silently).

## Architecture

**`knee.py` is the single source for every kernel.** `MODE` (`prep` / `train` / `infer`) selects `run_prep`, `run_train` or `run_infer`. `build.py` copies it into `kernels/<name>/<name>.py` and writes `kernel-metadata.json`, which carries the input sources: competition data, cache kernels, pseudo-label teacher kernels, label datasets, and trained model kernels for infer. It sets options by **regex-rewriting top-level assignment lines** (`put()`), e.g. `^EPOCHS, FOLD_LIST = ...` or `^BACKBONE, IMG = '...', 224`. Keep those lines' exact shape when editing `knee.py`, or the build asserts. `blend` mode runs **ours first**: a `%%writefile` of `knee.py` (infer, `BLEND` 0), then a cell that runs it as a subprocess (timeout `ours_min`, default 60) and renames its output to `ours.csv`; then the public base unchanged except its clock start (`NB_T0 = time.time()` → `globals().get('NB_T0', ...)`, asserted exactly once), so the base's own 9h guards count our time; then a last cell that rank-blends `ours.csv` into the base's `submission.csv` at base weight `blend=`. Our two cells never raise: any failure keeps the base file. (Appended last, our part would hit the base's late-clock guards on the hidden test and drop out silently.)

**Data flow**
1. **prep** (CPU, sharded with `SHARD`; Kaggle output is capped at 20 GB):
   - writes one `cache/<study>.npz` per study: `x` uint8 `[5 slots, N_SLICE, SIZE, SIZE]` plus mask `m [5]`, plus `prep_done.json`;
   - slots are `SLOTS` (sagittal/coronal/axial × fat-sat);
   - each series is sorted along its stacking axis, `N_SLICE` positions are sampled with `pick_idx`, and the image is cropped to `CROP_MM` and percentile-normalised;
   - everything is oriented to a canonical **left knee**: sagittal is reversed for right knees, other planes are mirrored.
2. **train**:
   - reads the caches attached via `CACHE`, using 5 **report-grouped** folds (`report_folds`: identical reports never straddle folds);
   - runs one process per GPU (DataParallel was much slower);
   - `to_input` builds 2.5D 3-slice stacks and does GPU augmentation, training on a random half of the slice centres and inferring on all of them;
   - writes `fold*.pt` (fp16 state dicts), `oof.csv` and `config.json`.
3. **infer**:
   - globs `config.json` across all attached kernels; every fold model is weighted equally;
   - preps the test set once per `(PREP, SIZE, CROP_MM)` group, feeds each model its own `IMG`, averages probabilities, then takes per-label ranks.

**`union` mode** joins two public notebooks:
- the base's cells run first and unchanged, then its `submission.csv` is copied to `/tmp/union/base.csv`;
- a first cell snapshots the docker's own timm and pydicom to `/tmp/union/site` before the base pip-installs its pins over them (helios958: timm 1.0.29 → 1.0.25, pydicom 3.0.2 → 3.0.1);
- `base2` is embedded with `%%writefile` and run through `python -m nbconvert --execute` in its own kernel, with `PYTHONPATH=/tmp/union/site` and a timeout of `union_h` hours after the base's `NB_T0`. Its only edit is `TIME_BUDGET = 8.0 * 3600` → the env var `UNION_BUDGET_S` (the real remaining time; unchanged when run alone; the build asserts 2 such lines). On timeout the whole process tree is killed via psutil (never `pkill` by name, since the notebook's own kernel is an ipykernel too);
- the base's file is restored whatever happens, and a last cell rank-blends the two only if ids, columns and values check out.
`first=base2 [lav_h=3.3]` reverses the order: base2 runs first via nbconvert with a fixed `lav_h`-hour budget (no snapshot needed), then the base with `NB_T0` patched as in blend mode, so the base's own deadlines count base2's time and only its late studies degrade; then the mix.
The two notebooks must pin the same docker image (asserted), and their data sources are merged. `union_base.csv` and `union_lav.csv` are written as diagnostics.

**`config.json` is the contract between train and infer.** It records BACKBONE, IMG, SIZE, N_SLICE, HEAD, PREP, LABEL_SRC, PSEUDO and the training knobs. Older runs lack some keys, and `run_infer` falls back to defaults (`HEAD` 'slot', `PREP` 1, `N_SLICE` 16, `SIZE` 256, `IMG` 224, `CROP_MM` 150). Keep those fallbacks when adding keys.

**Key invariants**
- Every other position of the 31-slice `pick_idx` equals the 16-slice positions (`test_dense_compat`), so 16-slice models still run on a 31-slice test prep. Only PREP 1 models may be 16-slice (`run_infer` asserts it).
- Prep behaviour depends on the module globals `PREP`, `SIZE` and `N_SLICE`, read inside `multiprocessing.Pool` workers (fork start method). `run_infer` sets those globals *before* calling `prep_all`.
- `PREP` 1 vs 2 changes the slice band, the crop and the slot fallback. Train and test prep for a given model must use the same version. `knee.py` defaults to `PREP = 2`, but the caches behind the best models (`knee-prep31*`, `knee-prep320*`) are PREP 1, so pass `prep=1` when building a prep kernel meant to match them. Training reads the version and `CROP_MM` from each shard's `prep_done.json` and asserts all shards agree. `cropmm=` sets the centre-crop field of view (150 mm default; the `knee-prep320z*` caches are a 110 mm zoom).

**Model** (`make_model`): a timm or RadImageNet ResNet-50 encoder per image, then a head.
- `HEAD='mil'`: per-finding attention over all (slot, slice) tokens, with slot and slice-position embeddings; missing slots are never encoded.
- `HEAD='slot'`: the legacy head, kept so old checkpoints load.
- RadImageNet takes the centre slice ×3 scaled to [-1, 1]; other backbones take neighbouring slices with ImageNet normalisation.
- LR depends on the backbone: 3e-4 by default, 1e-4 for RadImageNet, CoAtNet and ConvNeXt, 3e-5 for DINOv2. `HEAD_LR` optionally splits the head off. Batch size is 4 per GPU, 2 for CoAtNet/ConvNeXt (memory).

**Targets** (`load_targets`):
- `LABEL_SRC` `v4b` = stevenleehans `llm_labels_v4_blend` with `SILENT_FIX` (report silent → low prior); `rubric` = our LLM rubric labels dataset; `consensus` = `consensus.py` output (mean of 4 LLM reads, uncertain cells NaN), in the same dataset.
- `MIRROR` p (build `mirror=`): per study, the contralateral knee (sagittal slices reversed, coronal/axial flipped on width), Medial↔Lateral Meniscus/OA targets swapped, MCL NaN. Such models (config `MIRROR` > 0) get flip TTA in OOF `predict` and in `run_infer`.
- A NaN target means "no loss on this cell": the train loss is `masked_bce`, which averages over non-NaN cells only.
- Optional soft bootstrapping: `PSEUDO_A` blends in the mean OOF of the teacher kernels in `PSEUDO_FROM`.
- The 58 expert-labelled ("gold") studies always get their gold labels.
- Evaluation is always against `y_eval` (v4b), whatever the training labels.

**Local analysis scripts** (`oof_report.py`, `stack.py`) import `knee` and monkeypatch `knee.find_input` to map Kaggle input names to `data/` and `labels/`.

## Repo layout notes

- `baseline/`: blend bases. The root holds our fork of the public jiweiliu stack (LB 0.943, the default); `baseline/<subdir>/` holds newer public notebooks as pulled (notebook + their kernel-metadata.json), selected with `base=<subdir>`.
- `ref/`: pulled public notebooks, for reading only. `ref/_logs/` holds run logs of 39 key public notebooks, and `ref/_heliosli_compact_src/` holds heliosli's R6/R7/S6 trainer.
- `research_notes/public_code_catalog.csv`: every public notebook with its best public score, model families and notes (as of 10-16). `research_notes/forum_2026-10-16/` holds every forum topic as text. `research_notes/public_code_2026-10-16/` holds the scripts that fetched them (`topscores.py` lists the top public notebooks by score; rerun it for the daily check).
- `labels/`: public and our own label datasets.
- `reports/`, `research_notes/`: competition research. The newest reports (`reports/2026-10-16 forum insights.md`, `reports/2026-10-16 public code insights.md`) drive the current plan in HANDOFF.md.
- `kernels/`: generated build output, never hand-edited. Edit `knee.py`/`build.py` and rebuild. Exceptions: `knee-meta` (DICOM header metadata for `stack.py --meta`) and `knee-profile` are hand-written one-offs that `build.py` does not produce.

## Judging results

- Trust report-label OOF first (~4400 studies).
- Gold-58 is noisy (±0.01–0.02) and has misled LB decisions before.
- Public LB differences under ~0.003 are noise; only the private LB decides prizes.
- The public score shows 3 decimals, but rank inside a block of equal scores follows the unrounded score; exact ties (an unchanged public notebook) are ordered by submission time, so a copy lands behind earlier copies. For two variants of the same base, a jump in our rank is the finer signal.
- Submit only to calibrate a major change, with the hypothesis in the message.
- Use Sonnet, not Opus, for multi-agent fan-out.
