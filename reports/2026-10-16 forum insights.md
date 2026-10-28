# Forum insights, 2026-10-16 ~06:00 UTC

Sources: every one of the 197 competition discussion topics with all comments (pulled through the Kaggle SDK on 10-16), plus the public leaderboard CSV of 10-16 05:43 UTC. Topic ids are in brackets. This adds to `reports/2026-10-15 leaderboard research.md`; anything already there is not repeated unless it changed.

## What changed since the 10-15 report

### Host rulings of 10-14 (Chris Carr, RSNA Director of Informatics)
- **Public weights trained on competition data are eligible for awards** [744056]. Winners must list the shared weights and their source datasets with the submission. Our base (goodpjw v21: Raptor, R6, R7, reader) is prize-eligible on this point.
- **External data** [743416]: "Datasets that require only registration and acknowledgement of licensing terms would clearly meet these criteria. Datasets that require an institutional research agreement or limit access to users with institutional affiliations clearly do not." No per-dataset list.
  - OAI through NDA: a participant reports the OAI permission group shows "Institutional Approval Required: No" and that individual sign-up with only an email via Login.gov was approved at once [743416]. An NDA help-desk reply says OAI is open to everyone [743416].
  - Still open: participants in China get Login.gov error LG22 [746792, 749145]. That is the "geo restrictions" case in Rule 2.6.b, and the host has not answered it.
  - **Revised risk for our final pick B (base, OAI-trained Raptor weights):** low for medals, moderate rather than "real" for prizes. Nothing to change in the plan.
- KneeCoT stays banned [734109]. DINOv3 weights need a gated licence from Meta and some applicants were refused [733313], so stay away.

### Leaderboard on 10-16 05:43 UTC (6,061 teams, measured)
| Public score | Ranks | Teams |
|---|---|---|
| 0.968 | 1 | 1 |
| 0.967 | 2–4 | 3 (Tucker's merged team #2; azamat1ch + matthewliew #3) |
| 0.963 | 17–24 | 8 |
| 0.960 | 51–71 | 21 |
| 0.958 | 105–181 | 77 |
| **0.957** | **182–584** | **403 (us: #477)** |
| 0.956 | 585–1391 | 807 |

- Medal lines at 6,061 teams: gold ≈ top 22, silver top 303, bronze top 606. **We are in bronze; the 0.957 block straddles the silver line.**
- The public split is ~390 studies; a single submission has SD ≈ 0.003 [748011]. Hidden test ≈ 1,322 studies [744230].
- The API and the CSV both give only 3 decimals (checked). **But rank inside a tie block follows the unrounded score** [748011, 748310]. One participant saw a blend keep its displayed 0.954 while rank went 437 → 320, then 233 → 156 at 0.956.

## Insights that change what we should do

### 1. Rank inside the 0.957 block is our paired test for the add-on (free)
Kaggle keeps each team's best score. If `knee-blend957-eff` beats base957 unrounded, our row jumps up inside the block. If it doesn't, nothing moves. Other teams joining the block can only push us down. So a jump of tens of places right after the blend scores means our members add signal; no jump means they don't, whatever the displayed score says.
- Baseline: **#477, block 182–584, at 10-16 05:43 UTC.** Re-download the leaderboard right after the blend scores (`python -m kaggle competitions leaderboard -c rsna-knee-abnormality-detection --download`).
- This answers "does weight 0.15 help" without a second submission, and later lets us compare final candidates.

### 2. Stronger and heavier pseudo-labels: the most-cited lever we have not used fully
Every strong solo report names image-model pseudo-labels, not better report extraction:
- Archit Konde (0.950 single fold): mixing out-of-fold predictions into the labels worked where masking and replacing did not; "50/50 already jumped, going heavier on the model added a bit more". It only works with true out-of-fold predictions [735304].
- Raymond Yuen (0.949): teacher models from several weak-label sources, 0.5/0.5 blend, "my main leverage"; different teachers help different findings [735304, 743148].
- colum2131 (ConvNeXt-T 0.953, DINOv2 0.951): image-model pseudo-labels blended at a fixed ratio. On gold, Effusion went 0.835 (LLM, report only) → 0.986 (image model on pseudo-labels) [745949, 735304].
- azamat1ch (team #3, 0.967; ConvNeXt-T 5-fold 0.959): "Short answer is labels … AI, not by hand. And don't rely on reports alone" [748295].
- Chris Deotte: with pseudo-labels "every single model can match our best ensemble CV score" [735304].

**Ours:** every member uses `pseudo=0.5` from `knee-train-rad`, the old 16-slice slot-head model at 0.877 OOF. Our best mean-OOF teacher is now eff320 5-fold + eff31-mil + rad31-mil at **0.897 OOF / 0.923 gold**. `load_targets` already averages several teacher kernels and merges split-fold kernels, so this needs **no code change**:

```
pseudo=0.7:knee-train-eff320,knee-train-eff320b,knee-train-eff31-mil,knee-train-rad31-mil
```

- Screen: eff31 at 224 px, folds 0–1, against eff31-mil (~0.7 GPU-h).
- Caveats:
  - The OOF score against v4b moves with the targets, so judge on gold as well.
  - The teacher's out-of-fold predictions on training studies come from models that saw the student's validation fold. That is mild leakage, the same as our current runs but a bit more with a stronger teacher.
  - EXPERIMENTS records "round-2 teachers: no gain". Those were eff2/rad2 on the prep-v2 cache, so that test was confounded and is not a clean negative.

### 3. We may be under-training
- Strong recipes train much longer than our 6 epochs:
  - Danial's CoAtNet (0.949) and the S6 ConvNeXt (0.950): 16 epochs each [746792, 749359].
  - Will: going from 25 to 50 epochs gave +0.004 public LB [740610].
  - Myo: ResNet at 0.94, 50 epochs with heavy augmentation and regularisation, best epoch 28 [743148].
- Our 16-epoch run (`knee-train`) overfit after epoch 5, but that was the old slot head on hard v4b targets with light augmentation. It predates MIL, pseudo-labels and the 288 px input.
- The `rot=` knob (±25°) has never been run.
- Screen arm: idea 2's targets plus `epochs=12 rot=25`, eff31 at 224, folds 0–1 (~1.4 GPU-h), in parallel with arm 2.

### 4. Lower the priority of the 110 mm zoom cache (eff320z)
- Ziad tested a centre crop of 140 → 110 mm: +0.0005, inside noise [737696].
- Cody found that 100 mm cut anatomy off and 140 mm "helped a ton"; Parag says the best field of view depends on the model [735304].
- The only zoom gains in public use crops **placed by an anatomy network** (S6: 100 mm via SKM-TEA V-Net anchors), plus the full field of view [749359]. Ours is centred on the image.
- Keep the centring check before spending any GPU on it. Rank it below ideas 2 and 3.

### 5. Already right in our pipeline (confirmed, don't touch)
- **RadImageNet with the centre slice ×3:** +0.020 over adjacent slices, a different input contract from ImageNet weights [737696]. RadImageNet also reads Synovitis better (0.78 vs 0.62–0.72 for ImageNet backbones) [737566]. Keep rad31-mil as a member.
- **Score every window at inference:** coverage climbs up to ~31 windows and then flattens [737696]. We infer on all 31.
- **Physical slice order by IPP projection:** worth +0.028 for a baseline that sorted by filename [745759].
- **Judge on report-label OOF:** r = 0.87 with the public LB vs r = 0.66 for gold over 15 submissions [748011].
- **fp16 only on T4:** one team lost 0.14 AUC to bf16 emulation hitting a time guard on the hidden test [744230].
- **Ours-first blend layout that never raises:** hidden reruns fail on things the 3 visible studies cannot show [749623, 749798, 742122]. Chris Deotte's advice is to run the infer path on ~1,300 train studies in a commit and guard against NaN and malformed DICOMs. Our own-only infer already passed the hidden test (LB 0.938), and the blend cells fall back to the base on any error.

## Confirmed dead ends (don't spend on these)
- **Better report extraction or label consensus.** Eight public label sets are indistinguishable on gold for 11/12 findings [747540]. Swapping the LLM is worth ~0.01 on gold [743148]. Tucker (now #2): simple Qwen labels, "label tweaks barely matter" [735304]. Our consensus and rubric runs agree.
- **Agreement masks:** Prateek reported a gain from dropping 25% of cells, but Archit and Tom saw nothing from masking, and our consensus mask gave no gain.
- **Bigger encoders:** DINOv2 S → B is a null at 224 px [735154, 738096].
- **DINOv3:** worse than v2 for several teams, and the licence is gated [733313, 740162].
- **Blend-weight tuning on the LB:** 13 submissions all read 0.937 [740439]. Per-label weights tuned on the public LB likely give points back on the private board [749570].
- **Laterality canonicalisation tweaks:** no measurable effect [737597].
- **Report-text aux losses or image → report generation:** no help [740162, 743148].
- **Vision-LLM labelling from images:** promising for Effusion (LLM 0.835 → ~0.92 on gold when given the MRI) [745949], but montages would have to be rendered on Kaggle and read for ~4,300 studies. Not feasible in 6 days.

## Facts worth keeping
- Gold labels are image reads by two MSK radiologists with a third adjudicating. The test set used the same process. **When report and image disagree, the image label wins** [733826, 733491].
- Report silence differs by finding: gold-positive rate when the report is silent is Synovitis 0.34, PF OA 0.21, Baker's 0.03, Medial OA 0.00 [733932]. Fracture is in 7% of reports but 31% of gold labels [741310]. Our `SILENT_FIX` already encodes this.
- There are 7 bilateral training studies. The host adjusted the report text or DICOM metadata so the scored knee can be identified [733826, 735639].
- Two training DICOMs have half-length PixelData; Kaggle tried to fix the dataset on 09-18 [737163, 741916].
- Both final picks count for the main track and the efficiency track [743356]. RuntimeSeconds covers the whole notebook run [733475].
- Efficiency tips if ever needed:
  - fp16 weights with half inputs and no autocast: 2.08× faster than autocast on a T4;
  - 16-thread DICOM reads;
  - parse PixelData straight from the file offset [743374].
