# Knee competition: rules, leaderboard, public code and model provenance (refresh 2026-10-17 16:00-16:30 UTC)

Scope: everything except the forum and the previous RSNA competition. Read-only towards Kaggle. Baseline: `reports/2026-10-16 public code insights.md`, `reports/2026-10-15 leaderboard research.md`, `research_notes/public_code_catalog.csv` (10-16). Raw data, scripts and outputs are in `research_notes/knee_2026-10-17/`; the refreshed catalog is `research_notes/public_code_catalog_2026-10-17.csv`.

## 0. Ten-line summary
1. **Nothing above 0.958 is public and no notebook has been created since 10-15 22:59 UTC** (listing of 1,333; 6 vanished, 3 renamed, 0 new). Only 7 notebooks were re-run after 10-17 06:40, none of them stronger.
2. The public board moved on: top score 0.969 (was 0.968), 29 teams at >= 0.963 (was 25), **the 0.958 block grew 446 -> 992 teams in 28 h** (ranks 142-992 now). We are **674 / 6,143** (533rd inside the block), about 60 places below the public bronze line (~614). Gold line is unchanged: ranks 21-29 = 0.963.
3. 846 of the 851 teams at 0.958 made their last submission on 10-16 or 10-17 (median 33 submissions): the block is a crowd of clones of two public families plus active private teams. 59 pulled notebooks (37 authors, 38 of them >= 0.956) form one near-identical clone cluster rooted in heliosli's code.
4. Only two public families reach 0.958: **A** = heliosli/goodpjw (R6 flip-TTA + nartaa Raptor 384/224, OAI) and **B** = lavinwins (20 DINOv2 + A5 + RadImageNet heads + dread Raptor + mattiaangeli CoAt family; no OAI, no R6). Our union of the two is genuinely unique.
5. **Noise dominates inside the block.** Test set = 1,322 studies (lavinwins' v5 log), public about 397. Simulated SD of a macro AUC: public 0.0031, private 0.0020; SD of the difference between two near-identical models: 0.0021 public, 0.0013 private. A unique 0.958 entry has roughly 1% gold / 20% silver / 39% bronze odds on private; a unique 0.961 entry 4-6% / 38-49% / 62-75%; 0.963: 10-16% gold.
6. **Rules:** 2 final submissions; ties go to the earlier submission (exact clones lose to earlier clones); external data must be "publicly available and equally accessible" or "reasonable" (the text names "geo restrictions"); the host (10-14) refuses to rule dataset by dataset and asks for disclosure. Prize winners must publish training code and weights, which no blend of third-party weights can do, so prizes are out of reach; medals follow rank.
7. **R7 / fastMRI must stay out.** The un-suffixed `heliosli/rsna-knee-r6-frozen-fe055406` has held R7 since 10-14 18:12. Open item: the R6 stacker metadata names its training run `cnx_r6fm` ("fm" could be flip-mirror or fastMRI; the manifest has no fm_aux entry; unverified).
8. New clean-licence component worth knowing: **nartaa/rsna-knee-s6fall-weights** (single ConvNeXt-S, 0.950 public alone, card says "No OAI data was used", competition data + SKM-TEA anchors, CC BY-NC 4.0, dataset v5 updated today 06:04 UTC). Public evidence for it next to R7 + Raptor: 0.957 vs 0.956.
9. Hidden runtime facts: base958 about 5 h 11 min (measured by us); lavinwins' hidden runtime is unknown (our test 57315208 pending); its own budget is 8 h.
10. New public weights appeared today with no cards: `neronrx/mycelis-rsna-knee-4fold-hive` (4-fold ConvNeXt 384, CC0, 10-17 08:53; its author sits at 0.958, last submission 09:18), `dkim54/rsna-knee-models-v11c..i` (CC BY-NC 4.0). No public notebook uses them; treat as undocumented.

## 1. Rules, evaluation, data, timeline (Kaggle MCP, pages saved as `page_*.md`)
| Item | What the page says | Implication for us |
|---|---|---|
| Metric | mean of 12 per-finding ROC AUCs | rank-based blends are safe; per-finding weights matter |
| Submissions | max 5 per day; "select up to two (2) Final Submissions"; team size <= 5; merger deadline passed (10-15 23:59 UTC) | pick two; if nothing is selected Kaggle falls back to the best two by public score (standard behaviour, verify in My Submissions) |
| Code requirements | notebook only, CPU and GPU <= 9 h, internet off, "freely & publicly available external data is allowed, including pre-trained models", file `submission.csv` | all components must be attached datasets/models; 9 h includes queueing as measured by us |
| Test set | "about 1300 studies"; hidden data replaces the 3-study example | 1,322 per lavinwins' log; public ~30% |
| Timeline | final submission 2026-10-22 23:59 UTC; winners' requirement deadline 2026-11-05 | last safe start for a 9 h notebook is well before 10-22 12:00 |
| External data (rule 2.6) | external data must be "publicly available and equally accessible ... at no cost" or satisfy the "Reasonableness" standard; costs and restrictions "(e.g., 'geo restrictions')" can exclude a resource; "acceptable unless specifically prohibited by the Host" | OAI is NDA-gated and reported blocked from some countries (forum 749145/743416); fastMRI needs a data-sharing agreement; both are grey. The weights themselves are free to every participant |
| Host guidance 743416 (10-14) | no dataset-by-dataset ruling; registration plus acknowledgement of terms "clearly" meets the criteria; an institutional research agreement or institution-only access "clearly" does not; prefers transparency about datasets used | disclose OAI, SKM-TEA and (if ever used) fastMRI; KneeCoT is banned |
| Winner licence | CC-BY-NC 4.0 on the winning submission and code; incompatible-licence input data or pretrained models need no licence grant (rule 2.5.a.4) | NC weights are tolerable inside a winner licence; nartaa's card explicitly licenses CC BY-NC use |
| Winner obligations | deliver training and inference code, weights as a public Kaggle dataset, documentation, video, forum post with code and weights | a blend of third-party weights without their trainers cannot comply: realistically no prize for public blends |
| Open source (rule 3.6) | public competition code is deemed licensed under an OSI-approved licence; only OSI-licensed open-source code may be used | Kaggle notebook default is Apache 2.0 (notebook licences are not exposed by the listing) |
| Private sharing | "Privately sharing code or data outside of Teams is not permitted"; sharing is fine if made available on the forums | only public notebooks may be reused |
| Tie-break | "the Submission that was entered first to the Competition will be the winner" (rule 3.7.b) | exact clones rank by submission time; our 10-16 12:49 base958 is behind every earlier clone on private too |
| Disqualification | sponsor may disqualify for "cheating, deception, or other unfair playing practices" (3.8.d); removal can withhold medals (3.8.e) | enforcement is discretionary; low but non-zero risk for OAI/fastMRI users |
| Efficiency track | $7k/$6k/$5k; eligible only if the submission is a selected final submission and beats the sample on private; score = AUC/(Benchmark - maxAUC) + RuntimeSeconds/32400 | not pursued |
| Prizes | $77,000: main ten places $9k, 7k, 6.5k, 6k, 5.5k, then 5k x5 | prize ranks need full reproducibility |
| Medal lines (Kaggle 1000+ team rule, 6,143 teams) | gold = top 10 + 0.2% (about 22), silver 5% (about 307), bronze 10% (about 614) | gold at 0.963+ today; silver and bronze both inside the 0.958 block |
| Data notes | 12 binary targets; only 58 labelled training studies, the rest via reports; transfer syntaxes include JPEG Lossless and JPEG 2000; prevalence differs between train, public and private | decoders (pylibjpeg wheels) are bundled by the public notebooks |

## 2. Leaderboard analysis (6,143 teams, downloaded 16:02 UTC; history from 5 saved snapshots in `lb_history/`)
### 2a. Blocks and lines (public, 3 d.p.; ranks inside a block follow the unrounded score, then submission time)
| Score | Ranks | Teams | | Score | Ranks | Teams |
|---|---|---|---|---|---|---|
| 0.969 | 1 | 1 | | 0.962 | 30-46 | 17 |
| 0.968 | 2-4 | 3 | | 0.961 | 47-64 | 18 |
| 0.967 | 5 | 1 | | 0.960 | 65-93 | 29 |
| 0.966 | 6-9 | 4 | | 0.959 | 94-141 | 48 |
| 0.965 | 10-12 | 3 | | **0.958** | **142-992** | **851** |
| 0.964 | 13-20 | 8 | | 0.957 | 993-1213 | 221 |
| **0.963** | **21-29** | **9** | | 0.956 | 1214-1664 | 451 |

Gold line (about rank 22) = 0.963 and sits inside the nine-team 0.963 block; silver (about 307) and bronze (about 614) are both 0.958, inside an 851-team block. Other big blocks: 0.943 (382 teams), 0.891 (147), 0.500 (182 empty submissions).

### 2b. Movement since 10-16 12:50 (teams at or above a threshold)
| Snapshot UTC | Teams | >= 0.963 | >= 0.960 | >= 0.959 | >= 0.958 | Rank 1 | Rank 22 | Ours |
|---|---|---|---|---|---|---|---|---|
| 10-16 12:50 | 6,085 | 25 | 74 | 112 | 446 | 0.968 | 0.963 | 707 |
| 10-16 18:01 | 6,100 | 25 | 78 | 123 | 585 | 0.968 | 0.963 | 339 |
| 10-17 07:46 | 6,123 | 28 | 89 | 135 | 855 | 0.969 | 0.963 | 553 |
| 10-17 14:41 | 6,140 | 29 | 92 | 138 | 978 | 0.969 | 0.963 | 662 |
| 10-17 16:02 | 6,143 | 29 | 93 | 141 | 992 | 0.969 | 0.963 | **674** |

The top 30 is stable: 27 of the 30 teams on 10-16 12:50 are still there (three entered at 0.963-0.964 with 73-122 submissions each). Rank 5 rose 0.966 -> 0.967 and rank 10 0.964 -> 0.965. The top moves about +0.001/day; the middle moves into the 0.958 block at about 20 teams/hour. Our rank falls as new clones of higher-scoring unrounded variants arrive, not because our score changed.

### 2c. Who is where (usernames from the LB file; team names not printed)
- Top: tuckerarrants' team #1 0.969 (4 members, 328 submissions), azamat1ch #2 0.968 (2 members), jiweiliu #12 0.965 (5), coolinlai #34 0.962 (5). 13 of the top 30 teams have 2-5 members; median submissions 141 (top 30) vs 107 (ranks 31-141) vs 33 (0.958 block) vs 4 (below 0.94).
- Public-code authors are inside the block, not above it: bliverrigpdoge #172, jenillangaliya #236, nartaa #244, mayankraaz #308, heliosli #330, goodpjw2008 #400, varshithakolupuri #408, lavinwins #666, matterhorn3838 #728, lingyu07 #820, dreaddevelopment #848, hengck23 #878, kiyoshiohno #885, cdeotte #952, pilkwang #953, neronrx #955. mattiaangeli #1256 (0.956), kirderf #1254 (0.956).
- The 0.958 block is overwhelmingly active clones: 846 / 851 last submitted on 10-16 or 10-17, 20% multi-member teams. Spearman(rank, last-submission time) inside the block is only -0.16, so order is mostly the unrounded score.
- Ours: rank 674, 0.958, 16 submissions, 533rd of 851 inside the block.

### 2d. How much of this is noise? (`noise_sim.py`, `shakeup_sim.py`; binormal model with our label prevalences, illustrative)
- Macro-AUC SD for one team at true 0.960: n = 390 -> 0.0031; n = 910 -> 0.0020. Two models whose errors correlate at 0.87 (public models do, per leoprovorov): SD of the difference 0.0021 (public) and 0.0013 (private). P(a model that is truly +0.001 better wins on public) = 0.71; +0.0025 -> 0.89.
- Empirical-Bayes shake-up from today's public distribution, unique (non-clone) entries, P(private rank within gold 22 / silver 307 / bronze 614), three parameter sets agree within a few points:
  | Public score | gold | silver | bronze |
  |---|---|---|---|
  | 0.957 | 0.2-0.6% | 13-14% | 29-31% |
  | 0.958 | 0.8-1.1% | 19-20% | 38-40% |
  | 0.959 | 1.2-1.8% | 26-29% | 44-52% |
  | 0.961 | 3.8-6.4% | 38-49% | 62-75% |
  | 0.963 | 10-16% | 58-78% | 76-91% |
  Exact clones are worse than these (ties go to the earliest submission). The numbers say a unique submission that shows >= 0.960 publicly is what changes the odds; a unique 0.958 changes them only slightly.

## 3. Public notebooks (listing refreshed 16:00 UTC; `kernels_web.json`)
### 3a. What changed
- 1,333 notebooks (692 with a public score). Creation dates stop at 10-15 22:59 UTC: public sharing has dried up.
- Re-runs or new versions after 10-17 06:40 UTC: lavinwins/top-1-gold (10:39; now a self-contained training notebook, no submission), kiyoshiohno/rsna-knee-template (08:49 version, 10:36 run, 826 s), goodpjw "does S6 help next to R6" (11:40 version, 14:17 run, 0.957), kaiwalyaatulraut (0.936), toghrulaliyev/rsna-knee-aze-team-dd6d24 (new, 2,898 s, DINOv2-large + Qwen2.5-7B, team LB 0.542: irrelevant, pulled to `ref/_new_2026-10-17c/`), plus two EDA notebooks.
- Renames (same notebooks): goodpjw `...-0-957` -> `...-0-958`, heliosli `...-0-9576` -> `...-0-9586`, goodpjw `coatnet-s6-...-0-954` -> `does-s6-help-next-to-r6-lb-0-957`. Six notebooks vanished (deleted or made private).
- Score changes: lavinwins/rsna-gold-medal 0.956 -> 0.958 (the slug now holds a different family; until 10-16 it was a heliosli R6/R7 + nartaa "Apex dual-reader" copy), lavinwins/top-1-gold 0.956 -> 0.958 (earlier version), kiyoshiohno template 0.957 -> 0.958, goodpjw/heliosli notebooks reached 0.958 on 10-16/10-17. Notebooks at 0.958: 6; at >= 0.957: 13; at >= 0.956: 57; at >= 0.950: 139.

### 3b. Ranked table (visible runtime = last run on the 3 example studies; hidden runtime is not published for any of them except our own measurements)
| # | Notebook (votes) | Public | Visible s | Version UTC | Families / what runs | External data | Weights licences | Clone status | Rule class |
|---|---|---|---|---|---|---|---|---|---|
| 1 | goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958 (307) | 0.958 | 495 | 10-16 12:48 | v27: R6 (rostigor hash-checked copy) + Raptor 384/224 mirror at heliosli's per-finding weights + reader at 1/10. v26 added R7 (0.4 of compact slot, same 0.958); R7 dataset still attached | OAI, SKM-TEA | nartaa other/NC, heliosli other, reader Apache-2.0 | own lineage (0.89 to hitarthjain0 S6 notebook) | yellow (R7 mounted) |
| 2 | heliosli/blend-mine-in-private-lb-luck-to-all-upvoters (171) | 0.958 | 496 | 10-15 17:03 | R6 `-w1` + Raptor 384 (0.65) + 224 (0.35) 50/50 rank blend, flip TTA; Raptor weight Synovitis .8, Lateral OA .2, PF OA .6, Baker's .35 | OAI, SKM-TEA | same | root of the 59-notebook clone cluster; 0.995 to picond's copy | yellow: **our base958** |
| 3 | heliosli/this-is-best-public-dataset-new-version-0-9586 (134) | 0.958 | 487 | 10-17 04:21 | V2 = #2 + retuned weights (Lat OA .15, PF OA .65, Synovitis .85, Baker's .30) + goodpjw reader 0.004-0.035; mounts the un-suffixed R6 slug (now R7) | OAI, SKM-TEA, reader labels | same | 0.87 to #2 | yellow; our base958v2 = same 0.958 |
| 4 | lavinwins/rsna-gold-medal (94) | 0.958 | 262 | 10-17 01:28 | family B: 20 DINOv2-S + A5 + RadImageNet heads (antoinegg1, prvsiyan V52) + dread Raptor x4 + mattiaangeli CoAt x4 (resgated, D4, Global96, repairv1) + pilkwang labels, "v6 combination layer, no new checkpoints"; 19 inputs; needs 2 x T4 | none declared; undocumented training data | CC0 (mattiaangeli, pilkwang, dread), CC BY-NC-SA 4.0 (antoinegg1 heads, prvsiyan, marwanmath), Apache-2.0 (DINOv2) | 0.88 to jiweiliu's 0.943 stack; unique among 0.958s | yellow (licence/undocumented, no OAI) |
| 5 | lavinwins/top-1-gold (46) | 0.958 badge | 66 | 10-17 10:39 | now a training notebook (competition data only, internet on) | none | n/a | unique | not a submission |
| 6 | kiyoshiohno/rsna-knee-template (45) | 0.958 | 826 | 10-17 08:49 | thin launcher that executes a versioned private logic dataset, with R7 + R6 `-w1` + nartaa + orthofoundation attached | opaque (R7 mounted) | n/a | unique | red/unknown |
| 7 | heliosli/this-one-is-old-version-0-9573-newone-you-can-use (179) | 0.957 | 440 | 10-14 18:13 | R6 flip + Raptor | OAI, SKM-TEA | same | in cluster | yellow |
| 8 | bliverrigpdoge/r7-raptor-knee-centred-convnext (112) | 0.957 | 497 | 10-14 08:26 | R7 + Raptor + private knee-centred KCD25 at 0.2 | OAI, fastMRI (R7), SKM-TEA | private KCD25 | 0.87 to velci-kcd25-20 | red |
| 9 | prvsiyan/the-bee-s-knees-final-rsna-push (108) | 0.957 | 496 | 10-15 20:44 | R6 (rostigor) + Raptor, 15,390 lines | OAI, SKM-TEA | same | own code | yellow |
| 10 | laimerbotrolenk/rsna-four-model-ensemble (25) | 0.957 | 534 | 10-15 14:12 | R7 30% + S6fall 20% + Raptor-384 TTA 32.5% + Raptor-224 17.5% (Aleksei's four-model ensemble, rank blend with an S6 correlation gate) | OAI, fastMRI (R7), SKM-TEA | same + nartaa S6fall CC BY-NC | own | red (R7) |
| 11 | leoprovorov/rsna-knee-0-957-explained (16) | 0.957 | 11 | 10-16 12:21 | analysis only: public models' error correlation 0.86-0.89 | n/a | n/a | n/a | n/a |
| 12 | goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957 (9) | 0.957 | 498 | 10-17 11:40 (v8 text, nothing scored) | v4: R6 + heliosli S6 (compact-v1 v13) + reader + Raptor = 0.957, i.e. **S6 next to R6 costs 0.001** (vs 0.958 without); v6/v7 retuned per-finding weights on the gold-58 and scored **0.957 / 0.955** vs heliosli's 0.958 | OAI, SKM-TEA | same | own | yellow |
| 13 | picond/rsna-knee-9576-r6-v1-fixed-readable (4) | 0.957 | 495 | 10-15 08:54 | exact-code copy (0.99) of #7 | OAI, SKM-TEA | same | clone | yellow |
| 14-57 | 44 more at 0.956 | 0.956 | 180-1031 | 10-11 to 10-15 | 38 of them are clones in the heliosli cluster (R7 or R6 + Raptor): evgendvorkin versia-7 (432 votes), heliosli 0-957-latest (224), degnonguidi, jenillangaliya, varshithakolupuri "one unused checkpoint" (adds nartaa's 224 checkpoint), mayankraaz, ...; others: rjings/rsna-versia-8 (family-B predecessor, 216 s), kupedata, tahmidtawsif, beraterolelk/kiza123123 (R7 + bliverrigpdoge leg-r9all) | mostly R7 = fastMRI | | | red for R7 |
| ref | nartaa/rsna-knee-0949-anatomical-mirror (94) | 0.949 | | | single CoAtNet 384 (OAI rows, JEV-blend labels), 96 slices, 94 windows, anatomical mirror | OAI | | | yellow |
| ref | nartaa/rsna-knee-s6fall-solo (22) | 0.950 | 157 | 10-12 | single ConvNeXt-S all-data on JEV-blend labels | none (SKM-TEA anchors) | CC BY-NC | | green/yellow |
| ref | dreaddevelopment Raptor single | 0.924-0.928 | | | CoAtNet 384, competition data only | none | CC0 | | green |

Clone structure (`copies.py`, line-set Jaccard on 129 pulled notebooks >= 0.94): one chain-connected cluster of 59 notebooks by 37 authors (38 at >= 0.956, root `heliosli/rsna-knee-blend-gold`), then 8 + 8 + 4 + 3 smaller clusters; 46 clusters in total. Notebook ordering by version date: copies of heliosli/evgendvorkin code are not independent submissions. Family signature of the 0.954-0.958 notebooks by attached datasets: R7 + Raptor 32 notebooks (max 0.957), R6 + Raptor 10 (max 0.958), Raptor + S6 14 (max 0.955), Raptor + S6 + reader 8 (0.954), four-model R7 + S6fall 1, family B 2-3.

## 4. Model and dataset provenance (cards fetched 10-17 with the Kaggle CLI/MCP; full table `dataset_provenance_2026-10-17.csv`, cards in `ds_meta/`)
| Component | Dataset | Licence on card | Training-data declaration | Flag |
|---|---|---|---|---|
| nartaa Raptor CoAtNet 384 + 224 (0.949 / 0.945 alone) | nartaa/rsna-knee-publication-swa-weights-20261007 (v1, 10-07) | other: research and educational use incl. the competition; adaptation and redistribution non-commercial, explicit CC BY-NC 4.0 winner licence | "Both models used the same 2,399 baseline OAI knees for masked external supervision"; OAI via NDA study 3407, acknowledgement text kept; weak-label sources stevenleehans, pilkwang, JEV | **OAI** (controlled access, NDA says open to all; some countries blocked per forum) |
| heliosli R6 ConvNeXt-S, 4 folds (bundle fe055406fad6a12f) | heliosli/rsna-knee-r6-frozen-fe055406-**w1**, rostigor/rsna-knee-r6-fe055406 (byte copies, manifest sha256 6508ce69...) | other | "No images, reports or labels inside"; forum table: no fm_aux entry; stacker metadata run dir `cnx_r6fm`; crop 352, mirror/flip-TTA | SKM-TEA anchors; fastMRI absent from manifest (fm = ? unverified) |
| heliosli **R7** (bundle 8c7b666e27dc0e8f) | heliosli/rsna-knee-compact-r7cnxfsss; varianceofx mirror ("Not for redistribution"); **un-suffixed r6-frozen v2 since 10-14 18:12** | other | fastMRI+ boxes as auxiliary loss, weight 0.3 (manifest and heliosli notebook) | **never use** |
| heliosli S6 (bundle 209a32c8b50a6b1c) | heliosli/rsna-knee-compact-v1 (13 versions; v13 restores v10), `compact-r6cnxfsss` and its mirrors (goodjane's copy claims Apache-2.0) | other | "fine-tuned on report-derived targets"; nothing external found | SKM-TEA anchors |
| anatomy.pt + code | heliosli/rsna-orthofoundation-fold0-epoch7-20261004 (43,640 downloads) | other: "Upstream weights retain their original licensing; no new open-source license is granted" | "unchanged SKM-TEA/KneeXNet auxiliary weights" | **SKM-TEA** (needed by all R6/R7/S6 pipelines); the OAI-derived ViT inside is not loaded by those notebooks |
| nartaa S6fall (0.950 alone) | nartaa/rsna-knee-s6fall-weights (v5, 10-17 06:04) | CC BY-NC 4.0 | ConvNeXt-S retrained from timm ImageNet-22k on JEV-blend report labels, 4,263 studies, gold-58 excluded; "No OAI data was used"; trainer, labels script and SHA256SUMS included | SKM-TEA anchors only; the cleanest strong component |
| goodpjw 2.5D ConvNeXt-T reader | goodpjw2008/rsna-knee-2-5d-convnext-reader | Apache-2.0 | train.py and make_labels.py read: ImageNet init, labels = mean of four public LLM label tables, gold-58 unused | clean |
| dread Raptor (0.924-0.928 single) | dreaddevelopment/raptor-knee-widedense, -maxspan, -native384, -native384dense, -arms, -arms-x | CC0 | CoAtNet-rmlp-2 384, 62-64 windows, 140 mm crop; no external data stated | clean |
| mattiaangeli CoAt family and DINOv2 folds; pilkwang weights | mattiaangeli/rsna-knee-coat-resgated-ep10-top3, ...-d4-depthzone-swa3-b2, ...-global96-top3, knee-mri-fold-weights; pilkwang/rsna-knee-weights | CC0 | none (no descriptions) | undocumented training data |
| RadImageNet heads | antoinegg1/e11-diverse-heads-v20, e9-...-v15; prvsiyan/v52-...-20260812; encoder copies marwanmath (CC BY-NC-SA), shigechan (card: original is MIT) | CC BY-NC-SA 4.0 on the heads | prvsiyan README: trained only on the competition split; antoinegg1: no card | NC share-alike, RadImageNet medical-image pretraining |
| DINOv2 | Kaggle Model metaresearch/dinov2 small/base/large/giant | Apache 2.0 | LVD-142M (facebookresearch/dinov2) | clean |
| LLM labels | pilkwang/rsna-knee-llm-labels | CC0 | twelve verdict/severity pairs per report | host: LLM-API labels allowed |
| sofiaanjenje E11/E13 (kernel outputs) | `/code/sofiaanjenje/rsna-knee-e11-train`, `e13-train` | n/a | DINOv2 heads on competition data (log read 10-16) | competition data |
| bliverrigpdoge KCD25 / leg-r9all | bliverrigpdoge/rsna-knee-leg-r9all (CC0, no card); KCD25 itself is a private source | CC0 | none | undocumented; KCD25 unusable |
| New 10-17 | neronrx/mycelis-rsna-knee-4fold-hive (CC0, ConvNeXt 384 x4, 839 MB), dkim54/rsna-knee-models-v11c..i (CC BY-NC 4.0), riadmohamed42/rsna-knee-dino-5fold-weights (CC0), abhiiiish ensemble (licence unknown) | | none (empty descriptions) | undocumented, no public notebook uses them |

## 5. Implications for our final picks
### 5a. Rule safety by component
- **Green** (competition data only, permissive licence): dread Raptor, goodpjw reader (Apache-2.0), nartaa S6fall (SKM-TEA anchors via anatomy.pt are the only external element), DINOv2, MIT RadImageNet.
- **Yellow, disclose**: R6 `-w1` and S6 (SKM-TEA anchors; R6 has no fm_aux in its manifest, but check the `cnx_r6fm` naming with heliosli or the forum before the final write-up), nartaa Raptor (OAI under NDA; the weights license is explicit; host asked for disclosure, not a ban), NC share-alike heads and undocumented CC0 weights in family B (lavinwins).
- **Red**: R7 anywhere (also silently through the un-suffixed R6 slug), isotriage JEV-OAI arm, KneeCoT, any kernel whose mounted dataset you did not hash-check. goodpjw's v26 and kiyoshiohno's template mount R7; goodpjw v27's text says R6-only.
- Disclosure list for the A-family pick: OAI (nartaa Raptor), SKM-TEA (anatomy.pt). Family B adds: CC BY-NC-SA 4.0 heads (antoinegg1, prvsiyan), undocumented mattiaangeli / pilkwang weights, RadImageNet, DINOv2.
- Prize path is closed for any blend of third-party weights (rule 2.8 needs training code plus public weights); the fight is for medals, which depend on rank and on not being disqualified (3.8.d-e). A fully OAI-free hedge exists: family B alone, or dread Raptor in place of nartaa's Raptor inside an R6 pipeline (cost: single-model strength 0.949 -> 0.928).

### 5b. Uniqueness vs exact copies
- 851 teams share the 0.958 block, mostly active clones of two public notebooks. An exact copy is tie-broken by submission time (rule 3.7.b), so it sits behind earlier clones on private as it does on public. Our base958 (10-16 12:49) and base958v2 are both exact-code runs: they are not unique.
- Unique but not better: two near-identical 0.958 blends are each an independent draw with SD about 0.0013 on private. Being unique helps only when the mean is also higher; the shake-up table in 2d says gold odds jump only from public >= 0.960-0.961 upward.
- The union (A first or base-first, B unchanged, 50/50 rank mix) is a different prediction vector from every clone, uses two public families with different backbones (ConvNeXt-S R6 + CoAtNet Raptor vs DINOv2 + CoAt family) and fits the 9 h budget only if lavinwins' hidden runtime L stays below about 3.1 h (HANDOFF gate: 5.2 h + L + 0.3 h < 8.6 h; base958 5 h 11 min measured). Decide on the measured L of 57315208 and on the union's own public score: >= 0.960 makes it pick A; 0.958-0.959 makes it a coin flip with the clones and only a better median than base958.
- **Pick B should differ in composition, not just in weights**, otherwise the two picks are one draw (correlation 0.9+). Candidates in order of independence: family B alone (L pending), a union variant with a third component, base958v2 (same family as A, not independent).

### 5c. Public components that could form a stronger unique blend inside 9 h
| Candidate | Evidence | Cost / risk |
|---|---|---|
| **A + B union** (in flight) | two different 0.958 families; public error correlation inside a family 0.86-0.89 caps single-family gains near 0.960 | needs L < about 3.1 h; fallback to A if B overruns (the build already does this) |
| **+ nartaa S6fall (20% rank weight, OAI-free, 0.950 alone)** | Aleksei's four-model notebook: R7 30 + S6fall 20 + Raptor 32.5/17.5 scored 0.957 vs R7 + Raptor 0.956; but goodpjw measured heliosli's S6 next to R6 as 0.957 vs 0.958 (a loss of 0.001), so S6fall next to R6 is an untested coin flip | one extra compact forward pass on the shared decode (the 4-model notebook shows 534 s visible vs 440-500 s for the 2-model R6/R7 + Raptor notebooks; hidden cost not published), wiring example exists (laimerbotrolenk); must keep R6, not R7 |
| Dread Raptor widedense/maxspan as a third CoAt reader | CC0, competition data only; already inside family B | little diversity next to nartaa Raptor |
| goodpjw reader at 1/10 | base958v2 = 0.958, no gain | free, no value |
| KCD25 add-on | +0.001 at 0.2 twice (bliverrigpdoge) | private dataset, unusable |
| Per-finding reweighting of A vs B | cheap | goodpjw's gold-58-tuned variants lost to heliosli's pre-fixed weights (0.957 / 0.955 vs 0.958) and the gold-58 gaps (18-35 positives per finding) are noise; weights must be fixed before any run |
| New undocumented weights (mycelis, dkim54) | unknown strength, no card, no notebook | licence/provenance risk, no evidence; skip |
Own models (0.9078 OOF, gold-58 0.934) lose at 0.05 weight twice; do not retry.

### 5d. Practical checks before 10-22
- Slug hygiene: un-suffixed R6 holds R7; use `-w1` or rostigor's copy with the bundle_id assertion. Dataset owners can still delete or alter files before our final runs; sources are read at submit time, so re-verify the day we submit.
- If nothing is selected on Kaggle, the best two public scores are used: set the selection explicitly.
- Public-LB differences under about 0.003 are noise (2d); judge by rank inside the block only for the same family.
- Watch list for the next two days: lavinwins/rsna-gold-medal (reruns and renames), heliosli (R6/R7 dataset swaps), nartaa S6fall dataset (v5 today), neronrx/dkim54 weights, any new notebook above 0.958 (none since 10-15).

## 6. Not verified
- Notebook code licences (the listing does not show them; Kaggle default is Apache-2.0).
- Hidden runtimes of every notebook except base958 and our own runs; the lavinwins hidden L is measured only by our pending submission 57315208.
- Whether R6's `cnx_r6fm` run name means flip-mirror or fastMRI; the forum table (749858) reads the manifest only. The R6 trainer source shipped in heliosli's notebooks (`ref/_heliosli_compact_src/`) has fm-aux off by default.
- Copy detection is by code similarity of the current versions (129 pulled notebooks >= 0.94), not Kaggle's fork graph.
- Shake-up numbers are a model, not a forecast.

## 7. Files
`research_notes/public_code_catalog_2026-10-17.csv` (24 columns: the 11 old ones plus author, version/run UTC, medal, author LB rank and score, external_data, copy_of, copy_sim, clone_cluster_root and size), `research_notes/knee_2026-10-17/` (`page_*.md` rules and pages, `lb_history/` + `lb_1601/` leaderboards, `lb_analysis.py` + `lb_analysis_output.txt`, `noise_sim.py`, `shakeup_sim.py` and outputs, `kernels_web.json`, `fetch_kernels.py`, `catalog2.py`, `copies.py`/`copies.csv`, `ds_meta/` cards, `dataset_provenance_2026-10-17.csv`, `provenance.py`, `goodpjw_reader_src/` two code files), pulled notebook `ref/_new_2026-10-17c/toghrulaliyev_rsna-knee-aze-team-dd6d24/`.
