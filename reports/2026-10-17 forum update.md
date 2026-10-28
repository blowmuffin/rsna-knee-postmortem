# Forum update, 2026-10-17 ~16:30 UTC

Scope: the competition discussion forum (all 203 topics, comments included) plus every host statement I could find (rules page, competition pages, dataset-level discussions). Read-only; nothing was posted or voted.

Method and coverage (so you know how far to trust it):
- Topic ids: Kaggle MCP `list_competition_topics` (203 topics now vs 197 in the 10-16 dump). Page 1 of the "Recent" and "New" listings was re-run at the end; nothing newer than topic 750260.
- Full re-fetch (topic + nested comments, kagglesdk) of 120 topics, which includes every topic whose last comment is later than the 10-16 06:00 dump, plus all external-data and host-reply topics. Text-diffed against the 10-16 dump: 6 new topics, 8 topics with new comments, the rest identical (host posts were not edited). The remaining ~80 topics were not re-fetched because the MCP listing shows no comment after 10-16 06:00 (all of them are older than 10-07 in "Recent" order); they are carried over from the 10-16 dump and marked so in `index.tsv`. The API returned 429 for long stretches (three agents share it), which is why I stopped the bulk re-fetch.
- Dataset-level discussions: checked ~25 key public datasets; only 5 have any discussion (listed in section 5), and only one has a reply.
- Raw files: `research_notes/forum_2026-10-17/` (new/updated topics, `index.tsv` for all 203, `dataset_discussions/`, `host_statements_all_to_2026-10-17.txt` = every host/Kaggle-staff comment, `rules_page_2026-10-17.md`).
- Competition page now shows 6,144 teams; our rank 675 (MCP `get_competition`).

Host/Kaggle people (handles used in the tags): Po-Hao "Howard" Chen (@javaduke95, lead organiser), Chris Carr (@cdcarr, RSNA Director of Informatics), Michelle Riopel (@michelleriopel, eligibility), Ryan Holbrook (@ryanholbrook, Kaggle, competition admin), @macruzbar and @shosys (tagged in every ask, never seen replying).

**Headline: no host or Kaggle person has posted anything since 10-14 13:20 UTC (Carr's two replies). Five new, specific questions (10-17) and 13 older asks are unanswered. Assume the 10-14 texts are the last word.**

---

## 1. Host rulings (everything the hosts have actually said)

Status: A = answered, P = partly (general rule, no yes/no on the specific thing), U = unanswered.

| Topic | Date | Who | Ruling in short | Status |
|---|---|---|---|---|
| 733343, 733826, 733491 | 08-06..08 | Chen | Gold labels = image review by two MSK radiologists plus a third adjudicator, same process for the test set. If report and image disagree, the image label is authoritative. Bilateral studies were reviewed one by one and the report text/DICOM adjusted so the scored knee is identifiable. | A |
| 733592, 734118 | 08-07, 08-10 | Chen | Reports are never available in the hidden test, before or after submission. | A |
| 733475 | 08-07 | Holbrook | GPU notebooks are eligible for the efficiency track. RuntimeSeconds = start to end of notebook execution, including package installs, model loading and DICOM reads. | A |
| 743356 | 09-25 | Deotte (participant, not host) | Both final picks count for the main track and the efficiency track. Not a host post; consistent with the Efficiency page text. | (A, secondhand) |
| 733965 (pinned) | 08-09 | Chen | Commercial LLM APIs may read the reports to make labels; this is not "private sharing". The service must be reasonably accessible and low cost; host reserves the right to judge. | A |
| 733965 comment | 08-27 | Chen | Non-commercial dataset licences alone do not exclude a dataset. Click-through registration generally satisfies "reasonably accessible"; institution-specific approval, negotiated agreements, IRB or long credentialing may be a barrier. Teams are responsible for the dataset's own terms (host cannot interpret or waive them). | A (general) |
| 734109 | 08-25 | Chen | KneeCoT is not allowed (needs a formal agreement with the source institution). May be revisited if HF terms change. | A |
| 741819 | 09-21 | Chen | OAI does not violate the rules if it is available to everyone with no institutional or legal-team sign-off and is free. Host cannot speak for OAI's own terms. | A, conditional |
| 743416 | 10-14 | Carr | No per-dataset list will be given. Rule 2.6.a applies ("publicly available and equally accessible"). Registration plus acknowledgement of licence terms clearly passes; an institutional research agreement or access limited to affiliated users clearly fails. Also apply "reasonableness" and disclose datasets and methods; the host would prefer winners to be separated by ingenuity, not privileged resources. | P |
| 744056 | 10-14 | Carr | Submissions that include publicly shared weights are eligible for awards. Participants should include the shared weights and their source datasets in the resource list delivered with the submission. Question was about weights trained on competition data. | A for competition-data weights; **U for OAI/fastMRI/other-external-data weights** |
| 748181 | 10-12 | Holbrook | A notebook-publishing lock applies one week before the deadline (so from ~10-15). Verbatim copies with scrubbed credit should be reported. Low-effort blends "usually" drop on the private board. A participant notes the lock covers new publishing, not updates to existing notebooks, in another competition. | A |
| 741916 | 09-18 | Holbrook | An attempt to replace corrupted DICOMs in the dataset "may have failed"; no later word. | P |
| 733797, 733971, 737350 | 08 | Riopel | Minors 13-18 need two Kaggle guardian-consent forms. | A |

Rules page (re-read 10-17, unchanged): 2 final submissions; 5 submissions/day; winner licence CC BY-NC 4.0, with a carve-out (2.5.a.4) letting incompatible-licence input data/pretrained models keep their own licence; 2.6.a/b external data and "geo restrictions"; 2.8 winner obligations = training code, inference code, weights as a public Kaggle dataset; 3.7.b ties go to the earlier submission; 3.9.b the sponsor may disqualify a prize submission or give the winner one week to remediate (licence conflicts, removal of violating software). The review wording is about prizes. Timeline: deadline 10-22 23:59 UTC; winners' materials due 11-05. Test set "about 1300 studies"; prevalence not guaranteed equal across train/public/private.

### Open questions, still unanswered (none has a host or Kaggle reply)
| Topic | Posted | Question | Votes |
|---|---|---|---|
| 743416 follow-ups | 10-14, 10-16 | Does a dataset whose provider blocks an eligible country (LG22 / NIH countries of concern) count as equally accessible? Same question for weights trained on it. | in a 98-vote thread |
| 749145 | 10-14 | Is OAI allowed given the NIH geo restriction; checked for medals or only prizes; derived pseudo-labels covered? | 15 |
| 748276 | 10-11 | Same geo question. | 15 |
| 747455 | 10-08 | Are public weights trained on OAI allowed (nartaa specifically named 10-13)? | 10 |
| 733965 comment | 10-13 | Can nartaa's and heliosli's orthofoundation weights be used in a prize-eligible final submission? | -- |
| **750236** | 10-17 | Will external-data compliance be checked for medal positions or only prizes; will medals be withdrawn; can answers be pinned before 10-22? | -4 |
| **750257** | 10-17 | nartaa's OAI-trained weights in a final submission? | -19 |
| **750258** | 10-17 | heliosli's SKM-TEA/KneeXNet `anatomy.pt` allowed? (no licence) | -20 |
| **750259** | 10-17 | R7 / fastMRI allowed? | -20 |
| **750260** | 10-17 | RadImageNet weights allowed (images only on request; MIT vs CC BY-NC-SA)? | -20 |
| 735121 | 08-14 | RadImageNet CC BY-NC-SA vs winner open-licence obligation. | 7 |
| 743420 | 09-26 | Does 2.5.a.4 also exempt weights trained on no-redistribution data (fastMRI, MRNet, KneeMRI) from the public-dataset duty in 2.8? | 4 |
| 739905 | 09-07 | CC BY-NC 4.0 vs "no limit on commercial use" in the winner licence; Apache-2.0 base weights. | 1 |
| 737950 | 08-28 | Does the KneeCoT logic extend to OAI/MRNet (DUA behind an institution)? | 5 |
| 744519 | 09-30 | Are all three planes and a fluid-sensitive series present in every hidden study? | 5 |
| 734681 | 08-12 | Is the private split stratified by site or are sites held out? | 8 |
| 740094 | 09-08 | Single-class label columns / corrupted DICOMs in the test set. | 0 |
| 748925 | 10-13 | Soft labels from a non-public proprietary labeller. | -1 |
| 733864 | 08-08 | How labels were produced (answered elsewhere by 733826). | 3 |

(The four "Karan Bhatti" asks 750257-750260 were posted within 3 minutes of each other and tag all six host/Kaggle accounts. All are heavily downvoted, which says how many people depend on these components.)

---

## 2. External-data status of each widely used component, and the risk for a final pick

The host test (Carr 10-14, Chen 09-21/08-27) in practice: free + registration/click-through = OK; institutional sign-off, affiliation-limited access, negotiated agreement = not OK. Enforcement: nothing announced for medals; Rule 3.9.b is a prize-time check. Precedent cited on the forum is the 2020 Deepfake Detection Challenge, where two prize-place teams were removed for unapproved external data (prize places only) [743416].

| Component (who) | External data / provenance (source) | Passes Carr's test? | Host ruling | Residual risk |
|---|---|---|---|---|
| **nartaa Raptor CoAtNet 384/224** (`nartaa/rsna-knee-publication-swa-weights-20261007`; in base958, heliosli 0.958, goodpjw v21 and ~70% of top public blends) | Retrained dreaddevelopment's Raptor recipe on competition labels plus **2,399 OAI knees** (NDA Study 3407, MOAKS reads), supervising **only PF OA, Lateral OA, Synovitis** (other cells masked); 3DReasonKnee (CC BY 4.0) used to match MOAKS rows to MRI series [746792, 749858, 750180]. Author's own recipe table: +0.005 public from adding OAI (0.940→0.945), not a controlled ablation. Epoch averaging was ranked on Gold-58. Weights released Apache-2.0; **training code now public (Apache-2.0, 10-17)**, OAI linkage manifest only via NDA [750180]. | Letter: yes for individuals (reports of immediate approval, group shows "Institutional Approval Required: No"; NDA/UMass replies say open to all) [743416, 746792]. **Geo hole:** NIH bars "countries of concern" incl. China from NDA data; Login.gov rejects Chinese registrations (LG22); one user says a download worked from a China IP, another says it failed; the NDA help-desk says the restriction does not apply to OAI [743416, 746792, 749145]. China, Russia and Venezuela are eligible countries (Rule 3.1). | Conditional yes (09-21). No answer on geo or on OAI-trained weights; Carr's 744056 reply is scoped to competition-data weights. | Medium-low for a medal, medium for a prize. Three of the 12 findings carry the OAI signal, so removing it would cost most on those three (base958 gives Raptor 0.8/0.6/0.2 weight on Synovitis/PF OA/Lateral OA). Must be disclosed. |
| **heliosli `anatomy.pt`** (`heliosli/rsna-orthofoundation-fold0-epoch7-20261004`; places the crops for R6/R7/S6; in ~70% of top public notebooks) | Dataset page: "unchanged SKM-TEA/KneeXNet auxiliary weights", "no new open-source license is granted". SKM-TEA = Stanford AIMI registration + data-use terms. KneeXNet = built from OAI images, no repo licence found [749858, 750258]. The ViT-L OrthoFoundation weights in the same dataset (OAI-pretrained) are not loaded by R6/R7/S6 [749858]. | SKM-TEA: letter yes. KneeXNet: public, but OAI-derived and unlicensed. | None. | Medium-low. **Present in every heliosli-based stack, including base958**, so an "OAI-free" claim for base958 would be wrong even without nartaa. |
| **heliosli R6** (bundle `fe055406`, our `-w1` copy) | Manifest has no fastMRI auxiliary entry; competition data + ImageNet (timm) [749858]. | Yes. | None. | Low, apart from anatomy.pt above. The compact-v1 and R7 dataset pages list the licence as "Other" and the requests for terms are unanswered [748406, 749213]. |
| **heliosli R7** (`rsna-knee-compact-r7cnxfsss`; now also behind `...r6-frozen-fe055406` without suffix) | fastMRI+ boxes as an auxiliary task (`fm_aux`, weight 0.3) [749858, 750259]. fastMRI images need an application stating affiliation and intended use, plus a data-sharing agreement that restricts redistribution of derived files [743420]. | **Weakest of the lot** (affiliation + intended use; closest to "limits access to users with institutional affiliations"). | None. | Medium-high. Keep avoiding R7 (already a rule in CLAUDE.md). |
| **goodpjw reader** (`goodpjw2008/rsna-knee-2-5d-convnext-reader`; in base958v2 and goodpjw v21+) | Frozen fastMRI+ ViTDet heads per edteoh [749858]. | As fastMRI. | None. | Adds a fastMRI exposure to base958v2 for no measured gain (v2 = 0.958, no rank gain). |
| **dreaddevelopment Raptor** (`raptor-knee-maxspan`, `native384`, `native384dense`) | Author on 09-04: all checkpoints use the same competition labels, "nothing external", ImageNet via timm [737696]. He also holds back a 0.945 set. | Yes. | n/a | Low. Clean OAI-free route for the Raptor family. |
| **mattiaangeli weights** (`coatnet-d4-depthzone-swa3-b2` CC0, `coat-resgated`, `global96`, `knee-mri-fold-weights`, DINOv2 members) | Described by another user as trained by Mattia on the competition's data [744056]; Mattia says only he and dreaddevelopment released genuinely new models [742050]. How the fold weights were made: asked 08-30, no reply [738202]. | Presumably yes (competition data + DINOv2/timm). | n/a | Low-medium (undocumented training data, no statement of any external set). |
| **antoinegg1 RadImageNet heads** (e9 v15, e11 v20) and **prvsiyan / marwanmath RadImageNet** | RadImageNet ResNet-50 weights are a public Google-Drive download; training images only on request; licence stated both as MIT (repo) and CC BY-NC-SA (735121). Heads: CC BY-NC-SA 4.0; provenance receipt requested 09-05, no reply [739633]. | Weights: free and anonymous, so the "equally accessible" test is easy; images-on-request is the only hook. | None (735121 and 750260 unanswered). | Low for ranking (about 60% of notebooks above 0.90 use RadImageNet, including our own rad31 models), medium for a prize (licence). |
| **DINOv2** (Meta) | Meta self-supervised pretraining, Apache-2.0. DINOv3 is gated and several applicants were refused [733313]. | Yes (v2). | n/a | Low. |
| **LLM report labels** (stevenleehans, pilkwang, JEV, ours) | Commercial and local LLMs allowed (Chen 08-09, 08-27). | Yes. | A | None. |
| **KneeCoT** | Banned [734109]. | No. | A | Do not touch. |
| **MRNet / KneeMRI** | Research-use agreement / CC BY-NC-ND; derivative-works clause vs the duty to publish weights [737950, 743420]. | Doubtful. | U | Not in any known top public stack. |

Cross-check on the lavinwins family, because it is our proposed OAI-free hedge: the current `lavinwins/rsna-gold-medal` source (`baseline/lavin958`) mounts dreaddevelopment, mattiaangeli, antoinegg1, prvsiyan, pilkwang and marwanmath datasets and two `sofiaanjenje/rsna-knee-e11-train`/`e13-train` kernels, with DINOv2 base/small; it mounts no nartaa or heliosli dataset. But `research_notes/public_code_catalog.csv` (10-16) tags the same ref as `helios-R6; raptor-coatnet-nartaa(OAI)`, which is probably an older version, and the member families match the 20-member ensemble by Mattia Angeli described in [749570] (so the lineage is Mattia's bundle plus tweaks; I could not confirm what lavinwins changed). **Treat any "OAI-free" statement as valid only for the exact version pulled in `baseline/lavin958`; the `sofiaanjenje` kernels' training data is not documented anywhere on the forum.** The forum account "Lavin Wins" has made no provenance statement in any thread (its comments are short and several sit at -7 to -11 votes).

Disclosure list the host asks for (744056, 743416) per candidate:
- base958 (heliosli blend-mine): nartaa weights (OAI via NDA 3407), heliosli anatomy.pt (SKM-TEA / KneeXNet-derived), heliosli R6.
- base958v2: the above plus goodpjw reader (fastMRI+ heads).
- lavinwins: mattiaangeli, dreaddevelopment, antoinegg1 (CC BY-NC-SA), prvsiyan, pilkwang, marwanmath (RadImageNet), sofiaanjenje kernels, DINOv2.
- A union carries the union of both lists.

---

## 3. Hidden-test, runtime and failure lessons

- Test set: about 1,300 studies (one post computes 1,322). The public board uses ~30% (≈390 studies, SD ≈ 0.003 for one submission) and the final private result the other ~70% [748011, 749570]. Prevalence is not guaranteed equal across splits; the host never answered whether the split is by site [734681].
- Scoring happens at submission time on the whole hidden set, so a hidden-set exception shows up as "Notebook Threw Exception" with no traceback. Nobody has a diagnosis route from Kaggle. Practitioner advice (Deotte, 09-27 and 10-15): run the exact submit path on ~1,300 train studies in a commit run; catch NaN predictions and malformed DICOMs [743779, 749623]. One user's 1,300-study train dry run took ~22 minutes and still failed hidden [749798]; another failed hidden at version 4 after a clean 100-study train dry run [745826].
- bf16 on T4/P100 is emulated (~50× slower): one team hit its own 8 h time guard after ~620 of 1,322 studies and wrote 0.5 for the rest (0.851 → 0.710). Fix = fp16; log per-study seconds and the number of studies that hit a fallback [744230]. (We are fp16 only.)
- 9 h limit covers only inference; a 20+-model ensemble can use all of it (Dread Development: ~9 h at first, 29 min after optimisation) [740474, 737696]. RuntimeSeconds for the efficiency track = the whole notebook, installs and loading included (Holbrook).
- Kaggle-side flakiness: dataset mount errors (ERRORED_MOUNTING_DATASET) for days around 10-07, resubmitting did not burn an attempt [746542]; a submission stuck "running" for >24 h, with a reply on 10-14 asking whether it was fixed [748403]; queues of ≥1 h on 10-12 [748575]. Budget for queue time inside the 9 h only if your design is time-sensitive (our union uses submit→COMPLETE as runtime).
- Corrupted training DICOMs (two files with truncated PixelData) were never confirmed fixed; the host did not say whether the test set is clean [737163, 741916, 740094]. Also unanswered: whether every test study has all three planes [744519].
- Dataset versions can change under you: heliosli withdrew then restored `rsna-knee-compact-v1` (10-10/11), and the name `r6-frozen-fe055406` now holds R7 [744056, 748406, 749858]. Always pin by `-w1` copy and assert the bundle id (we do).
- Tie rule: earlier submission wins (Rule 3.7.b). Exact copies share one private score.

---

## 4. Top-team technical insights (what moved scores)

Teams at the top, from their own posts:
- **Tucker Arrants (merged team #2, 0.967 on 10-16)**: simple Qwen report labels, "label tweaks barely matter"; ResNet/EfficientNet at 224/288 px; judges only on report-label OOF; treats <0.003 CV as noise; the image "student" beats the label "teacher"; ~25 min preprocessing+inference with TTA [735304, 740610, 745214].
- **azamat1ch (team #3, 0.967)**: single architectures 5-fold, ConvNeXt-T 0.959, EffNetV2-S 0.956, CoAtNet 0.952; "labels, AI not by hand, don't rely on reports alone"; states **no external datasets, only competition data** (10-12) [735304, 748295]. Another poster (diet1236364) gives the same advice and azamat1ch confirms it.
- **datnt114**: public 0.961 with OOF 0.937, gold-58 0.938 [748011].
- **Chris Deotte**: pseudo-labels let "every single model" match his best ensemble CV; ran one L40S agent loop to a 0.945 ensemble without public notebooks; "trust your CV" [735304, 743303, 744495].
- Pseudo-labels from image models blended with LLM labels at a fixed ratio is the most repeated lever (Archit Konde, Raymond Yuen, colum2131) [735304].
- Public-LB trick risk: 1−p submissions hide true scores, so some strong teams may sit lower than they are [748011].

Technique evidence, new or sharpened:
- nartaa 0.949 recipe: CoAtNet 384, 94 windows, native 320 crop, anatomical-mirror TTA (+0.001), OAI +0.005, JEV-blend labels +0.002; dead ends: window width flip on an older checkpoint, crop 288/304 (-0.001/-0.002), fresh crop training (-0.003) [746792]. S6 ConvNeXt-S retrained on those labels: 0.950 all-data, 0.948 folds 1-4; fold-0 OOF 0.907 beat CoAtNet's 0.890 locally yet lost 0.001 publicly (noise) [749359]. Their S6 final weights are epoch 16 with no SWA and no gold selection; only the CoAtNets used Gold-58 to rank epochs [750180].
- Ensembling a 224 px / 0.8-crop Raptor at 35% beside the 384 one kept the displayed score but moved rank 437→320 and 233→156 [746792, 748011]; this is the "rank inside a tie block" measure.
- Public/CV correlation: weak-label OOF r = 0.87 with public LB, gold-58 r = 0.66 [748011].
- Label work: eight public label sets indistinguishable on gold for 11/12 findings [747540]; a tuned LLM choice is worth ~0.01 on gold at most [743148].
- Dread Development: raising eval windows from 48 to 72 moved his board to 0.944 for +8 min and then stopped paying; extra slices dilute the head's softmax once the whole span is covered [737696]. In another team's pipeline swapping RadImageNet for ImageNet at identical architecture cost -0.063, while Dread found RadImageNet negative in his own [737696]. His 0.945 set is unreleased [748181].
- Hidden-test shake-up expectations: Holbrook says low-effort ensembles usually fall; participants expect many 0.957 blends to drop; about 135 teams share each 0.001 near the medal lines [749570].

---

## 5. What is new since 10-16

New topics:
- **750236** (10-17 11:59, Michał Choiński): consolidated ask on external-data checks, medals vs prizes, removal, geo, pinning before 10-22. 0 replies.
- **750257, 750258, 750259, 750260** (10-17 13:39-13:42, Karan Bhatti): one question each for nartaa OAI weights, anatomy.pt (SKM-TEA/KneeXNet), R7 (fastMRI), RadImageNet. Each states prevalence in top notebooks (nartaa dataset in ~90% of notebooks ≥ 0.95; anatomy dataset ~70%; R7 about a third / ~40% with copies; RadImageNet ~60% of notebooks ≥ 0.90). 0 replies.
- **749858** (10-16 05:50, edteoh): provenance table for the public 0.956/0.957 blends (OAI via nartaa; fastMRI+ via R7 and goodpjw reader; SKM-TEA via anatomy.pt; R6 has no fm_aux; the orthofoundation ViT-L is OAI-derived but not loaded by R6/R7/S6; `...r6-frozen-fe055406` now holds R7). Two comments, no author corrections.

New comments on old topics:
- 743416: Richard_61 (10-16 06:11) restates the geo-block point with two yes/no questions and tags the full host list; hengck23 (10-16 09:02) tells everyone to "move on" because the host will not change anything in the last week. No host reply.
- 749359 (10-17 06:04) and dataset thread 750180: nartaa published full training code under `training_code/` (Apache-2.0): trainer, argv, pinned input hashes, pool and targets, blend script, CoAtNet trainer with its OAI wrapper. Corrections: S6 used the JEV-blended labels, a pool of 4,321 studies, no per-cell weights, epoch 16, no SWA; 86 studies missing from the pool were deleted locally by the authors, not removed by the host; CoAtNet SWA epochs were chosen on Gold-58. nartaa did not copy heliosli's anatomy.pt/cloud code because his dataset states no licence.
- 748181 (public-notebook rant): heliosli defends sharing; a commenter notes his notebook comments were switched off [748181, 748295]; a user asks whether releases stop for the last week (no answer).
- 735304, 743698, 747658, 749798: no technical content that changes anything.

Unreplied provenance/licence requests on datasets (all 0 comments): `heliosli/rsna-knee-compact-r7cnxfsss` model-use terms [749213], `heliosli/rsna-knee-compact-v1` weight licence [748406], `antoinegg1/...e9...v15` provenance receipt [739633], `mattiaangeli/knee-mri-fold-weights` how made [738202].

Not seen: any host or Kaggle post dated after 10-14 13:20 UTC, any pinned external-data list, any change to rules, timeline or prizes.

---

## 6. Implications for our final picks

1. **Plan on silence.** Three days without a host word, ten open asks including four specific ones from today, and Carr's explicit "no per-dataset list" mean a ruling that names nartaa/anatomy/fastMRI/RadImageNet before 10-22 is unlikely, and a late reversal (a ban applied to the leaderboard) is even less likely. The practical test remains Carr's 10-14 text. Re-check 750236/750257-60/743416/749145 on 10-19 and 10-21 anyway.

2. **Where enforcement can actually bite.** The only stated review is for prize winners (Rules 2.8, 3.9.b: disqualify or one week to remediate); nothing is announced for medals, and hundreds of teams (about 90% of notebooks ≥ 0.95) share the nartaa dependency, so a leaderboard-wide purge is impractical. For a gold target (rank ≈ 22 of 6,144 by the usual formula; public gold line 0.963), only ranks 1-10 are prize-audited, but if a top-10 team is disqualified the next team is audited. So OAI exposure matters most if our private rank lands near 10-12, and then the winner package (weights as public dataset, training code, resource list with source datasets) must be deliverable. That package is feasible for base958 only if we can publish what we used: nartaa's training code is now public; heliosli's R6 training code is in `ref/_heliosli_compact_src/`; anatomy.pt has no licence.

3. **The union is not a compliance hedge.** Any member that contains OAI, anatomy.pt or fastMRI makes the whole submission non-compliant if the host rules against it, regardless of its blend weight (50% in the union, 5-15% in own add-ons). The union carries both disclosure lists. A real hedge needs one pick with no nartaa, no heliosli (anatomy.pt/R6) and no R7/fastMRI/goodpjw reader: that is the lavinwins-only candidate (current version), whose own residual flags are RadImageNet, CC BY-NC-SA heads, and the undocumented `sofiaanjenje` training kernels and mattiaangeli fold weights. Weigh it against the plain downside: lavinwins' public notebook shows 0.958 like base958 (our own hidden-run score of it, ref 57315208, was still pending in HANDOFF), so the hedge may cost nothing on the public board; the cost is only the uncertainty of an unverified private behaviour and tie-order on identical copies (use a unique variant, as already planned).

4. **A pick that contains base958 should be treated as carrying two separate external-data exposures** (OAI through nartaa; SKM-TEA/KneeXNet through anatomy.pt). Our HANDOFF disclosure list already names both; keep it. Do not move to base958v2 or goodpjw v21+: they add a fastMRI+ exposure (the weakest on Carr's test) for no measured gain.

5. **Own members stay clean but add little.** Own add-ons use competition data, ImageNet timm weights and RadImageNet (rad31); they are compliant under any reading except a hypothetical RadImageNet ruling that would also sink a large part of the board (60% of notebooks ≥ 0.90). The 10-16 and 10-17 add-on tests (0.957 vs 0.958) are consistent with the forum's pattern: image-model pseudo-label students beat report-label teachers, so the usable lever left for our own models is the stronger-pseudo-label recipe, not more members at 0.05 weight.

6. **Rank-inside-block is the only fine signal, and public differences under ~0.003 are noise** (≈390 public studies, SD ≈ 0.003). The private score uses ~70% of ~1,300 studies (≈920 studies); with tie-order going to the earlier submission, a unique variant beats an exact copy in a tie.

7. **Late public releases:** the one-week publishing lock (~10-15) applies to new notebooks; updates to existing notebooks and datasets (heliosli updates, Dread's unreleased 0.945) are not obviously locked, and a participant reports last-hours drops in another competition. Keep the twice-daily notebook check through 10-22, but any new base must pass the same external-data screen as above before it is considered.

8. **Hidden-run hygiene we already follow** (fp16 only, ours-first blend that never raises, rank-NaN guard, 3×13 and id checks). Add nothing; the forum has no new failure mode beyond bf16 emulation, mount errors and queue waits.

Questions worth asking the user before the final pick: (a) is a hedge pick with zero OAI/anatomy/fastMRI worth more than a second unique base958-family union; (b) acceptance of the medal-level (not prize-level) enforcement assumption in point 2.
