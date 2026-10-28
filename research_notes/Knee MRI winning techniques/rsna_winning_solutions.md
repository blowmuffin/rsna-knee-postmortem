# RSNA winning solutions (2021-2025): study-level classification from multi-series MRI/CT, and what transfers to RSNA 2026 Knee Abnormality Detection

Evidence tags used below: **[ABL]** = the write-up reports a number for the change (ablation, CV, or LB delta; confounded ones are marked "confounded"). **[CLAIM]** = author says it helped, with no number. **[NEG]** = author says it did not help. Scores: 2022-2024 write-ups quote log-loss style metrics (lower is better, e.g. 0.37 -> 0.33); 2025 quotes AUC-style scores (higher is better). All write-ups were read in full through the Kaggle API on 2026-10-04 unless stated under Gaps. Kaggle URLs are given as full links to the write-up.

Quick map of what I read: 2024 Lumbar Spine ranks 1, 2, 3, 4, 5, 6, 7 (hengck23's part), 8, 9 + a community "Summary of Top Team Solutions"; 2023 Abdominal Trauma ranks 1, 2, 3; 2022 Cervical Spine ranks 1, 2, 5; 2025 Intracranial Aneurysm ranks 1, 2, 3, 4, 5; 2021 MGMT rank 1.

---

## Q1. RSNA 2024 Lumbar Spine Degenerative Classification (multi-series MRI, 25 targets): what did the top 1-10 do?

### Takeaway
Ranks 1-6 and 9 were all "localise the disc level (slice index + x,y keypoint) -> crop a small region -> small 2D CNN (ConvNeXt-S / EffNetV2-S / CoAtNet / ResNet18) + attention/RNN over 3-30 slices -> ensemble"; ranks 7-8 used a one-stage model whose detection heatmap pools features instead of cropping, and matched or complemented the two-stage models. Every localiser was supervised by host-provided or community-crowdsourced keypoints, which our knee data does not have.

### Cited Findings
**Placements** — 1 Avengers (NANACHI), 2 IanPan-Kevin-Yuji-Bartley, 3 SonySpine s & tkmn & Moyashii, 4 SPINE CHART, 5 Two people, 6 NVSpine, 7 HLIP, 8 K_mataro, 9 Adam Narai; 1,874 teams — [RSNA results page](https://www.rsna.org/news/2024/november/2024-ai-challenge-winners).

**1st — Avengers (NANACHI), single author, trained on Colab T4** — [write-up](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution)
- Stage 1 is split in two. (a) Slice-index ("instance_number") prediction for sagittal volumes: a 3D ConvNeXt on the whole series padded to depth 32, with a classification head (CE) and a regression head (L1), median-ensembled over 5 folds. Reported accuracy of the predicted slice vs truth: cls 71.1% exact / 27.0% off by 1 / 1.4% off by 2 / 0.4% worse; reg 67.5 / 30.6 / 1.6 / 0.3 [ABL]. (b) x,y keypoint regression: 2D ConvNeXt-base and EffNetV2-L at 512 px, L1 loss on normalised coordinates, pretrained on the community coordinate dataset ("slightly better than ImageNet-pretrained") [CLAIM]. Axial slice-to-level assignment reused hengck23's DICOM geometry projection.
- Stage 2: crop around the predicted keypoint on 5 slices (centre = predicted slice) at 512 px scale; crop sizes e.g. sag-T2 for canal stenosis 96 left / 32 right / 40 up / 40 down px, axial +-96 px. ConvNeXt-small / EffNetV2-S encoder, then Attention-MIL, then bi-LSTM + aux attention loss.
- Documented LB deltas: Attention-MIL moved public LB 0.37 -> 0.35 [ABL]; adding bi-LSTM + aux losses + ensembling moved 0.35 -> 0.33 [ABL, confounded: three changes]. "MIL better than simple 2.5D" [CLAIM].
- Augmentation: random shift of keypoint (+-10 px) and of predicted slice index (+-2, with probability matched to the stage-1 error rate) — "crucial for robustness to stage-1 errors" [CLAIM]. After crop: BrightnessContrast p=0.25, ShiftScaleRotate (shift 0.1, scale +-0.1, rot 20).
- [NEG] Mamba and self-attention instead of bi-LSTM; sharing weights between aux and main attention; sag-T1 for canal stenosis, sag-T2 for foraminal; long training (used 7 epochs ConvNeXt-S, 14 epochs EffV2-S); bigger models (ConvNeXt-large < base < small); ViTs (convs better).

**2nd — Ian Pan / Kevin / Yuji Ariyasu / Brendan Artley** — [write-up](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution)
- Ensemble of small ConvNeXt-S models; axial and sagittal handled separately, one model per target; all models treat level and left/right as the same task (one shared model, 3 classes). Slice classifiers (hengck23 code for axial; 2.5D timm model for sagittal) choose slices; YOLOX (trained on Artley's coordinates) gives per-level boxes, then horizontal levelling of each box before cropping; sagittal MIL over 5 slices; axial for non-spinal targets uses only the left or right half so both sides share one model.
- Label-noise removal: using the team's ensemble OOF (CV 0.3687), drop training samples where |label - OOF prediction| >= 0.8, with class coefficients on moderate/severe; author states it "improved our score by 1% on both public and private" [CLAIM, no ablation table; idea came 2 days before deadline]. Post-processing: multiply the highest predicted spinal-severe value per study by 1.25.
- A third-party summary reports the team trained on 27 crop variants (3 slice shifts x 9 keypoint shifts) and averaged the same 27 at inference as TTA, which "greatly improved performance" [CLAIM, second-hand] — [summary post](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/541279). The same summary author could not reproduce the OOF-filter gain (score "consistently dropped") [NEG, one replication], and the 6th-place team lists "denoising techniques" under what did not help (below).

**3rd — SonySpine s & tkmn & Moyashii** — [write-up](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/sonyspine-s-tkmn-moyashii-3rd-place-solution)
- Stage 1: CenterNet keypoint detector (EffNet-B6 + FPN) for disc levels, trained on competition + community coordinates, pseudo-labelled on the rest, and the team hand-corrected all annotations; axial slices assigned to levels via real-world coordinates; second CenterNet (EffNet-B4) for the spinal-canal point on axial. Crops at several sizes — "almost no difference in accuracy due to cropping settings" [ABL-ish, qualitative].
- Stage 2: 2D encoder + attention over 15 equally spaced sag-T1 + 15 sag-T2 + 10 axial-T2 slices (no slice-index estimate; a transformer/attention head over all slice features). Encoders: ResNet18 (160/224), MNasNet-S, EffNet-B4, EffNetV2-RW/S, ConvNeXt-N/T (224/320), MaxViT-N (256).
- Key design: level-agnostic and side-agnostic training. Each (level, side) crop group is an independent sample, giving "5x (canal) and 10x (foraminal/subarticular) the data"; right-side data is mirrored (sagittal slice order reversed, axial flipped) so left and right share one model ("accuracy improved") [CLAIM, no number].
- Aux losses (other conditions, slice-level), severity class weights CE [1, 2, 4], TTA = axial flip "improved LB", drop-path 0.2-0.3, 10-20 epochs, lr 2.5e-5 with OneCycle, batch 2-8, aug: BrightnessContrast, Blur, Distortion, ShiftScaleRotate, CoarseDropout, optional Mixup [CLAIM].
- Numbers: best single model CV 0.3858; 30-model simple average (15 centre + 15 side, varied input/arch/aug/aux/pseudo-label) CV **0.3643** (-0.0215) [ABL]. Temperature scaling T=0.91 on canal-stenosis logits. Pseudo labels: "change in accuracy ... not significant", kept for diversity [ABL-ish, qualitative].
- [NEG] one-stage solution, multi-level multi-disease models, per-level or per-side specialised models, 3D-CNN, 2.5D-CNN + attention, 2D-CNN + LSTM, focal loss, long epochs.

**4th — SPINE CHART (tattaka, yu4u)** — [write-up](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/spine-chart-4th-place-solution)
- Keypoint stage: 2.5D CNN + LSTM predicts the disc level of each axial slice, UNet heatmaps for left/right keypoints; sagittal keypoints from UNet (caformer_s18 / convnext_tiny / resnetrs50 / swinv2_tiny, SCSE decoder, 0.2 BCE + 0.8 Dice) on 20 x 256 x 256 volumes.
- Crop size = 2x the distance between neighbouring keypoints (adaptive scale per patient); axial affine-aligned to put left/right keypoints at fixed positions; +-2 slices around the predicted disc gap.
- Classifier: per-slice 2D backbone, transformer encoder + attention pooling; "key technique ... apply attention pooling to the features before the transformer and calculate the auxiliary loss" [CLAIM]. Backbones caformer_s18, resnetrs50, rdnet_tiny, maxxvitv2_nano.
- Stacking: Nelder-Mead-guided MLP and LightGBM/XGBoost per level; no ablation numbers.

**5th — siwooyong + Ahmed El Fazouani** — [write-up](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/two-people-5th-place-solution)
- Stage 1: heatmap detector for 25 classes, one model per series type (sag-T2, sag-T1, axial-T2), 2D UNet + sequence model (transformer/LSTM) over slices, EfficientNet-B5. "Gaussian-expanding labels" (Gaussian in x,y; halved weight per slice away in z) sped convergence and improved z accuracy [CLAIM]; external coordinate dataset gave "further improvements" [CLAIM].
- Stage 2: crop +-2 slices in z, +-32 px in x,y, (5,64,64) resized to (5,128,128); CNN + RNN with **level-wise sequence modelling** (RNN only across the 5 targets of the same level, not all 25) — "significantly improved" [CLAIM]. ConvNeXt best of regnet/effnet/convnext.
- **Two-step training**: first train the backbone for AUC without the metric-weighted loss, then freeze the backbone and fine-tune only the head with the weighted/any loss; training directly on the metric overfit the weights [CLAIM, "significantly improve"].
- **CutMix p=1.0** was the most useful augmentation vs flip/rotate/brightness/blur/mixup for stage 2 [CLAIM, no number]. Final: backbone-diverse ensemble, and a "TTA-like" ensemble (run stage 2 once per individual stage-1 model, then average) gave the best private LB.
- [NEG] adding the stage-1 mask as a stage-2 channel; bigger CNN in stage 2; label smoothing; +-1 z-jitter and L/R label flip gave no significant gain.

**6th — NVSpine (Theo Viel, Darragh, Dieter)** — [write-up](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/nvspine-6th-place-solution)
- Three heterogeneous pipelines aggregated by an MLP that optimises the competition metric (per-target independent logits, weights shared across levels). Final CV 0.382, public LB 0.355, private LB 0.401 (shows a CV/LB gap in the other direction) [ABL].
- Model 1: CoAtNet-rmlp-2-384 regresses 5 (x,y) disc points on centre frames; square crop = 20% of image size around the point; CoAtNet-1/2 224 + RNN on the crop. Training both sagittal modalities in one model "worked better", more frames (5 not enough) and 3 RNN heads for left/right/centre helped, plus MixUp; this sagittal-only pipeline reaches ~0.42 private [ABL].
- Model 2: SpineNet points corrected with the coordinates file, z-localisation regressing the number of annotations per slice, variable number of slices per crop. Model 3: study-level "IVD model" tiling the 3 views' crops into one 288x288x9 image, view-dropout p=0.1, EffNetV2-S with 3D pointwise convs.
- "Augmenting the target coord position was very effective" for crop robustness [CLAIM].
- [NEG] SpineNet / medical-specific models for the condition-level modelling, pseudo-labelling on external data, denoising techniques, encoder-decoder with auxiliary localisation, bi-encoder joint axial/sagittal.

**7th — HLIP; hengck23's one-stage part** — [post](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539439)
- One-stage "point-masking and pool": predict a pixel-wise level heatmap (Jensen-Shannon loss, DSNT to xyz), predict a pixel-wise grade, pool grades with the heatmap. Hard to train because coordinate and grade objectives compete; labelling errors (L5/S1 confusion) hurt.
- Submission blend 0.75 two-stage + 0.25 one-stage; "even with bugged implementation, the one-stage model improves the two-stage model" [CLAIM; the figure is an image, no number]. Note from the author: top teams used "multi-crop, multi-level" transformer/LSTM models rather than single-crop single-level.

**8th — K_mataro, Taro_pan** — [write-up](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/k-mataro-8th-place-solution)
- No cropping: a 2D classifier predicts all 25x3 labels per image; keypoint detection is a sub-task and its heatmap acts as attention weights for feature pooling; a 1D classifier stacks per-instance outputs. Claimed benefits: keeps whole-image context, robust to detection ambiguity [CLAIM].
- Self-supervised multi-view similarity (xyz coordinates) slightly improved public/private but not CV — author guesses test labels are cleaner than train [CLAIM]. Manual relabelling with a custom tool "didn't end up being very useful" [NEG].

**9th — Adam Narai** — [write-up](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/adam-narai-9th-place-solution)
- Heatmap keypoints (DeepLabV3Plus-ResNet34, 512 px, MSE, per series type). **Slice selection by rule**: middle slice for sag-T2, fixed mm offsets from the middle for sag-T1, axial slices projected from sagittal coordinates. 50x50 mm ROIs, 5 slices as channels, 128 px (ResNet18) or 224 px (Swin-T / ConvNeXt-N) + GRU head, 3-5 epochs, 5-fold; ensemble of 6 model types x 5 folds. Right-side axial crops mirrored so both sides share a model. Channel-shuffle augmentation (p=0.5) "significantly improved performance" for axial [CLAIM].

**10th-14th** — not read in full; per the community summary the 10th used YOLO/SpineNetV2 for ROIs, 11th and 12th used rule-based slices/adaptive crops and shift augmentation, 14th exists — [summary post](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/541279).

**Cross-team patterns (community summary of the top solutions)** — [summary post](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/541279)
- Almost all two-stage; per-level treatment gives ~5x (10x with L/R) effective data; single-stage 7th/8th matched or beat two-stage when modelled well.
- Slice-index estimation: trained models (1st, 2nd, 4th) or rules (9th, 11th); teams with slice estimation used ~3-5 slices, teams without used ~10-30 equally spaced slices (2nd, 3rd, 4th did not estimate slice indices).
- Common 2nd-stage pattern: small 2D backbone per slice -> LSTM/GRU -> attention pooling (or transformer encoder across all views) -> head; ConvNeXt and EffNetV2 popular, all small. No clearly superior way to fuse multiple views (concatenate pooled vectors vs transformer over all tokens).
- Shifting keypoints/slice index in training was effective against stage-1 errors (2nd, 11th, 12th).
- The community coordinate dataset ("Coordinate Pretraining Dataset", crowdsourced keypoints) was used by 1st, 3rd, 4th, 5th, 6th — [dataset post](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/524500).

### Inferences
- 2024's decisive resource was localisation labels (host coordinates + a community dataset of ~5 disc keypoints per sagittal image). None of the winners built a localiser without keypoints; the knee competition provides none, so the two-stage recipe cannot be copied directly.
- The top teams' stage-2 heads (small 2D CNN, per-slice features, attention pooling, optionally a bi-RNN or a transformer over all tokens) are essentially what our per-finding attention-MIL already is; the documented headroom there is small (0.37 -> 0.35 -> 0.33 LB in the 1st-place log, mixed with aux loss and ensembling).
- Contradictory evidence on sequence heads: bi-LSTM beat self-attention/Mamba for the 1st, while the 3rd and the 2025 3rd found LSTM/MIL did not help in their setups; head choice is pipeline-dependent, not a robust win.
- The most transferable "no-localisation" lessons are those that also appear without coordinates: shared model across mirrored/equivalent instances, small backbones, few epochs, many diverse members, aux losses.

### Gaps
- 7th (HLIP full write-up), 10th, 11th, 12th, 13th, 14th write-ups were not read; 2nd-place detailed comment thread (comment 3012803, 27-crop TTA) not read directly.
- No ablation table exists in any 2024 write-up isolating the crop-vs-full-image gain; the closest evidence is that one-stage heatmap-pooling models matched two-stage ones (7th/8th) and that the 3rd's one-stage attempt failed.

---

## Q2. RSNA 2023 Abdominal Trauma, 2022 Cervical Spine, 2025 Intracranial Aneurysm, 2021 MGMT: key study-level techniques

### Takeaway
2022 -> 2023 -> 2025 form one lineage: supervised segmentation/detection of the anatomy of interest -> crop -> 2.5D CNN (+ RNN/attention) over slices -> max/top-N aggregation, with auxiliary segmentation losses, soft slice-level labels derived from study labels, and flip TTA; the largest documented ablation gains in these competitions come from segmentation-based pretraining/aux losses and from crops/resolution, not from architecture swaps. 2021 MGMT (no localisation labels, weak signal) was dominated by noise and shake-up.

### Cited Findings

#### RSNA 2023 Abdominal Trauma (4274 CT studies; liver/spleen/kidney grade, bowel, extravasation)
- **1st Team Oxygen (Nischay Dhankhar, Qishen Ha, Harshit Sheoran)** — [write-up](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/writeups/team-oxygen-1st-place-solution-team-oxygen)
  - 3D segmentation -> study-level crop from organ-mask boundaries; 96 equidistant slices reshaped to (32, 3, 384, 384) 2.5D (3 adjacent slices as channels), so the model sees the whole organ stack.
  - **Soft slice labels** = study label x per-slice organ visibility (mask area normalised to 0-1); negative study -> all-zero slice labels. Model outputs per-slice sigmoids; study prediction = max over slices.
  - **Auxiliary segmentation loss** (Dice on predicted masks from the last and second-last encoder blocks, weight 0.125) made training "much more stable" and gave "+0.01 to +0.03" [ABL, author-quoted range, no table]. Models: CoaT-Lite-Medium/Small + GRU, EffNetV2-S + GRU; AdamW, cosine, 4-fold group CV; aug Perspective/HFlip/VFlip/Rotate; best single OOF 0.326, ensemble OOF 0.31x; no major post-processing [ABL].
- **2nd Theo Viel** — [write-up](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/writeups/on-strike-2nd-place-solution)
  - Explicitly built on 2022 winners' ideas. 2D models + RNN; "the key to achieve good performance with 2D models is cleverly sampling frames to feed meaningful information and reduce label noise" — an organ-visibility classifier chooses frames inside the organ (random frame inside the organ for organ-grade targets).
  - Crop models (3D ResNet18 crops organs -> 2D CNN + RNN on 11 frames sampled uniformly inside the organ; CoatNet-1 + RNN) "improve performance on kidney, liver and spleen by a good margin". Scores: 2D+RNN ConvNeXt-v2 public 0.41 / private 0.39 -> + crop models (MaxViT + CoatNet-RNN) 0.37 / 0.35 -> ensemble 0.35 / 0.35 [ABL, crop step confounded with backbone swap]. Ensembling "did not really help on private" [ABL/NEG].
  - Aug: HFlip, ShiftScaleRotate, colour/blur, ElasticTransform, CutMix p=0.5 (p=1.0 for crop models). 384 px centre crop of 512; 1 of 2 frames, stack capped at 600; DICOM loading with dicomsdl on GPU -> "pipeline without ensembling < 4 h" [ABL, timing].
  - [NEG] YoloX extravasation boxes; sequential head in stage 1 (not needed once crop models were strong).
- **3rd Yuji Ariyasu** — [write-up](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/writeups/aillis-jp-yuji-ariyasu-3rd-place-solution)
  - 2022 1st-place 3D segmentation code -> organ cubes (mask slightly dilated because context around the organ matters); 2.5D+3D classifiers on (8, 15, 3, 128, 128); mask x box input for liver "very effective"; a custom sampler so each batch contains one organ type; two crop sizes; pre-training on image-level bowel/extravasation labels. Single weighted-average ensemble; **stacking that optimises the metric overfit** and was worse than simple post-processing. Final CV 0.3316 [ABL].

#### RSNA 2022 Cervical Spine Fracture (CT; C1-C7 fractures; weighted BCE)
- **1st Qishen Ha** — [write-up](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/writeups/qishen-ha-1st-place-solution)
  - Stage 1 3D UNet (ResNet18d / EffV2-S, 128^3) trained on only the 87 provided masked studies, then predicted masks for all ~2k studies. Stage 2: crop each vertebra (14k samples), 15 even slices x 5 channels + predicted mask as 6th channel, 2D CNN + LSTM (type 1, per vertebra) and a patient-level type 2 (7 x 15 images in one forward pass, small backbones). 512 px EffV2-S, 384 px ConvNeXt-T/nano/NFNet; 7.5 h submission time.
  - **"Training a 3D CNN on this data did not give me satisfactory results"**, so he backed off to 2.5D + LSTM [NEG].
- **2nd RAWE** — [write-up](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/writeups/rawe-2nd-place-solution-segmentation-2-5d-cnn-gru-)
  - 2.5D CNN + UNet segmentation (EffNet-B0) on 87 studies; crop each vertebra to 24 evenly sampled slices; CNN (EffV2-S / ResNeSt50d) + bi-GRU + attention; GRU weight re-initialisation and SpatialDropout "a little improvement" [CLAIM]; heavier augmentation was not better [NEG].
- **5th Speedrun (Christof Henkel, Psi, Pascal Pfeiffer)** — [write-up](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/writeups/speedrun-5th-place-solution-team-speedrun)
  - Slice-level pseudo labels = predicted vertebra visibility x study label (2D vertebra-visibility classifier trained on the 87 masked studies); ROI box per study from segmentation (0.05/0.95 quantiles of per-slice boxes); 2.5D+3D model (stride-5 neighbours) on full images and on ROI crops; heavy regularisation (mixup, random crop-resize, shift/scale/rotate, brightness); inference upscaled by >= 1.125.
  - Stage 3: feed-forward net on mean/min/max of slice predictions optimising the competition metric; 30/70 blend with max-aggregated output "improves our scores by 2-3 LB points" [ABL, self-reported].

#### RSNA 2025 Intracranial Aneurysm (CTA/MRA/MRI; 13 locations + present; AUC-based; 2xT4 inference limit)
- **1st tomoon33** — [write-up](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution)
  - Coarse-to-fine nnU-Net vessel segmentation -> ROI (140 mm cube centred on largest DBSCAN cluster) -> 3D classifier whose backbone is the segmentation-pretrained nnU-Net; per-location "vessel-region-masked pooling" of decoder features + location-aware transformer; auxiliary task reconstructing a small sphere at each annotated aneurysm; BCE heads, all 14 labels independent.
  - **Ablation (folds 0-2, 96x192x192)**: final 0.902; without backbone pretraining **0.794**; without auxiliary segmentation loss **0.876**; loss weights all 1.0 -> 0.884 (aux sphere weight 1.0 vs 0.1/0.05 for cls); without seg-model 3 0.891; no location transformer 0.896; Dice instead of Focal-Tversky++ 0.899; full-res 128x256x256 **0.916** (+0.014 vs 96x192x192) [ABL].
  - Inference on 2xT4: 4.0 s preprocess + 10.7 s vessel segmentation + 3.3 s classification per series; TTA = left-right flip; **fail-safe**: if segmentation/ROI fails, output the OOF mean per class instead of a prediction [ABL, timing]. Left-right mirroring disabled for the fine seg models to keep asymmetry.
- **2nd Pengcheng Shi et al. (Medical Image Insights / UZH)** — [write-up](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution)
  - Stage 1: nnU-Net 2D on 9 slices (3 per axis at 1/4, 1/2, 3/4) -> vascular ROI. Stage 2: 3D nnU-Net multi-task (vessel + aneurysm seg + classification), resized to 224^3, cross-attention pooling (faster convergence than average pooling), modality head for stability, encoder-only inference, 8x flip TTA with **left/right label swap**, external TopCoW/TopBrain annotations; ran ~9 h on 2xT4 with a conservative 2-fold ensemble.
  - **Ablation (public/private)**: baseline 0.844/0.813 -> + TTA 4x 0.871/0.825 -> better segmentation annotation data 0.888/0.857 -> + L/R-flip augmentation with label swap + TTA 8x 0.898/0.862 -> + 2nd fold ensemble 0.900/0.867 [ABL; steps are cumulative and some confounded with annotation upgrades]. Authors saw a 3-4% public-private gap and noted deleted DICOM fields in the test set broke dicom2nifti.
- **3rd TmT / YYama et al.** — [write-up](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/3rd-place-solution)
  - YOLOv8 detects the vessel region from mid-sagittal and mid-coronal slices (GT boxes = bounding boxes of mask MIPs, mAP50 > 0.95), 3D ROI 90-120 mm; 3D ResNet-18 (timm-3d) with a 14-class head applied to **every feature-map position**, top-N-mean aggregation over positions; **reducing stride from 2 to 1 in selected conv layers** (feature map 8^3 -> 25^3, input 128^3 -> 196^3) moved single-fold LB **0.77 -> 0.81** [ABL, two changes at once]. 11-model ensemble 0.86/0.84 [ABL].
  - **A model estimating voxel spacing from image appearance (many test DICOM tags were missing) improved private 0.84 -> 0.85** [ABL]. [NEG] vessel segmentation, modality/plane aux heads, MIL/LSTM/decoder heads.
- **4th Harshit Sheoran (8-day speedrun)** — [write-up](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/4th-place-solution)
  - Segmentation model "not happy to learn beyond 0.6 Dice" in the time available, so he trained a DINOv3 ViT to regress the crop box (x1,x2,y1,y2) from 48 equidistant 128x128 slices at patient level ("far more reliable and efficient"); ROI kept deliberately large (95%+ of aneurysm coordinates inside). Then a plain 2D CoaT-Lite-Medium slice classifier at 384 px, max over slices.
  - **Cumulative table (CV; public/private)**: CoaT-L-M on ROI crops 0.805; 0.78/0.78 -> + Rotate(-25,25) 0.83; 0.80/0.81 -> + 2.5D (-2,0,+2) 0.86; 0.86/0.83 -> + soft pseudo-distillation on the remaining 500k unlabeled slices 0.89; 0.86/0.84 -> + MaxViT+CoaT ensemble 0.896; 0.87/0.84 [ABL]. Weighted sampling did not help despite ~1:250 slice imbalance [NEG].
- **5th HoangHuyen** — [write-up](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution)
  - Hand-annotated YOLO boxes for aneurysms around host centroids (+-10 slices) and for the brain on mean-of-series images; **brain-box cropping "improves the model's accuracy by about 0.03-0.05"** [CLAIM]; horizontal flip with left/right label swap +0.01 [CLAIM]; only box-containing slices used for positive series. 384 px ViT-L / EVA-L / MiT-B4-FPN (multi-task cls + seg from boxes).
  - External TOF-MRA datasets pseudo-labelled + "clean trainset" step (flip negatives to positive when two models give > 0.9): OOF AUC MiT-B4 0.8549 -> 0.8629, ViT-L 0.8503 -> 0.8558, EVA-L 0.8551 -> 0.8579 [ABL, two changes confounded]. Ensemble OOF 0.882, public 0.89.

#### RSNA 2021 MGMT brain tumour (MRI, 4 sequences, binary; no localisation labels)
- **1st Firas Baba** — [write-up](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/writeups/tunisia-ai-i-love-this-competition-1st-place-solut)
  - Winning model was one of his first baselines: ResNet10 3D, 256 px, BCE, 15 epochs, no ensembling. CV was "almost random" for early models; the same EfficientNet-B0 trained 100 times with identical settings gave CV 0.53-0.62; to rank ideas he trained 100 then 250 models per idea (20-50 seeds x 5 folds) and averaged OOFs. "Biggest central image" slice-selection trick: CV +0.01-0.02 [ABL, only 100%-successful experiment]. Ensembling, 2D CNNs, DICOM metadata, brain-pretraining, stacking, deep CNNs did not help [NEG]. He expected a shake-up; "most public models fail to learn any useful patterns".

### Inferences
- Across 2022/2023/2025, the localisation stage was trained on host-provided masks/boxes/centroids (87 masked studies in 2022; masks "given" in 2023; `train_localizers.csv` centroids in 2025; keypoints in 2024). The 2021 MGMT competition, the one without such labels, produced near-noise results. This is the key structural difference to our knee setting (58 expert study-level labels, no localisation).
- Segmentation-derived signals appear in the largest documented effects: 2025 1st (+0.108 from seg pretraining, +0.026 from aux seg loss), 2023 1st (+0.01-0.03 aux seg loss), 2023 2nd (public/private -0.04 from organ crops, confounded). Pure architecture changes tend to be reported as +0.00-0.01.
- Soft slice-level labels constructed as "study label x anatomical visibility" (2022 5th, 2023 1st, 2023 2nd) are the same idea as MIL attention but make the slice supervision explicit; the visibility part needs organ localisation.
- 3D CNNs from scratch failed in 2022 1st and 2024 3rd (in 2021 the winner found 2D CNNs failed and used a small 3D ResNet10 on a noise-dominated task); 3D worked in 2025 mainly when initialised from nnU-Net segmentation weights (1st) or with a tiny ResNet-18 and large feature maps (3rd).

### Gaps
- Not read: 2022 3rd (Darragh), 4th, 6th ("3D CNN + TD-CNN"), 2023 4th-10th, 2025 6th-9th, and any 2021 ranks 2-10. The 2025 2nd-place paper cited in its write-up (arXiv 2606.26706) was not opened.
- The RSNA 2023 review paper (arXiv 2407.16165) abstract page contains no technique-level comparison; no aggregate meta-analysis of winners' techniques was found there.
- Private-LB values for 2025 1st place are not in the retrieved write-up; only CV ablations are.

---

## Q3. Recurring patterns across competitions and which gave the largest documented gains

### Takeaway
Ranked by documented effect size: segmentation-based pretraining/aux supervision (+0.026 aux loss, +0.108 backbone pretraining in the 2025 1st ablation) > anatomy-focused crops/ROI (about -0.04 log-loss in 2023 2nd, +0.03-0.05 claimed in 2025 5th, but unmeasured in 2024) > spatial resolution/feature-map size (+0.014 and ~+0.04 LB) > 2.5D and rotation (+0.025-0.03 CV each in 2025 4th) > flip-TTA/L-R-swap augmentation (+0.01-0.03 public) > diverse ensembles (-0.02 log-loss in 2024 3rd, nil in 2023 2nd) > soft pseudo-label distillation (+0.03 CV but only +0.01 private in 2025 4th) > sequence/attention heads (+0.02 LB in 2024 1st). Label-noise filtering was +1% for the 2nd-place lumbar team but failed to replicate for others.

### Cited Findings
**Localisation / cropping to the anatomy of interest**
- Used by nearly all winners 2022-2025 (see Q1/Q2). Documented measurement: 2023 2nd public 0.41/private 0.39 -> 0.37/0.35 after adding organ crop models [ABL, confounded with MaxViT swap] — [2nd write-up](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/writeups/on-strike-2nd-place-solution). 2025 5th: brain box crop "+0.03-0.05" [CLAIM] — [5th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution).
- Level/side-agnostic crop pooling multiplies data 5-10x (2024 3rd, 9th) — [3rd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/sonyspine-s-tkmn-moyashii-3rd-place-solution). Crop-size choice itself was insensitive ("almost no difference") for the 3rd; adaptive crop = 2x keypoint spacing (4th).
- Crop robustness: shift crop coordinates and slice index in training (1st, 2nd, 6th, 11th, 12th) [CLAIM, "crucial" in the 1st] — [1st](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution). Counter-note: the 5th tried +-1 z-jitter of the stage-1 slice and saw no significant gain [NEG] — [5th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/two-people-5th-place-solution).
- Single-stage alternatives that avoid hard crops: heatmap-weighted pooling (7th, 8th) matched/complemented two-stage — [7th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539439), [8th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/k-mataro-8th-place-solution). The 3rd's own one-stage attempt did not work [NEG] — [3rd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/sonyspine-s-tkmn-moyashii-3rd-place-solution).
- Localiser type does not matter much: CenterNet (3rd), UNet heatmap (4th, 5th, 9th), coordinate regression (1st, 6th), YOLO/YOLOX (2nd, 2025 3rd, 5th), ViT box regression (2025 4th), nnU-Net seg (2025 1st/2nd, 2023/2022).
- Cheap/rule-based ROI also reached the top 10: 9th used middle slice and fixed mm offsets with 50x50 mm ROI — [9th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/adam-narai-9th-place-solution); 2021 1st used "biggest central image" (+0.01-0.02 CV).

**Higher-resolution crops / resolution**
- 2025 1st: 128x256x256 vs 96x192x192 +0.014 [ABL]. 2025 3rd: larger feature maps via stride 2->1 and input 128->196: 0.77 -> 0.81 LB (single fold) [ABL] — [3rd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/3rd-place-solution). 2022 1st trained EffV2-S at 512 px; 2023 1st at 384 px; 2024 winners used 128-320 px on crops (3rd: 224/320; 9th: 128 or 224; 5th: 64x64 crop upsampled to 128). 2022 5th upscaled test images by >= 1.125 at inference [CLAIM].
- Bigger backbones did not help: ConvNeXt-large < base < small (2024 1st), bigger stage-2 CNN [NEG] (5th), ViTs worse than convs (1st); 2024 top teams "all commonly used smaller models" (summary).

**Sequence models over slice features (LSTM / GRU / transformer / attention-MIL)**
- 2024 1st: Attention-MIL then bi-LSTM; summary says teams commonly used 2D backbone -> LSTM/GRU -> attention pooling. 2022-2023 winners: CNN + LSTM/GRU/bi-GRU+attention (1st, 2nd 2022; 1st, 2nd, 3rd 2023). Counter-evidence: 2024 3rd and 2025 3rd found LSTM/MIL/2.5D+attention did not help in their setup; 2023 2nd found the sequential head unnecessary once crops were strong. So: head type is setup-dependent, [no clear winner per the summary on multi-view fusion].
- Level-wise sequence modelling (RNN only across targets of the same level) "significantly improved" the 5th's score [CLAIM]; a Location-Aware Transformer across 13 vessel locations added only ~+0.006 in the 2025 1st ablation (0.902 vs 0.896) [ABL].

**Multi-stage pipelines**
- 2022 5th: a third-stage NN on (mean, min, max) of slice predictions gave +2-3 LB points [CLAIM]; 2023 3rd: metric-optimising stacking overfit [NEG]; 2024 4th and 6th used MLP/GBM stacking, with 6th reporting "per-target independent MLP" best. For an AUC metric (no class-weighted log-loss) the metric-optimising aggregation motive is moot.

**Pseudo-labelling / knowledge distillation**
- 2025 4th: soft pseudo-distillation on 500k unlabeled slices: CV 0.86 -> 0.89, public 0.86 -> 0.86, private 0.83 -> 0.84 [ABL] — [4th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/4th-place-solution).
- 2025 5th: +0.003 to +0.008 OOF AUC from external data pseudo-labels + cleaning (confounded) [ABL]. 2024 3rd: pseudo labels "not significant" (used for diversity) [ABL-ish]; 2024 6th: pseudo-labelling on external data [NEG].
- Slice-level "study label x visibility" soft labels (2022 5th, 2023 1st/2nd) are a form of structure-aware weak supervision (see Q2).

**Auxiliary losses**
- Aux segmentation: 2025 1st 0.876 -> 0.902 [ABL]; 2023 1st "+0.01-0.03" [ABL-range]. Aux attention score fed to CE directly + aux depth head (2024 1st); aux slice-level and other-condition losses (3rd); aux loss after attention pooling before transformer is "key" (4th). Modality/plane aux heads: helped stability in 2025 2nd, did not help in 2025 3rd; "encoder-decoder with auxiliary localisation" did not help the 2024 6th.

**Augmentation**
- CutMix p=1.0 "most significant" (2024 5th) [CLAIM]; CutMix p=0.5/1.0 (2023 2nd). MixUp: used by 2024 3rd (optional), 6th (sagittal models), 2022 5th; Rotation +-25 deg: +0.025 CV in 2025 4th [ABL]; channel shuffle for axial (2024 9th) [CLAIM]; L/R flip with label swap: +0.01 (2025 5th) [CLAIM], and as part of the 2025 2nd step 0.888 -> 0.898 public (with 8x TTA) [ABL, confounded]; flip TTA on axial improved LB (2024 3rd) [CLAIM]; heavier augmentation not better (2022 2nd) [NEG].

**Test-time augmentation / inference tricks**
- 2025 2nd: TTA 4x public 0.844 -> 0.871, private 0.813 -> 0.825 [ABL]; 2024 2nd: 27-pattern crop-shift TTA "greatly improved" [CLAIM, second-hand]; 2024 5th: running stage 2 once per stage-1 model and averaging ("TTA-like" ensemble) gave the best private LB [CLAIM].
- Inference speed: 1-in-2 frames, GPU preprocessing, dicomsdl (< 4 h, 2023 2nd); encoder + classification head only (2025 2nd); ~18 s/series for a 3-stage nnU-Net pipeline on 2xT4 (2025 1st); 7.5 h submission (2022 1st). Fail-safe to OOF prior on pipeline failure (2025 1st).

**Ensembling**
- 2024 3rd: single best CV 0.3858 -> 30-model average 0.3643 [ABL]; 2025 4th: +0.006 CV from adding a MaxViT [ABL]; but 2023 2nd: ensembling "did not really help on private" [NEG]; 2021 1st: did not help [NEG].

**Label noise**
- 2024: 2nd OOF filter (drop |label - OOF| >= 0.8) +1% [CLAIM] vs failed replication [NEG, summary author] and "denoising did not help" (6th); 3rd hand-relabelled keypoints; 8th relabelling [NEG]; 8th suggests test labels cleaner than train and SSL helped LB but not CV [CLAIM]. 2025 5th flipped negatives with > 0.9 model score to positive (confounded gain). 2025 2nd manually corrected left/right/supra-infraclinoid label mix-ups [CLAIM].

**Validation reliability**
- 2021 1st: identical configs gave CV 0.53-0.62 across seeds; needed 20-50 seeds x 5 folds per idea; 2024 6th: private 0.401 vs CV 0.382 vs public 0.355; 2025 2nd saw a consistent 3-4% public-private gap with test DICOM fields deleted.

### Inferences
- The only techniques with large documented effects require segmentation/keypoint supervision that the knee data lacks. Techniques with portable, supervision-free evidence are: shared modelling across mirrored/equivalent instances (3rd/9th lumbar; 2025 L/R-swap), aux/slice-level deep supervision (several), crop/shift jitter, many small diverse members (3rd), CutMix (5th lumbar; 2nd 2023), and feature-map/spatial resolution for small lesions (2025 3rd).
- Documented effect sizes of generic tricks (TTA, ensembling, pseudo-labels, heads) are mostly +0.003 to +0.02 and shrink on the private test; this is the same order as our logged per-experiment gains (~+0.002 LB).

### Gaps
- No write-up gives a clean A/B of "crop vs full image" for MRI (2024) because the one-stage attempts differ in many ways; the best isolating numbers are from 2023 CT (confounded) and 2025 claims.
- Several [CLAIM] items (CutMix, level-wise RNN, two-step training, noise-filter +1%, channel shuffle) lack numbers; their effect sizes are unknown.

---

## Q4. Which ideas are feasible for the knee competition without extra annotations (no boxes/keypoints), and which transfer?

### Takeaway
Direct copies of the winners' supervised localisers are not available; the transferable pieces are (a) supervision-free training tricks with some cross-competition evidence (slice-level deep supervision, CutMix, feature-map resolution, shared modelling of medial/lateral compartments, shift jitter), (b) cheap robustness/diagnostic checks motivated by 2025 (missing DICOM tags in test, fail-safe), and (c) heuristic or public-model ROI extraction, which has no RSNA precedent without labels and carries rules and runtime risk.

### Cited Findings
- Project state (from our own logs): own ensemble saturated at ~0.898 OOF / LB 0.938; resolution 224 -> 288 gave +0.006 OOF (11/12 labels up); prep v2 (wider slice band, true-scale crop) no gain; encoders bigger than ResNet-50/res > 288, EMA/mixup/SWA, TTA, metadata, extra pseudo rounds, Raptor knobs: no gain; external knee datasets: KneeCoT banned, others unanswered; prep already crops to a fixed physical size (`CROP_MM`), samples 31 slices per series, and canonicalises everything to a left knee (sagittal order reversed and other planes mirrored for right knees) — [EXPERIMENTS.md](D:/kaggle/rsna-knee/EXPERIMENTS.md), [CLAUDE.md](D:/kaggle/rsna-knee/CLAUDE.md), [HANDOFF.md](D:/kaggle/rsna-knee/HANDOFF.md).
- HANDOFF flags an unexplained ~0.015 LB shortfall at equal gold score and asks whether hidden-test scans miss planes/sequences or metadata (forum topic 744519 unanswered) — [HANDOFF.md](D:/kaggle/rsna-knee/HANDOFF.md).
- In RSNA 2025, test-set DICOMs had many deleted tags/multi-frame quirks: the 2nd place switched from dicom2nifti to pydicom with slice-count-based spacing; the 3rd trained a spacing regressor that lifted private 0.84 -> 0.85 — [2nd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution), [3rd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/3rd-place-solution).
- Public segmentation resources (unverified depth): TotalSegmentator MRI exists (`total_mr`); a Slicer-forum thread says the femur comes from `total_mr` and tibia/fibula/patella from an `appendicular_bones_mr` module, and that it has no module for the whole knee (menisci/ligaments not covered) — [Slicer forum](https://discourse.slicer.org/t/how-to-segment-the-tibia-using-totalsegmentator). fastMRI+ provides 16,154 expert bounding boxes and 13 study-level labels over 22 pathology categories for the fastMRI knee set, but each knee exam is a single coronal PD/T2 series — [fastMRI+ paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8983757).
- All localisation supervision in the winners' pipelines came from host data or hand annotation: 2022 1st "only 87 samples w/ 3d mask" — [1st 2022](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/writeups/qishen-ha-1st-place-solution); 2025 5th hand-drew boxes around host centroids ("does not require specialized medical knowledge") — [5th 2025](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution); 2025 4th regressed a crop box from mask-derived boxes — [4th 2025](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/4th-place-solution).
- Rule-based slice/ROI selection worked in the top 10 (9th lumbar: middle slice + mm offsets) — [9th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/adam-narai-9th-place-solution).

### Inferences
Ranked by (evidence strength x feasibility without annotations). All are hypotheses for our setup, not verified results.

**Tier 1 — cheap, motivated, low risk**
1. **Check test-DICOM metadata completeness** (PixelSpacing, ImagePositionPatient/ImageOrientation, SliceThickness, number of series/planes per study) against train, because our prep relies on physical-size crops and slice sorting. 2025 showed this is a real failure mode (2nd and 3rd). Zero GPU cost; could explain part of the unexplained LB shortfall noted in HANDOFF. If tags are missing, add a fallback (2nd: spacing from slice count; 3rd: learned spacing regressor +0.01 private).
2. **Slice-level deep supervision inside the MIL head**: feed per-token attention/aux logits to the study-label CE (1st lumbar `aux_attn_score`), plus the 3rd's slice-level aux predictions. No new labels, small code change; evidence is 4 lumbar write-ups + 2025 1st aux ablation (+0.026 with real seg masks; magnitude with weak labels is unknown).
3. **CutMix on slice stacks / images with high probability** (5th lumbar "most significant", 2023 2nd). Mixup is on our known-not-to-work list (forum + ours) but CutMix is not logged as tested; for localised findings CutMix label-mixing is semantically odd, so treat as an A/B on 2 folds.
4. **Fail-safe path** (2025 1st): on any prep/inference exception output a stored OOF-mean prior per label; relevant to the 9 h limit and the efficiency track.

**Tier 2 — plausible, moderate cost**
5. **Feature-map resolution instead of more input pixels** (2025 3rd: stride 2->1 in selected layers, +0.04 LB single fold with input change): applies to EfficientNet-B0/ResNet-50 on small structures (lateral meniscus, fracture are our weakest labels in HANDOFF/EXPERIMENTS). Costs more compute per image; our logs say input resolution gains stop above 288, which this alters differently.
6. **Shared head/encoder across mirrored compartments** (medial/lateral meniscus, medial/lateral OA): analogous to the 3rd/9th lumbar level-/side-agnostic sharing (5-10x data per condition). Our canonical left-knee orientation already realises the L/R part of the 3rd's "Split LR".
7. **Shift/scale jitter of the sampled slice grid and crop window** (1st, 2nd, 6th lumbar) — generic robustness; check it is already in `to_input` (our training uses a random half of slice centres).
8. **OOF-disagreement filtering or down-weighting** (2nd lumbar +1% claim, but failed replication): we already blend 0.5 report label + 0.5 OOF; hard filtering of |label - OOF| large cases is the harder version. Test offline on existing OOFs before GPU; the 58 gold studies are too noisy to judge it (project note: gold is not a reliable LB proxy). Expect small gain, risk of harming rare classes.
9. **Cross-finding transformer on per-finding pooled vectors** (2025 1st Location-Aware Transformer +0.006; 5th lumbar level-wise RNN "significant"): our cross-label stacker gave no gain, so expect little.

**Tier 3 — no RSNA precedent without labels, or rules/runtime risk**
10. **Foreground/anatomy crop heuristics**: 2025 5th's brain-box crop claims +0.03-0.05, but our prep already crops to a fixed physical size and prep v2 showed no gain, so the likely headroom is the fraction of image area that is still background; measure that first (CPU-only).
11. **Attention-derived ROI (self-supervised second pass)**: use Grad-CAM/feature-map attention from the trained MIL model to define a finding-specific crop for a high-resolution second model. Closest precedents (7th/8th lumbar heatmap pooling) were supervised with coordinates; none of the winners used weakly supervised ROI. High effort, uncertain.
12. **Bone masks from TotalSegmentator MRI -> compartment ROIs** (tibial plateau medial/lateral, intercondylar notch, patella) to emulate "disc level" coordinates. Seg-based crops are the largest documented family of gains, but here the segmenter is not trained on our data, covers bones only (per the Slicer thread), has unknown behaviour on thick-slice 2D knee MRI, adds nnU-Net runtime inside a 9 h 2xT4 limit (2025 1st spent ~10.7 s per series for fine segmentation), and its competition eligibility is unconfirmed (our log lists a similar "public checkpoint eligibility" question as unanswered).
13. **Small self-annotation of keypoints/boxes** (e.g., 100-300 sagittal/coronal slices) to train a ROI regressor like the 2025 4th (box regression from 48 equidistant slices) or 5th (hand boxes). Requires anatomical knowledge to place ACL/meniscus landmarks (unlike 2025 5th, where centroids were provided) and is annotation by definition.

**Do not expect gains from** (consistent with winners' [NEG] lists and our log): bigger encoders/ViTs, long schedules (3rd: "long epochs" [NEG]; 1st: 7-14 epochs), 3D CNNs from scratch (2022 1st, 2024 3rd), focal loss and label smoothing (3rd, 5th lumbar), more pseudo-label rounds (2024 3rd/6th; ours saturate), and ensembling beyond a diverse set (2023 2nd, 2021 1st).

**Evaluation caution**: the 2021 winner needed 20-50 seeds x 5 folds per idea to detect effects of ~0.01; our ~0.002 gains are below single-run noise on 58 gold studies and public LB.

### Gaps
- Test-set size, the number of knees/sites in test, and per-study series availability are unknown, so runtime and robustness estimates for any added stage (segmentation, ROI) cannot be made.
- Whether external segmentation models or datasets (TotalSegmentator, fastMRI+, SKM-TEA, OAI-ZIB) are allowed under the competition rules is unconfirmed; our own logs show KneeCoT banned and other cases unanswered.
- I found no RSNA winner who derived an ROI without host-provided localisation labels; whether attention-derived or heuristic ROIs help knee MRI is untested in the sources read.
- TotalSegmentator MRI class list (especially any knee soft-tissue classes) was not verified against its official documentation; only a forum thread was found.
- Effect sizes for [CLAIM]-tagged techniques are unknown, so the Tier 1/2 ordering is judgement, not measured.
