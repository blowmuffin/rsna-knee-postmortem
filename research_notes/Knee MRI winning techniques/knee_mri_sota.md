# State of the art (2023-2026) in deep learning for knee MRI abnormality classification: architectures and training recipes reproducible in a Kaggle notebook

Compiled 2026-10-04 for the RSNA 2026 Knee Abnormality Detection Kaggle competition (12 findings, macro ROC-AUC, ~4.4k report-labelled training studies, T4 GPUs, ~60 GPU-h left).

Reading guide / source-quality notes (applies to everything below):
- "Verified in full text" = I read the paper PDF/full text myself (ELNet, MRPyrNet, Kunitomi thesis, Kaggle lumbar-spine write-ups). Most other numbers come from abstracts (Europe PMC) or from a small-model summary of the page; those are marked "(abstract)" or "(page summary)". Paywalled pages (Springer, PMC mirror) could not be opened; for those I relied on abstract text returned by Europe PMC.
- MRNet public benchmark caveat: the MRNet test set is hidden, so nearly every follow-up paper reports AUC on the 120-exam public validation split (ELNet and MRPyrNet say so explicitly). Differences of 0.01-0.02 AUC are within noise and several papers pick "best config" on the same split. Treat any MRNet AUC above ~0.97 for ACL as unverified.
- Competition context: RSNA states the 2026 challenge is the first to combine images with multilingual radiology reports (>5,000 exams, reports in 12 languages, $77k prizes, ends Oct 22) and that the test labels were annotated by expert radiologists — [RSNA news](https://www.rsna.org/news/2026/august/ai-challenge-knee-mri). RSNA's own pages disagree on site count (16 sites on five continents in the news release vs 19 sites on the challenge page) — [RSNA news](https://www.rsna.org/news/2026/august/ai-challenge-knee-mri); [RSNA challenge page](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge). Multi-site data means domain shift matters (see Q1/Q2 external-validation drops).

---

## Key Question 1: What do MRNet-benchmark leaders and newer (2023-2026) work use, what AUCs do they report, and what mattered?

### Takeaway
The MRNet lineage (per-plane 2D CNN, slice-wise features, max-pool across slices, logistic regression across planes) is ~0.92-0.94 average val-AUC (abnormal/ACL/meniscus); the best-documented gains over it come from (a) multi-scale / centre-region feature pooling (MRPyrNet: +2.2 ACL, +4.6 meniscus AUC points), (b) per-slice-independent normalisation (ELNet), (c) slice-sequence modelling (bi-GRU / transformer / bi-LSTM+attention-MIL) and (d) learnable fusion across planes. Claims of 0.97-0.985 on MRNet from 2025 papers are on the small public val split and are not independently verified. For multi-finding, report-labelled data (our situation) the closest published system is a ResNet18 + slice-transformer ("Res3DKneeModel", Eur Radiol 2026) with external AUC median 0.78.

### Cited Findings
**MRNet baseline and the "classic" recipe**
- MRNet (Bien 2018): AlexNet slice-feature extractor (ImageNet-pretrained), max-pool across slices, FC+sigmoid per plane, logistic regression to combine three planes; 256x256 input; test AUC abnormal 0.937, ACL 0.965, meniscus 0.847; MRNet series: sagittal T2, coronal T1, axial PD, 17-61 slices (mean ~31); histogram-based intensity standardisation; ±25° rotation, ±25 px shift, 50% flip; ~6 h per model on a GTX 1070 — [Bien et al., PLoS Med (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509)
- External validity: MRNet (ACL) on the Rijeka KneeMRI set scored AUC 0.824 without retraining and 0.911 after retraining (<30 min) — i.e. big domain-shift penalty — [Bien et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509)
- General radiologists beat the model on ACL sensitivity (0.906 vs 0.759) and meniscus specificity (0.882 vs 0.741) — [Bien et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509)
- Stanford leaderboard top average validation AUC was 0.917 as of Oct 2020 (leaderboard metric = mean of abnormal/ACL/meniscus) — [search summary of LearnOpenCV/arXiv 2010.01947 results](https://arxiv.org/pdf/2010.01947)

**ELNet (Tsai et al., MIDL 2020) — verified in full text**
- ELNet val AUC: abnormal 0.941, ACL 0.960, meniscus 0.904 (MRNet re-run in same table: 0.936/0.956/0.826). Single-plane per task: coronal for meniscus, axial for ACL and abnormality (chosen empirically). Trained from scratch, ~211k parameters (K=4), 256x256, Adam 1e-5 to 3e-5, 200 epochs, ~1.5 h on a GTX 1070 — [ELNet arXiv](https://arxiv.org/pdf/2005.02706)
- Core trick = multi-slice normalisation (normalise over slice-wise statistics instead of BatchNorm across slices; layer norm for ACL/abnormal, contrast norm for meniscus) + BlurPool. Ablation (AUC meniscus/ACL/abnormal): MultiSliceNorm+BlurPool 0.904/0.960/0.941; BatchNorm+BlurPool 0.751/0.871/0.841; MultiSliceNorm+MaxPool 0.848/0.923/0.943; BatchNorm+MaxPool 0.797/0.906/0.880. Authors report BatchNorm made training diverge after 10-15 epochs because it standardises across slices that should be processed independently — [ELNet arXiv](https://arxiv.org/pdf/2005.02706)
- Table values are "best performing model per pathology", i.e. selected on the same val split — [ELNet arXiv](https://arxiv.org/pdf/2005.02706)
- On KneeMRI (917 sagittal PD exams) ELNet 5-fold AUC 0.913 — [ELNet arXiv](https://arxiv.org/pdf/2005.02706)

**MRPyrNet (Dunnhofer et al., MIDL 2021) — verified in full text**
- Plug-in for MRNet/ELNet: Feature Pyramid Network + "Pyramidal Detail Pooling" (PDP) that pools sub-regions at several grid sizes (D) around the centre of the slice before series-wise max-pool. MRNet val AUC ACL 0.955 -> 0.976 and meniscus 0.843 -> 0.889 with MRPyrNet; ELNet meniscus 0.869 -> 0.895, ACL 0.940 -> 0.960 (mean of 3 seeds); sensitivity rose strongly while specificity dropped slightly — [MRPyrNet paper](http://proceedings.mlr.press/v143/dunnhofer21a/dunnhofer21a.pdf)
- Ablation (ACL/meniscus AUC): neither module 0.943/0.828; FPN only 0.960/0.859; PDP only 0.971/0.855; both 0.972/0.879. Multi-scale centre crops fed to plain MRNet ("slice-level pyramid") already gave 0.969/0.865 — [MRPyrNet paper](http://proceedings.mlr.press/v143/dunnhofer21a/dunnhofer21a.pdf)
- Pooling choice per finding: average pooling inside PDP was better for meniscus (0.879 vs 0.849 with max) while max or average were equal for ACL; authors argue ACL tears give a focal high signal (max works) while meniscal tears have more balanced signal (avg works) — [MRPyrNet paper](http://proceedings.mlr.press/v143/dunnhofer21a/dunnhofer21a.pdf)
- Code is GPL-3.0, no pretrained weights listed — [MRPyrNet GitHub](https://github.com/matteo-dunnhofer/MRPyrNet)

**Transfer learning + augmentation study on MRNet (arXiv 2010.01947)**
- Best config: ImageNet-pretrained ResNet18 per plane + logistic regression; validation AUC abnormal 0.9381 / ACL 0.9557 / meniscus 0.9081 (avg 0.934). Per-plane AUC (axial/coronal/sagittal): ACL 0.934/0.926/0.897; meniscus 0.882/0.852/0.781; abnormal 0.949/0.913/0.950. Authors' conclusion: transfer learning and a carefully tuned augmentation policy (per task/plane probabilities grid-searched; flips, contrast/gamma/brightness, CLAHE/sharpen/emboss, 150 px crops) were the crucial factors; BCE with inverse class weights; batch size 1 — [arXiv abstract](https://arxiv.org/abs/2010.01947); [per-plane numbers via ar5iv page summary](https://ar5iv.labs.arxiv.org/html/2010.01947)
- Note the plane ranking (axial > coronal > sagittal for meniscus/ACL) is confounded with sequence in MRNet (sagittal=T2, coronal=T1, axial=PD) — [ELNet dataset description](https://arxiv.org/pdf/2005.02706)

**2023-2025 MRNet-benchmark papers (all MRNet val unless noted)**
- 3D weighted multi-view CNN (Front Bioeng Biotechnol 2023): 3D residual blocks with shared weights across views, MLP-attention weighted feature fusion; MRNet ACL AUC 92.86% vs MRNet 92.58%, MVCNN 91.94%, 3D ResNet-10 91.78%; weighted feature fusion beat label fusion and concatenation; more views helped (4>3>1); trained from scratch, 256x256, 17-61 slices — [Frontiers 2023 (page summary)](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2023.1268543/full)
- MLFANet-SA (Med Phys 2025): "slice-aligning" model finds boundary slices of the diagnostic region without ROI/segmentation labels + multi-level aggregation with channel-wise Top-K pooling and cross-slice fusion; MRNet ACL AUC 0.981 (sens 0.961, spec 0.941), private dataset 0.975 — [Med Phys abstract](https://doi.org/10.1002/mp.70130)
- KneeXNet (Front Bioeng Biotechnol 2025): graph conv over landmark nodes + multi-scale 3D fusion + contrastive pretraining; claims MRNet "test" AUC abnormal 0.985 / ACL 0.972 / meniscus 0.968 using 25 entropy-selected slices per plane, 100 epochs. Page summary also lists oddly generic baseline numbers (SVM 0.872 ... SENet 0.956) and calls the 120-exam split a "test set" even though MRNet's test set is hidden — treat as unverified/optimistic — [Frontiers 2025 (page summary)](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2025.1590962/full)
- Waseda thesis (Kunitomi 2026): learnable plane/sequence fusion weights + slice attention + bi-GRU "continuity" module over slice features, ResNet18 ImageNet, Adam 3e-5, BN frozen, grad-clip 1.0. Their own re-implemented MRNet baseline is weak (0.852/0.856/0.847 ACL/meniscus/abnormal), attention+prior 0.910/0.899/0.877, bi-GRU 0.934/0.924/0.918; ablation: zeroing any one of the T1/T2/PD planes lowers ACL AUC (best fixed weights 0.878 vs learned 0.910). Master's thesis, weak baseline: use only as directional evidence that sequence modelling across slices beats pure max-pooling — [Kunitomi thesis PDF](https://www.ams.giti.waseda.ac.jp/data/pdf-files/2026_kunitomi_mt.pdf)
- Dual-branch EfficientNet-B2 for ACL (medRxiv 2025) test AUC 0.93 (preprint, no further detail verified) — [medRxiv](https://doi.org/10.1101/2025.09.09.25335391)
- Stanford CS231n-style study (arXiv 2508.17567) reports ACL AUC 0.9969 with ImageNet ResNet50 + conv-skip classifier head on 103 test exams; class project, tiny test set — likely optimistic. Its ablation ranking (classifier head design > pretraining source; unfreezing 3-5 layer groups best; SGD-Nesterov + cosine) is a hyper-parameter hint only — [arXiv 2508.17567](https://arxiv.org/html/2508.17567v2)
- Meta-analysis (31 studies) pooled AI ACL-tear detection sensitivity 0.94 / specificity 0.93 vs radiologists 0.85/0.90 — [KSSTA 2026](https://doi.org/10.1002/ksa.70478)

**Multi-finding, report-labelled system closest to our task**
- "Res3DKneeModel" (Eur Radiol 2026, 23 conditions): 3D Slice Transformer built on TransMed — ImageNet-pretrained ResNet18 with residual 3D conv blocks (3x3x3 conv + 3D BN + Tanh) inserted after the stem max-pool and inside each residual block; each slice becomes a token with learnable positional embedding + class token -> encoder-only transformer; 23 binary outputs. Inputs: axial, coronal, sagittal PD-FS TSE + sagittal T1 (no contrast); 256x256; fixed 32 slices (99% of series had <33 images); per-sequence histogram normalisation, clip at 99.9th percentile, rescale [0,1], ImageNet mean/var; 5-fold CV. 3,121 internal studies; labels extracted from the final radiology reports by medical students and reviewed by two radiologists (i.e. report-derived, like ours); external test 458 studies — [Eur Radiol full text via Europe PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13035746); [code](https://github.com/spacecheck/Res3DKneeModel)
- Their per-condition external AUC and generalisation: median 0.78 (range 0.57-0.99) external vs 0.85 internal CV; mean absolute internal-external gap 0.05 ± 0.03; no ablations reported — [Eur Radiol](https://pmc.ncbi.nlm.nih.gov/articles/PMC13035746)
- KneeFusionNet (Arthroscopy 2026): DenseNet-based multimodal fusion for ACL/PCL/MCL/LCL; internal AUC ACL 0.971, PCL 0.906, MCL 0.919, LCL 0.924; external ACL 0.888, PCL 0.867, MCL 0.862, LCL 0.874; multimodal fusion beat single-modality for all ligaments — [Arthroscopy abstract](https://doi.org/10.1002/arj.70510)
- Domain shift evidence elsewhere: YOLOv4-based meniscus detector 95.4% internal vs 78.8% external accuracy — [Hung et al., JMRI 2023](https://doi.org/10.1002/jmri.28284)

**Adjacent Kaggle/RSNA evidence (RSNA 2024 Lumbar Spine, same host, multi-series DICOM MRI)**
- 1st place: 2-stage (localise then classify) with 2.5D 5-slice crops, ConvNeXt-small / EfficientNetV2-S encoders, then attention-MIL; adding bi-LSTM + auxiliary losses + ensembling moved public LB 0.35 -> 0.33 (attention-MIL alone 0.37 -> 0.35 vs simple 2.5D). Random shift of the slice index (±2) at training time "crucial" for robustness. Did not work: Mamba or self-attention instead of bi-LSTM, long training (7 epochs ConvNeXt-S, 14 EffNetV2-S), larger backbones (ConvNeXt-large < base < small), ViTs (convs better); trained on Colab T4 — [1st place write-up](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution)
- 2nd place: ensemble of small ConvNeXt-S MIL models per target; removed noisy training samples using ensemble OOF predictions (excluded rows where |label - OOF pred| >= 0.8, then re-weighted classes) = ~+1% on public and private LB; multiplied max spinal-severe prediction by 1.25 as post-processing — [2nd place write-up](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution)
- 7th place (hengck23): a one-stage model (heatmap-pooled grades) blended 0.25 with the two-stage model still improved it; notes top teams used multi-crop, multi-level transformers/LSTMs rather than single-crop models — [7th place post](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539439)
- Both 1st and 2nd depended on coordinate/ROI annotations that the knee competition does not provide (see Q3).

### Inferences
- The most reliably reproducible architecture gains (all cheap on a T4): per-slice (not batch-across-slices) normalisation or frozen/eval BN, sequence modelling over slices (bi-GRU/bi-LSTM or a small transformer) before attention pooling, multi-scale/centre-weighted spatial pooling for small lesions, learned (not fixed) cross-plane/sequence fusion, aux losses, small convnets (ConvNeXt-S/EfficientNet) over large or ViT backbones.
- Pooling should arguably be per finding: max/top-k for focal lesions (ACL, fracture, contusion), mean/attention for diffuse/balanced signal (meniscus per MRPyrNet, effusion/synovitis/OA) — inferred from MRPyrNet's avg-vs-max result plus the physical nature of each finding; not tested on those findings in any paper I found.
- Our current design (2.5D EfficientNet-B0 @288 + RadImageNet ResNet-50 @224 with attention-MIL over all series/slices) already matches the structure of the 1st-place lumbar solution; remaining headroom is likely in: slice-order modelling (bi-GRU/transformer), OOF-based label-noise filtering, per-finding heads/pooling, and ensembling diversity — not in bigger backbones.
- Published headline AUCs on MRNet cannot be compared to our macro-AUC on 12 findings; the realistic external AUC range for report-labelled multi-finding models is 0.6-0.95 depending on finding (Q2).

### Gaps
- Could not find a verified MRNet *test*-set (hidden) result for any 2023-2026 method; all numbers are val-split.
- MRPyrNet and ELNet numbers are 2020-2021 (outside the 2023-26 window) but remain the best-documented ablations; no 2023-2026 paper with a comparable, verified ablation set was found.
- Page-summary-only details (KneeXNet, 3D weighted multi-view CNN, 2010.01947 per-plane table) were not cross-checked against the PDFs.
- Pure transformer/ViT/Mamba results on knee MRI are thin (Mamba hybrid thesis exists but its numbers were not extractable); the one Kaggle report found says convs beat ViTs and Mamba did not help.

---

## Key Question 2: Multi-finding knee MRI beyond ACL/meniscus (cartilage/OA, effusion, bone marrow edema/contusion, Baker's cyst, synovitis, fracture): which sequences/planes/resolutions and what AUCs?

### Takeaway
Only a handful of papers cover several non-ligament findings. With routine 2D clinical protocols and report-derived labels, external AUCs are high for effusion (0.94) and mid-range for Baker's cyst (0.85), bone marrow edema (0.82), retropatellar cartilage (0.80), lateral meniscus tear (0.77) and tibial/femoral cartilage (0.74-0.77). Higher numbers (cartilage 0.93, meniscus 0.93, BME 0.83) come from thin-slice 3D CUBE data with compartment-level labels and ROI segmentation (extra annotations we lack). No deep-learning paper focused on knee fracture or synovitis classification from routine MRI was found.

### Cited Findings
- Eur Radiol 2026 (3,121 training studies, report-derived labels; PD-FS in 3 planes + sagittal T1; external AUCs): effusion 0.94; prepatellar bursitis 0.91; ACL tear 0.89; MCL tear 0.88; Baker's cyst 0.85; medial meniscus extrusion 0.86; medial meniscus tear 0.83; bone marrow edema 0.82; retropatellar cartilage pathology 0.80; lateral meniscus tear 0.77; femoral cartilage 0.77; tibial cartilage 0.74; lateral meniscus extrusion 0.73; lateral meniscus degeneration 0.74; non-tear ligament findings 0.58-0.69; medial meniscus degeneration 0.58; PCL 0.64 — [Eur Radiol](https://pmc.ncbi.nlm.nih.gov/articles/PMC13035746)
- Astuto et al. (Radiology: AI 2021, UCSF): hierarchical pipeline of V-Net segmentation of 11 ROIs -> bounding-box extraction -> 17 3D-CNN classifiers; input is sagittal 3D fast-spin-echo CUBE (0.5 mm slices, 384x384 acquired, resampled to 512x512; 3T GE); 1,435 studies / 294 patients; labels = simplified WORMS grades per compartment; holdout AUC cartilage 0.93, meniscus 0.93, ACL 0.90, bone marrow edema 0.83 (BME sens 70%, spec 85%); cartilage full-thickness sensitivity 71%, partial 72%. Collateral ligaments (coronal) excluded. AI assistance raised trainee-attending kappa for cartilage 0.42 -> 0.61 — [Astuto et al. (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8166108)
- Effusion: dense network on OAI sagittal TSE slices (1,628 knees) detected non-zero MOAKS effusion-synovitis with only ~62% accuracy on 163 test knees (still "better than the radiologist on average" on a 36-knee subset) — [Sci Rep 2022 abstract](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8873489/). OMERACT validation of an OAI-trained DL effusion quantifier found only Kendall tau 0.34-0.43 against expert semi-quantitative KIMRISS scores — [OMERACT/Semin Arthritis Rheum 2024](https://doi.org/10.1016/j.semarthrit.2024.152420)
- Bone marrow lesions: segmentation approaches on intermediate-weighted fat-suppressed sagittal images (Dice 0.60-0.70; patch CNN trained on 673 manually segmented datasets) — not classification — [Bioengineering 2024](https://doi.org/10.3390/bioengineering11040374); [OAC Open 2022](https://doi.org/10.1016/j.ocarto.2022.100234). A modified MRNet trained on sagittal IW-FS images with image-level BML labels (ISMRM abstract) shows weak supervision is feasible, no AUC verified — [ISMRM 2021 abstract](https://cds.ismrm.org/protected/21MProceedings/PDFfiles/4062.html)
- MRNet-style three-plane results for ACL/meniscus in a 3-plane model: see Q1.
- Medial vs lateral meniscus, region-level (Tack et al., OAI): on 2,399 DESS scans AUC anterior horn/body/posterior horn 0.94/0.93/0.93 (medial) and 0.96/0.94/0.91 (lateral); adapted to IW TSE: 0.84/0.88/0.86 medial, 0.95/0.91/0.90 lateral; multi-task network jointly regresses a bounding box so the CNN implicitly attends to the menisci — [Tack et al., Front Bioeng Biotechnol 2021](https://doi.org/10.3389/fbioe.2021.747217)
- Meniscus with arthroscopy labels, coronal PD-FS only, 199 exams: distilled student AUC medial 0.792, lateral 0.751 (undistilled lateral 0.672; teacher using arthroscopy+MRI 0.834/0.746) — [Ying et al. 2023](https://doi.org/10.3389/fbioe.2023.1326706)
- Weakly vs fully supervised sagittal IW-FS meniscal classification (LGSA-UNet, 1,756 knees): binary AUC 0.85-0.95 internal, 0.82-0.87 external — [Jiang et al. 2025](https://doi.org/10.1007/s10278-024-01198-4)
- Radiographs can infer MRI-defined cartilage/meniscal damage only at AUROC ~0.73/0.72 (OAI, ConvNeXt-Base multi-task) — a reminder that MRI-specific cues are needed — [Front Physiol 2026 (abstract via Europe PMC)](https://doi.org/10.3389/fphys.2026.1858407)
- Commercial ResNet-based Keros model: medial meniscus AUC 0.96, lateral 0.91, ACL 0.99 — [Behr et al., Int Orthop 2025](https://doi.org/10.1007/s00264-025-06531-2)
- Dataset/label sources are not interchangeable: SKM-TEA (155 qDESS scans with 3D box labels for cartilage lesion, ligament tear, meniscal tear, effusion) and OAI/MRNet/fastMRI are external knee MRI data — useful for ideas only given the competition rule — [SKM-TEA](https://arxiv.org/pdf/2203.06823)

### Inferences
- Even the best report-labelled multi-condition model lands at 0.74-0.85 AUC on the same hard classes we struggle with (lateral meniscus, cartilage/PF, BME), so large leaps on those labels from architecture alone are unlikely; label quality (see Q4) and aggregation are the main levers.
- High-AUC claims on cartilage/BME/meniscus (0.83-0.96) coincide with at least one of: isotropic thin-slice 3D data, compartment-level or region-level annotation, or ROI segmentation. None of those exist in our data.
- Plane/sequence use in these papers is "feed everything": Eur Radiol 2026 feeds all PD-FS planes + T1 into one model; Astuto uses only sagittal 3D FSE. No paper gives a clean per-finding plane ablation for the non-ligament findings.

### Gaps
- No peer-reviewed DL classifier for knee fracture or Baker's-cyst-specific work beyond Eur Radiol's 0.85 and a YOLO cyst-detection model (F1 0.879, a detection paper not verified in detail).
- No synovitis classifier from routine MRI found (only effusion-synovitis volume segmentation on OAI, Dice 0.79 in a 2026 OAC paper) — [OAC 2026 abstract](https://doi.org/10.1016/j.joca.2026.06.007)
- Per-finding plane/sequence ablations for non-ligament findings: not found.
- Eur Radiol 2026 reports no ablations, so which of its design choices mattered is unknown.

---

## Key Question 3: Evidence on resolution, slice-thickness handling, ROI cropping (knee localisation) and per-finding plane selection

### Takeaway
Almost everything on knee MRI classification uses 256x256 slices (MRNet/ELNet/MRPyrNet/Eur Radiol 2026) and either a fixed slice count (32) or variable-length bags; I found no controlled study of input resolution or slice-thickness handling for knee classification. The strongest ROI evidence is indirect: centre-region multi-scale cropping gave +1.4/+2.2 AUC points (ACL/meniscus) in MRPyrNet without any labels, and Kaggle lumbar-spine winners got their gains from localise-then-crop at high resolution, which needs coordinate labels that we do not have. Plane preference is confounded with sequence in all MRNet papers.

### Cited Findings
**Resolution and slice handling**
- 256x256 input across MRNet, ELNet, MRPyrNet, 2010.01947 (256, with 150 px crops in augmentation), and Eur Radiol 2026 (256, 32 fixed slices, 99% of series <33 images) — [MRPyrNet](http://proceedings.mlr.press/v143/dunnhofer21a/dunnhofer21a.pdf); [Eur Radiol](https://pmc.ncbi.nlm.nih.gov/articles/PMC13035746)
- Astuto's best cartilage/meniscus numbers come from 0.5 mm isotropic-ish CUBE data upsampled to 512x512 — [Astuto et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC8166108); a 2025 YOLOv9-cropped sagittal PD-FS ACL pipeline used only 128x128 crops (253 patients, AUC 0.913) — [Front Radiol 2025](https://www.frontiersin.org/journals/radiology/articles/10.3389/fradi.2025.1691048/full)
- KneeXNet claims 25 slices/plane via entropy-based slice selection; MLFANet-SA learns boundary slices to "slice-align" the diagnostic region (no ROI labels needed) — [KneeXNet (page summary)](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2025.1590962/full); [MLFANet-SA](https://doi.org/10.1002/mp.70130)
- A literature search specifically for "resolution / slice thickness" effects on DL knee classification returned no study that isolates them (only reconstruction/accelerated-protocol work) — [Europe PMC search results](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=(TITLE_ABS:%22knee%22%20AND%20TITLE_ABS:%22MRI%22%20AND%20(TITLE_ABS:%22resolution%22%20OR%20TITLE_ABS:%22slice%20thickness%22)%20AND%20TITLE_ABS:%22deep%20learning%22)&resultType=core&format=json&pageSize=30)

**ROI / knee-joint localisation**
- MRPyrNet: disorders are localised in small regions near the slice centre; multi-scale centre crops fed to MRNet ("slice-level pyramid", D=5) gave ACL 0.969 / meniscus 0.865 vs MRNet 0.955/0.843 (val, same paper), and robustness to ±20% translations was only slightly lower (ACL 0.967 vs 0.976, meniscus 0.879 vs 0.889) — [MRPyrNet](http://proceedings.mlr.press/v143/dunnhofer21a/dunnhofer21a.pdf)
- Tack et al. add a bounding-box regression head to the meniscus tear classifier so the network focuses on the menisci (needs box labels) — [Tack et al. 2021](https://doi.org/10.3389/fbioe.2021.747217)
- YOLOv9 segmentation-based ACL ROI crop -> CustomCNN AUC 0.913 (DenseNet121 0.903), 253 patients; authors state the effect of ROI errors on classification is unquantified, so no with/without-crop comparison exists in that paper — [Front Radiol 2025](https://www.frontiersin.org/journals/radiology/articles/10.3389/fradi.2025.1691048/full)
- Lumbar-spine 1st place: stage 1 predicts slice index (3D ConvNeXt, depth padded to 32) and (x,y) coordinates (2D ConvNeXt-base/EfficientNetV2-L at 512x512; pretraining on an external coordinate dataset was slightly better than ImageNet); stage 2 crops fixed windows around the predicted coordinate on 5 slices and runs MIL. Window sizes per target (e.g. ±96 px axial, 96/32 px sagittal). Slice-index jitter augmentation was key to robustness against stage-1 errors — [1st place write-up](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution)
- Lumbar-spine 2nd place used YOLOX boxes (from provided coordinates) to crop, per-target specialist models, and treated left/right and levels as the same class to multiply data — [2nd place write-up](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution)

**Plane/sequence selection per finding**
- MRNet-era: per-task best plane differs by paper (ELNet: coronal for meniscus, axial for ACL/abnormal; 2010.01947 per-plane: axial best for ACL and meniscus, sagittal worst for meniscus, sagittal/axial best for "abnormal"); all in a dataset where plane = fixed sequence (sagittal T2, coronal T1, axial PD) — [ELNet](https://arxiv.org/pdf/2005.02706); [ar5iv 2010.01947](https://ar5iv.labs.arxiv.org/html/2010.01947)
- Learned fusion weights beat fixed plane weights: dropping any one of three sequences lowered ACL AUC (fixed best 0.878 vs learned 0.910) — [Kunitomi thesis](https://www.ams.giti.waseda.ac.jp/data/pdf-files/2026_kunitomi_mt.pdf); weighted feature fusion beat label fusion and concatenation, more views better (4>3>1) — [Frontiers 2023](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2023.1268543/full)
- AnyMC3D (Dec 2025): frozen DINOv2/DINOv3 ViT-L + LoRA (rank 8, ~1.3M trainable params/task), view-specific LoRA adapters and queries, query-based attention pooling over slice embeddings; late fusion across views. 2D adapted foundation models (avg AUC 0.894 over 12 tasks) beat 3D-from-scratch (0.683-0.833); includes shoulder MRI tasks (private, 11-12k volumes, AUC ~0.865) but no knee task; code availability not confirmed — [AnyMC3D](https://arxiv.org/html/2512.12887)
- Lumbar winners found that adding sagittal T1 or a different sagittal sequence for some targets did not help (e.g. sagt1 for SCS, sagt2 for NFN) — a reminder that more series is not automatically better — [1st place write-up](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution)

### Inferences
- Without ROI labels, the cheapest evidence-backed analog of ROI cropping is a centre-biased multi-scale crop/pyramid (MRPyrNet) or an unsupervised/CAM-based knee-centre estimate; true two-stage localise-then-crop (lumbar style) would need pseudo boxes (e.g. from attention maps of the current MIL model) — untested in the literature.
- Resolution is probably not the first lever for series-level labels like effusion/Baker's cyst, but for small-structure findings (lateral meniscus, cartilage, BME, fracture) every published high-AUC result used thinner slices or higher resolution or ROI crops; testing 288 -> 384/448 on central crops for those findings only is the evidence-consistent experiment.
- Because plane is confounded with sequence in MRNet, MRNet per-plane rankings should not be transferred to RSNA; in our data a learned series-type-aware gate is safer than fixed plane rules.

### Gaps
- No controlled resolution or slice-thickness ablation for knee classification; no paper on spacing-aware resampling for heterogeneous 2D knee series.
- No labelled evidence on automatic knee-joint localisation improving classification AUC on MRI (only X-ray OA and ACL segmentation papers).

---

## Key Question 4: Findings relevant to our weak labels (synovitis, patellofemoral OA, lateral meniscus, fracture, bone contusion) — visual cues, sequences, architecture tricks

### Takeaway
Lateral meniscus tears and cartilage/BME are hard even for human readers (lateral AUC 0.82-0.87 vs medial 0.87-0.94 for radiologists on DL-reconstructed MRI), so report-derived labels for these are inherently noisier and published models plateau around 0.75-0.85. Synovitis is best seen on fluid-sensitive FS sequences (FLAIR-FS ≈ contrast-enhanced T1) but is often merged with effusion in scoring; fracture lines show as low-signal lines on T1 with surrounding oedema on FS sequences. There is almost no published DL work on fracture or synovitis classification, so tricks must be borrowed from related tasks.

### Cited Findings
**Lateral meniscus**
- Human performance: radiologists on 5-sequence 3T DL-reconstructed MRI — lateral meniscus sens 72% / spec 91% / AUC 0.82 vs medial 89%/87%/0.87 [Leonhardt AJR 2026](https://doi.org/10.2214/ajr.26.35094); lateral AUC 0.85-0.87 vs medial 0.94 [Walter, Radiology 2025](https://doi.org/10.1148/radiol.241249); lateral AUC 0.86 (sens 76%) vs medial 0.93 [Vosshenrich AJR 2025](https://doi.org/10.2214/ajr.25.32878); lateral sens 63-68% vs medial 94-95% [Johnson, Skeletal Radiol 2026](https://doi.org/10.1007/s00256-025-05058-2)
- In 544 arthroscopy-confirmed patients 3T MRI sens/spec were 91.8%/79.9% (medial) vs 80.8%/85.4% (lateral); accuracy was lower in ACL-injured knees — [J Clin Med 2021](https://www.mdpi.com/2077-0383/10/4/606). For meniscus posterior-root pathology at 3T (287 patients, KSSTA 2015) sensitivity was 0.770, specificity 0.729, PPV low, and "some root tears may not be identified until arthroscopy" — [KSSTA 2015 abstract](https://pubmed.ncbi.nlm.nih.gov/25377189)
- DL (CNN) lateral meniscus AUC: 0.781 (medial 0.882) on 100-patient clinical validation [Skeletal Radiol 2020](https://link.springer.com/article/10.1007/s00256-020-03410-2); 0.751 coronal-PD-FS-only distilled model (medial 0.792) [Ying 2023](https://doi.org/10.3389/fbioe.2023.1326706); 0.77 external vs medial 0.83 in report-labelled 3D slice-transformer [Eur Radiol 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13035746); but 0.95/0.91/0.90 (anterior horn/body/posterior horn) on OAI IW-TSE with region labels and a bounding-box multi-task head [Tack 2021](https://doi.org/10.3389/fbioe.2021.747217); 0.91 for a commercial ResNet model [Behr 2025](https://doi.org/10.1007/s00264-025-06531-2)
- Cue/aggregation: meniscal tears give "balanced" signal so average/regional pooling beat max in MRPyrNet (0.879 vs 0.849) — [MRPyrNet](http://proceedings.mlr.press/v143/dunnhofer21a/dunnhofer21a.pdf)
- Coronal PD-FS was the single plane used in the arthroscopy-labelled lateral meniscus study (AUC 0.751); MRNet-era models found coronal best for meniscus in ELNet — [Ying 2023](https://doi.org/10.3389/fbioe.2023.1326706); [ELNet](https://arxiv.org/pdf/2005.02706)

**Patellofemoral (PF) OA / cartilage**
- Report-labelled model: retropatellar cartilage pathology external AUC 0.80, femoral 0.77, tibial 0.74 (axial/sagittal/coronal PD-FS) — [Eur Radiol 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13035746); cartilage lesion AUC for radiologists 0.71 (conventional) vs 0.78 (DL super-resolution) — [Walter, Radiology 2025](https://doi.org/10.1148/radiol.241249); thin-slice 3D CUBE with compartment labels reached cartilage AUC 0.93 — [Astuto](https://pmc.ncbi.nlm.nih.gov/articles/PMC8166108)
- Literature summary: MRI sensitivity for patellofemoral defects is 87% for patella vs 72% for trochlea (review snippet) — [search summary (Weill Cornell record)](https://vivo.weill.cornell.edu/display/pubid22749495)
- Deep-learning PF work surfaced by my search is mostly patellar-cartilage segmentation and landmark-based patellofemoral instability measurement (U-Net landmarks), not OA classification — [search result, likely Eur Radiol 2024 instability-index paper; not opened (paywalled)](https://link.springer.com/doi/10.1007/s00330-024-10596-9)

**Synovitis / effusion**
- FLAIR with fat suppression vs contrast-enhanced T1 for peripatellar synovitis: strong correlation, site-level accuracy 92.1-93.9% (33 patients), i.e. non-contrast fluid-sensitive sequences can capture synovitis — [Radiology 2017](https://pubs.rsna.org/doi/10.1148/radiol.2016160155)
- MOAKS effusion-synovitis is scored on fluid-sensitive (T2/IW/PD) images and conflates effusion and synovitis; even DL on OAI sagittal TSE reached only ~62% accuracy for non-zero scores and low agreement with experts (tau 0.34-0.43) — [Sci Rep 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8873489/); [OMERACT 2024](https://doi.org/10.1016/j.semarthrit.2024.152420)
- Effusion is the easiest of the group (external AUC 0.94); Baker's cyst 0.85 — [Eur Radiol 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13035746)

**Fracture / bone contusion (bruise)**
- Standard MRI cues: on T1 the fracture line is a low-signal line against bright marrow; on fat-suppressed PD/T2/STIR the surrounding oedema is bright; bone bruises are low on T1 and high on FS T2 — [MRI Master knee fractures](https://mrimaster.com/knee-fractures-mri/); [Open Orthop J 2012 bone bruise review](https://openorthopaedicsjournal.com/VOLUME/6/PAGE/464/FULLTEXT/) (educational/review sources, not DL)
- DL BME/contusion numbers: AUC 0.83 (CUBE, 3D CNN, sens 70%/spec 85%) [Astuto](https://pmc.ncbi.nlm.nih.gov/articles/PMC8166108); 0.82 external (report labels) [Eur Radiol](https://pmc.ncbi.nlm.nih.gov/articles/PMC13035746); BML segmentation Dice 0.60-0.70 on IW-FS [Bioengineering 2024](https://doi.org/10.3390/bioengineering11040374)

**Label-noise / weak-supervision tricks**
- OOF-based removal of high-loss samples gave ~+1% LB in the 2nd-place lumbar solution (threshold |label - OOF prediction| >= 0.8) — [2nd place](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution)
- Arthroscopy-informed teacher -> MRI student knowledge distillation improved lateral meniscus AUC 0.672 -> 0.751 (needs arthroscopy labels) — [Ying 2023](https://doi.org/10.3389/fbioe.2023.1326706)
- Report-label extraction pipelines: BioBERT needed ~1,000 annotated reports to match traditional NLP on meniscal-tear labelling, plateau at 1,250-1,500 — [Br J Radiol 2023](https://doi.org/10.1259/bjr.20220769); the Eur Radiol 2026 labels were extracted from reports by trained students with two-radiologist review — [Eur Radiol](https://pmc.ncbi.nlm.nih.gov/articles/PMC13035746)

### Inferences
- Since expert readers themselves are weaker on lateral than medial meniscus (AUC gap ~0.05-0.08) and misses cluster at the posterior horn/root and in ACL-injured knees, report-derived lateral meniscus labels likely contain extra false negatives; a lateral head should (i) see coronal and sagittal PD-FS series, (ii) use regional or mean-style pooling plus a few top-k slices, and (iii) benefit from OOF noise filtering / soft labels more than from capacity.
- Synovitis should be driven by FS fluid-sensitive series (PD-FS/T2-FS/FLAIR-FS if present); its signal is diffuse and merges with effusion, so mean/attention pooling and a shared effusion-synovitis latent (multi-task coupling with effusion label, the easiest finding) is a plausible trick — inferred, not tested in the literature.
- Fracture and contusion are focal and pair a T1 non-FS cue (line) with FS oedema; a head fed both T1 non-FS and FS series with max/top-k pooling is the natural design; contusion and fracture co-occur so multi-task coupling could help — inferred.
- PF OA likely needs axial and sagittal PD-FS views of patella/trochlea at higher in-plane resolution; centre-crop multi-scale features may help lateral/PF cartilage most.

### Gaps
- No DL paper on knee fracture classification or synovitis classification from routine clinical MRI; no study quantifying which series best detects PF cartilage loss for DL.
- Several radiology cues above come from educational pages or search-engine summaries and are not DL evidence.
- A search summary also stated that posterior-horn lateral meniscus tears have the lowest MRI sensitivity of any meniscal location, but I could not pin that sentence to a specific paper, so it is not listed as a cited finding.
- I could not verify the qDESS-vs-contrast synovitis correlation (rho 0.96) to a primary link, so it is omitted.

---

## Key Question 5: Open-source code and pretrained weights (license; pretraining dataset; competition-rule implications)

### Takeaway
Code exists for ELNet, MRPyrNet, Res3DKneeModel and several foundation models, but few ship knee-specific weights. The relevant public weights are RadImageNet (already used), MRI-CORE (ViT-B, trained on 110k Duke MRI volumes incl. knee; permissive license, segmentation-focused repo), and DINOv2/v3-based recipes. OrthoFoundation (knee X-ray+MRI, DINOv3) reports SOTA but I could not confirm that weights are released. Whether weights pretrained partly on knee MRI are allowed is a rules question the user must confirm; the user stated only public pretrained weights are allowed.

### Cited Findings
- ELNet: official code at github.com/mxtsai/ELNet (license not verified); weights 850 kB (K=4), trained from scratch on MRNet — [ELNet arXiv](https://arxiv.org/pdf/2005.02706)
- MRPyrNet: GPL-3.0, PyTorch 1.7.1, MRNet+MRPyrNet and ELNet+MRPyrNet pipelines, no pretrained weights listed — [MRPyrNet GitHub](https://github.com/matteo-dunnhofer/MRPyrNet)
- Res3DKneeModel (Eur Radiol 2026): train.sh with Slurm, SliceTransformer over ResNet-18 features, supports per-pathology training; pretrained weights and license not mentioned in README — [Res3DKneeModel GitHub](https://github.com/spacecheck/Res3DKneeModel)
- MRI-CORE: ViT-B (DINOv2-style SSL per the paper), pretrained on Duke-110K (116,806 MRI volumes, ~6.9M slices, 18 body regions including knee, internal Duke data; no public datasets such as OAI/fastMRI listed), input 1024x1024; paper states CC BY 4.0, GitHub README says Apache 2.0, SAM-based ViT-B segmentation repo with `MRI_CORE_vitb.pth` on Google Drive; downstream tests include normal/abnormal classification — [MRI-CORE arXiv](https://arxiv.org/abs/2506.12186); [MRI-CORE arXiv HTML](https://arxiv.org/html/2506.12186); [mazurowski-lab/mri_foundation](https://github.com/mazurowski-lab/mri_foundation). The license statements conflict between sources.
- OrthoFoundation: DINOv3 backbone, ~1.2M unlabeled knee X-ray and MRI images from internal and public databases, 14 downstream tasks, "ranked first in MRI structural injury detection", matches supervised baselines with 50% labels; paper license CC-BY-NC-ND (paper, not weights); weight/code release not found — [OrthoFoundation arXiv](https://arxiv.org/abs/2601.18250)
- PRISM: 336,476 MRI volumes from 34 datasets (8 public, 26 private), 44 downstream tasks, no knee-specific task listed; weight availability not stated — [PRISM arXiv](https://arxiv.org/abs/2508.07165)
- RadImageNet: 1.35M images (CT/MRI/US, 131,872 patients); on small datasets RadImageNet beat ImageNet by 4.8% AUC (ACL MRI) and 4.5% (meniscus MRI) — [RadImageNet, Radiol AI 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758); disputed: independent study found ImageNet best in six of seven datasets with only the knee dataset favouring RadImageNet — [search summary of arXiv 2302.08272](https://arxiv.org/pdf/2302.08272); CS231n study found ImageNet ACL AUC 0.9969 vs RadImageNet 0.9594 (p=0.017, n=103 test) — [arXiv 2508.17567](https://arxiv.org/html/2508.17567v2)
- AnyMC3D (DINOv2/DINOv3 ViT-L + LoRA multi-view recipe): no code link found; DINOv3 license not checked — [AnyMC3D](https://arxiv.org/html/2512.12887)
- Foundation-model segmentation toolkit for MSK MRI (SAM/SAM2/MedSAM fine-tuning with automatic prompting, open source "AutoMedLabel") — [arXiv 2501.13376](https://arxiv.org/abs/2501.13376)
- Kaggle lumbar-spine winning code/weights are public on Kaggle/GitHub (1st: Colab notebooks + model datasets; 2nd: github.com/yujiariyasu/rsna_2024_lumbar_spine_degenerative_classification; 7th: github.com/hengck23/solution-rsna-2024-lumbar-spine) — [1st](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution); [2nd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution); [7th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539439)

### Inferences
- Because competition rules reportedly forbid external knee MRI training data but allow public weights, MRI-CORE (internal Duke data) and RadImageNet are the safest foundation-style weights; weights pretrained on OAI/MRNet/fastMRI-style knee data (e.g. SKID-type SSL, OrthoFoundation if released) are a grey area that should be checked against the competition rules text before use.
- On a T4 with ~60 GPU-h, a ViT-L LoRA recipe (AnyMC3D) is likely too heavy; ViT-B MRI-CORE at 1024 px is also heavy, though it could be run at lower resolution or used only for feature extraction (untested).

### Gaps
- Licenses for ELNet code and Res3DKneeModel not verified; MRI-CORE license conflict unresolved.
- No confirmed public knee-MRI classification weights for any 2023-2026 paper.
- I did not verify the competition rules text myself (the user supplied that constraint).

---

## Key Question 6 (derived): Which ideas are reproducible in our setting, which need annotations we lack?

### Takeaway
Reproducible with no extra labels and modest compute: slice-order modelling before MIL pooling, per-finding heads/pooling, OOF label-noise filtering, multi-scale centre crops, slice-index jitter augmentation, small-backbone ensembles. Needs annotations we do not have: ROI/coordinate two-stage cropping, region/compartment-level meniscus or cartilage labels, segmentation-guided classification, arthroscopy-informed distillation.

### Cited Findings
(Evidence is cited in Q1-Q5; this section only organises it.)
- Needs no extra annotations: bi-LSTM/bi-GRU + attention-MIL [1st place lumbar](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution), slice-transformer [Eur Radiol 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13035746), multi-scale centre pooling [MRPyrNet](http://proceedings.mlr.press/v143/dunnhofer21a/dunnhofer21a.pdf), per-slice-independent normalisation [ELNet](https://arxiv.org/pdf/2005.02706), OOF noise removal [2nd place lumbar](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution), slice-aligning without ROI labels [MLFANet-SA](https://doi.org/10.1002/mp.70130), learned multi-plane fusion [Frontiers 2023](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2023.1268543/full).
- Needs extra annotations: bounding-box/ROI heads [Tack 2021](https://doi.org/10.3389/fbioe.2021.747217), V-Net ROI segmentation + per-compartment classifiers [Astuto](https://pmc.ncbi.nlm.nih.gov/articles/PMC8166108), YOLO knee/ACL ROI cropping [Front Radiol 2025](https://www.frontiersin.org/journals/radiology/articles/10.3389/fradi.2025.1691048/full), arthroscopy teacher distillation [Ying 2023](https://doi.org/10.3389/fbioe.2023.1326706), coordinate-based lumbar two-stage pipelines [1st place lumbar](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution).

### Inferences
Suggested priority order for the remaining ~60 GPU-h (my judgement, ranked by evidence-per-cost; none of these were validated on RSNA 2026 data):
1. OOF-based noisy-label down-weighting/removal for weak findings (cheap, +1% in an analogous Kaggle task; labels are report-derived).
2. Add a slice-sequence module (bi-GRU or 1-2 layer transformer with positional embeddings) between per-slice encoder and attention-MIL; consider random slice-index/series dropout jitter.
3. Per-finding pooling heads (max/top-k for ACL/fracture/contusion; mean/attention for effusion/synovitis/OA/meniscus) with a series-type embedding so the model can route PD-FS vs T1 vs T2.
4. Centre-biased multi-scale crops (e.g. 1.0x, 0.75x, 0.5x centre windows) for lateral meniscus/PF/cartilage/fracture, at higher resolution than 288 only for those heads.
5. Multi-task coupling (effusion-synovitis, fracture-contusion) and ensembling across diverse backbones/seeds; drop larger backbones/ViTs unless budget remains.

### Gaps
- No evidence that any of these specific combinations works on RSNA 2026 data; they must be tested against the EXPERIMENTS.md log and local CV before use.
