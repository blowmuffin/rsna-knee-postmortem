"""Catalog of every public notebook: score, votes, model family from attached datasets, notes for the key ones.
python catalog.py <kernels_web.json> <out.csv>"""
import json, sys
import pandas as pd

FAM = [  # (substring of an attached source, family tag)
    ('nartaa/rsna-knee-publication-swa', 'raptor-coatnet-nartaa(OAI)'), ('nartaa/rsna-knee-s6fall', 'S6-nartaa'),
    ('heliosli/rsna-knee-r6', 'helios-R6'), ('heliosli/rsna-knee-compact-r7', 'helios-R7(fastMRI)'),
    ('heliosli/rsna-knee-compact-v1', 'helios-S6'), ('heliosli/rsna-knee-compact-r6', 'helios-S6'),
    ('heliosli/rsna-orthofoundation', 'helios-code'), ('goodpjw2008/rsna-knee-2-5d-convnext-reader', 'goodpjw-reader'),
    ('bliverrigpdoge/', 'kcd-private'), ('dreaddevelopment/raptor-knee', 'raptor-coatnet-dread'),
    ('mattiaangeli/knee-mri-fold-weights', 'dino-mattia'), ('pilkwang/rsna-knee-weights', 'dinov2-pilkwang'),
    ('antoinegg1/', 'radimagenet-heads'), ('prvsiyan/rsna-knee-v52', 'radimagenet-heads'),
    ('mattiaangeli/rsna-knee-coat', 'coat-mattia'), ('tonylica/', 'tony-assets'),
    ('marwanmath/resnet-50-radimagenet', 'radimagenet')]

NOTES = {
    'heliosli/blend-mine-in-private-lb-luck-to-all-upvoters':
        "TOP 0.958. Raptor 384+224 (0.65/0.35, mirror TTA) + helios R6 (4 folds, flip TTA, fluid+state stackers); "
        "per-finding Raptor weight Synovitis .8, Lateral OA .2, PF OA .6, Baker's .35, else .5. R6 stage ~6 min on the 3 visible studies (529 windows/study).",
    'goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-957':
        "Our base (v21 = 0.957). v26 = v21 + the 0.958 per-finding weights. Own 2.5D ConvNeXt-T reader alone 0.929; "
        "reader at full weight on the 0.956 base -> 0.954; at 1/4 -> no gain; kept at 1/10.",
    'goodpjw2008/rsna-knee-coatnet-s6-2-5d-mil-lb-0-954':
        "Third weaker reader on nartaa+S6: displayed 0.954, ranked higher. S6-distilled reader (gold 0.931) at full weight: 0.953.",
    'heliosli/this-is-best-public-dataset-new-version-0-9576':
        "R6 + flip TTA with Raptor 50/50 = 0.957 (0.9576 reported). The R7 -> R6+flip step is what moved 0.956 -> 0.957.",
    'heliosli/0-957-latest-dataset-score-in-one-click':
        "R7 (ConvNeXt-S, crop 352, knee-joint anchors, fastMRI aux, refit Effusion/Baker's state stackers) + Raptor 384, 50/50 = 0.956.",
    'heliosli/rsna-knee-blend-gold-dataset-restored': "S6 4-fold ConvNeXt-S + Raptor, 70/30 = 0.954.",
    'heliosli/rsna-knee-blend-gold': "S5 compact + Raptor, 70/30 = 0.951.",
    'nartaa/rsna-knee-0949-anatomical-mirror':
        "Single CoAtNet-rmlp-2 384 (OAI MOAKS rows + JEV-blend labels, 16 ep, top-3 epoch average), 96 slices 26/22/18/12/18, 140 mm, "
        "native 384 -> 320 centre crop, 94 windows, anatomical mirror TTA: 0.949. Plain 0.948; crop 304 0.947; 288 0.946; 94->24 windows -0.006.",
    'nartaa/rsna-knee-0945-efficient-224crop':
        "Same weights dataset, 0.8 centre crop -> 224, one view: 0.945 alone; at 35% of the Raptor slot moves rank inside the block.",
    'nartaa/rsna-knee-s6fall-solo':
        "helios S6 pipeline (ConvNeXt-S, full FOV + 100 mm knee-centred crops via SKM-TEA V-Net anchors, 64 windows, 16 ep) on nartaa JEV labels: "
        "0.950 single; folds 1-4 0.948; fold-0 OOF 0.907 vs CoAtNet 0.890.",
    'bliverrigpdoge/r7-raptor-knee-centred-convnext':
        "Private KCD25 add-on (ConvNeXt-S + EffNetV2-S, knee-centred medial/lateral compartment crops from sag+cor plus axial, ImageNet init, "
        "d25 soft labels) at 0.2 on the 0.956 R7 base -> 0.957.",
    'bliverrigpdoge/velci-kcd25-20':
        "Same KCD25 at 0.2 on the 0.954 base -> 0.955. The only comp-data-only add-on with a measured +0.001, twice.",
    'varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956':
        "224 checkpoint at 35% of the Raptor slot: displayed unchanged, rank 437->320 (0.954 recipe), 233->156 (0.956). Rank inside a tie block = paired signal.",
    'leoprovorov/rsna-knee-0-957-explained-why-the-open-ensembl':
        "Analysis: public models' error correlation rho ~0.86-0.89 -> family ceiling ~0.960; public-half SD ~0.003.",
    'matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed':
        "CoAtNet + ConvNeXt MIL two-reader blend: 0.950 (first step past the 0.943 DINO stacks: fewer, more diverse models).",
    'mattiaangeli/bend-the-knee-to-the-dinosaurs':
        "DINOv2/v3 folds + RadImageNet heads + CoAtNet arms, per-label weights: 0.941. Lateral Meniscus weight 1.00 = pure CoAtNet (tuned on the public LB).",
    'jiweiliu/rsna-knee-fast-2xt4-inference': "4-way CoAt consensus (resgated, D4, Global96, Repair-v1): 0.943; our first base.",
    'pilkwang/rsna-knee-baseline-v1': "20-member DINOv2-S ensemble: 0.891 (first public baseline).",
    'dreaddevelopment/knee-mri-twelve-findings-from-a-single-model':
        "Raptor CoAtNet 384 single: 0.924 at 42 windows, 0.927 at 62, 0.928 with 64 slices, 0.932 with 80. Slice density, not span, is the lever.",
    'dreaddevelopment/knee-mri-training-the-twelve-finding-model':
        "Raptor trainer: inputs in [0,1] (no ImageNet norm), 12 random windows train / 24 eval, bb 3e-5 head 1e-3, wd 0.02, OneCycle 15% warm-up, "
        "clip 3, pos_weight (1-p)/p in [1,10], 16 ep, top-3 epoch average. Gold screen: coatnet384 .9025, swinbase384 .8825, cnn336 .8833, "
        "effv2l480 .8716, cnbase384 .8754, maxvit384 .8438.",
    'hengck23/mavit-288-baseline-01-1-5sec-study-1gpu':
        "MaViT linear attention, 6 views x 8 slices, 288 px, 130 mm: 0.935 at 1.5 s/study (GM baseline).",
    'paiky1995/rsna-knee-0-926-lb-coatnet-fine-tune-blend':
        "Fine-tuning the public Raptor CoAtNet on a Kaggle T4: 48 min/epoch on 3,575 studies; holdout weak AUC fell 0.908 -> 0.870; "
        "3-way blend 0.926 LB. Retraining a 384 CoAtNet on Kaggle GPUs is not viable.",
    'romantamrazov/rsna-knee-dinosaur-v5-train':
        "ConvNeXt-T 224 diversity arm: 3-epoch encoder adaptation on soft labels, frozen features, label-aware transformer head on gold folds; blend beta 0.04-0.12.",
    'sofiaanjenje/rsna-knee-e11-train': "RadImageNet heads (E11) for the DINO stack; fold-0 weak-val AUC 0.79 after epoch 0.",
    'wguesdon/rsna-knee-dinov2-at-meniscus-resolution':
        "DINOv2 at 336, ridge+attention heads: OOF 0.788 vs extracted labels; LLM labels beat the rule extractor mainly on non-English reports.",
    'karttikjangid05/rsna-knee-which-llm-labels-can-you-trust':
        "Label sets vs gold-58: jev_hybrid .904 (tuned on gold), steven_v4 .893, consensus .889, steven_v2 .887, yunus .880, jev_final2 .875, "
        "pilkwang .870, qwen .852. No set clearly better on 11/12 findings.",
    'dariushafshar/rsna-report-pseudo-labels-add-0-022-auc-4-5':
        "Gold-only model + 250 agreement-masked pseudo studies: +0.022 on gold folds (tiny n).",
    'dehaiwang/rsna-knee-v21-dinosaur-training': "DINOv2 2 seeds x 3 folds on Kaggle: OOF 0.735 (weak).",
    'stevenleehans/rsna-knee-exp025-champ-5fold-seed7': "DINOv2 5-fold, last-3-epoch averaging: derived OOF ~0.79; LB 0.823.",
    'prvsiyan/the-bee-s-knees-final-rsna-push':
        "Ablations on 0.940: residual heads = no change; Raptor intensity-gain change -> 0.922 (big regression).",
    'kiritohayate95/rsna-knee-abnormality-detection-train-dinov2': "Fitted LB ~ 0.899 x gold + 0.113 over 5 submissions.",
}

raw = json.load(open(sys.argv[1]))
rows = []
for k in raw:
    ref = k['scriptUrl'].removeprefix('/code/')
    srcs = [s.get('dataSourceUrl') or '' for s in (k.get('dataSources') or [])]
    fams = sorted({f for s in srcs for sub, f in FAM if sub in s})
    rows.append(dict(ref=ref, title=k.get('title', ''), best_public_score=k.get('bestPublicScore'), votes=k.get('totalVotes') or 0,
                     created=str(k.get('dateCreated', ''))[:10], last_run=str(k.get('lastRunTime', ''))[:10],
                     run_seconds=k.get('lastRunExecutionTimeSeconds'), model_families=';'.join(fams),
                     private_inputs=sum(1 for s in srcs if not s), n_inputs=len(srcs), notes=NOTES.get(ref, '')))
d = pd.DataFrame(rows).sort_values(['best_public_score', 'votes'], ascending=[False, False])
assert set(NOTES) - set(d.ref) <= {'dreaddevelopment/knee-mri-training-the-twelve-finding-model'} | set(NOTES), 'unknown refs'
d.to_csv(sys.argv[2], index=False)
print(len(d), 'rows;', (d.notes != '').sum(), 'annotated;', 'missing notes for', sorted(set(NOTES) - set(d.ref)))
print(d.groupby('model_families').best_public_score.agg(['count', 'max']).sort_values('max', ascending=False).head(20).to_string())
