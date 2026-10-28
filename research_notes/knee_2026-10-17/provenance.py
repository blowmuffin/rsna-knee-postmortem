"""Dataset/model provenance table -> dataset_provenance_2026-10-17.csv (cards fetched into ds_meta/ on 10-17; flags are my reading of the cards)."""
import json, glob, os, collections, csv
raw = json.load(open('kernels_web.json'))
use = collections.defaultdict(lambda: [0, 0, 0.0])  # n>=0.955, n>=0.957, best
for k in raw:
    sc = k.get('bestPublicScore') or 0
    for s in k.get('dataSources') or []:
        u = (s.get('dataSourceUrl') or '').removeprefix('/datasets/').removeprefix('/models/').removeprefix('/code/')
        if sc >= 0.955: use[u][0] += 1
        if sc >= 0.957: use[u][1] += 1
        use[u][2] = max(use[u][2], sc)
FLAG = {  # ref -> (training-data declaration as read from card, external-data / rule flag)
 'nartaa/rsna-knee-publication-swa-weights-20261007': ('Card: both checkpoints used the same 2,399 baseline OAI knees for masked external supervision (PF OA, Lateral OA, Synovitis per forum 749858); weights licensed for research/education incl. the competition, adaptation/redistribution non-commercial incl. a CC BY-NC 4.0 winner licence; OAI notice / NDA acknowledgement (NDA study 3407)', 'OAI-trained (controlled access). Disclosure needed. The weights themselves are open to all participants'),
 'nartaa/rsna-knee-s6fall-weights': ('Card: heliosli S6 architecture retrained from timm ImageNet-22k init on JEV-blend report labels, 4,263 competition studies, gold-58 excluded; "No OAI data was used"; trainer and label script published; CC BY-NC 4.0', 'Competition data only + SKM-TEA anchors via the orthofoundation dataset. OAI-free 0.950 single model'),
 'heliosli/rsna-orthofoundation-fold0-epoch7-20261004': ('Card: private inference bundle; OrthoFoundation fold-0 epoch-7 EMA weights, project inference source and unchanged SKM-TEA/KneeXNet auxiliary weights; upstream licences retained, no new licence granted', 'SKM-TEA (anatomy.pt places the crops for R6/R7/S6). The OrthoFoundation ViT (OAI-derived per forum 749858) is not loaded by the R6/R7/S6 notebooks. Licence "other"'),
 'heliosli/rsna-knee-r6-frozen-fe055406': ('Card: version 2 = R7 bundle 8c7b666e (since 10-14 18:12); version 1 = R6 fe055406fad6a12f. No images/reports/labels inside', 'UN-SUFFIXED SLUG NOW HOLDS R7 (fastMRI). Never mount without a bundle_id assertion'),
 'heliosli/rsna-knee-r6-frozen-fe055406-w1': ('Card: private immutable copy of R6 bundle fe055406fad6a12f, byte-identical to version 1 of the un-suffixed dataset (manifest sha256 6508ce69...). No images/reports/labels', 'R6: no fm_aux entry in the manifest (forum 749858); SKM-TEA anchors. Stacker metadata names the run dir cnx_r6fm (fm = flip-mirror or fastMRI: unverified)'),
 'rostigor/rsna-knee-r6-fe055406': ('Card: versioned copy of R6 version 1, bundle fe055406fad6a12f, manifest sha256 6508ce69...', 'Same R6 bytes as -w1 (independent copy = redundancy)'),
 'heliosli/rsna-knee-compact-r7cnxfsss': ('Card: private R7 ConvNeXt four-fold bundle with refitted Effusion and Baker state stackers. "No KneeCoT"', 'R7 = fastMRI+ boxes as auxiliary task (fm_aux 0.3 in manifest and in heliosli notebook). NEVER USE'),
 'varianceofx/rsna-knee-compact-r7cnxfsss': ('Mirror of heliosli R7 v1 (bundle 8c7b666e27dc0e8f); "Not for redistribution"', 'R7 / fastMRI. NEVER USE'),
 'heliosli/rsna-knee-compact-v1': ('Card: private inference bundle, compact model fine-tuned on report-derived targets (13 versions; latest v13 restores v10 = bundle 209a32c8b50a6b1c cnx_s6)', 'S6: nothing external found (forum 749858); SKM-TEA anchors'),
 'heliosli/rsna-knee-compact-r6cnxfsss': ('Card: same description as compact-v1 (bundle 209a32c8 = S6)', 'S6 (the name r6cnxfsss is misleading: it is the S6 bundle)'),
 'varianceofx/rsna-knee-compact-r6cnxfsss': ('Unmodified mirror of heliosli r6cnxfsss v2 (bundle 209a32c8b50a6b1c)', 'S6'),
 'goodjane/rsna-knee-compact-r6cnxfsss': ('No description; licence field says apache-2.0 (a mirror: the uploader did not create the model)', 'S6 mirror; licence claim unreliable'),
 'goodpjw2008/rsna-knee-2-5d-convnext-reader': ('Card + train.py / make_labels.py read: ConvNeXt-T 2.5D, 3 of 5 folds, EMA, report-derived soft labels = mean of four public LLM label tables, gold-58 not used; Apache-2.0', 'Competition data + public LLM label tables only. Clean'),
 'dreaddevelopment/raptor-knee-widedense': ('Card: two CoAtNet-rmlp-2 384 checkpoints, 0.924 single model; CC0', 'Competition data only (no external data stated)'),
 'dreaddevelopment/raptor-knee-maxspan': ('Card: CoAtNet 384, 64 slices over 2-98%, 140 mm crop, 0.928 single; CC0', 'Competition data only'),
 'dreaddevelopment/raptor-knee-native384': ('Card: CoAtNet 384, native 384 corpus, 0.926; CC0', 'Competition data only'),
 'dreaddevelopment/raptor-knee-native384dense': ('Card: CoAtNet 384, 64 slices, native 384, gold-58 0.9174, LB untested at card time; CC0', 'Competition data only'),
 'dreaddevelopment/raptor-knee-arms': ('Card: ConvNeXtV2-B 336 + CoAtNet 384 + EffNetV2-L 480 arms, rank-mean 0.915; CC0', 'Competition data only'),
 'dreaddevelopment/raptor-knee-arms-x': ('Card: single CoAtNet 384 arm, 44 slices, 0.917; CC0', 'Competition data only'),
 'mattiaangeli/knee-mri-fold-weights': ('No description; CC0-1.0', 'UNDOCUMENTED training data (DINOv2 folds)'),
 'mattiaangeli/rsna-knee-coat-resgated-ep10-top3': ('No description; CC0-1.0', 'UNDOCUMENTED training data (CoAtNet)'),
 'mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2': ('No description; CC0-1.0', 'UNDOCUMENTED training data (CoAtNet D4)'),
 'mattiaangeli/rsna-knee-coatnet-global96-top3': ('No description; CC0-1.0', 'UNDOCUMENTED training data (CoAtNet Global96)'),
 'mattiaangeli/opencv-python-headless-4120088-x86': ('Pinned manylinux OpenCV wheel; licence unknown', 'Code wheel only'),
 'pilkwang/rsna-knee-weights': ('No description; CC0-1.0', 'UNDOCUMENTED training data (DINOv2-S folds)'),
 'pilkwang/rsna-knee-llm-labels': ('Card: LLM-read report labels, twelve verdict/severity pairs per study; CC0', 'Labels from competition reports (LLM-API labels explicitly allowed by the host)'),
 'antoinegg1/rsna-knee-e11-diverse-heads-v20': ('No description; CC-BY-NC-SA-4.0', 'NC share-alike heads on RadImageNet descriptors; training data undocumented'),
 'antoinegg1/rsna-knee-e9-radimagenet-heads-v15': ('No description; CC-BY-NC-SA-4.0', 'NC share-alike heads; undocumented'),
 'prvsiyan/rsna-knee-v52-radimagenet-heads-20260812': ('Card: five frozen heads on 24 RadImageNet ResNet-50 slice descriptors, trained only on the RSNA Knee training split per README; CC-BY-NC-SA-4.0', 'NC share-alike heads; RadImageNet encoder upstream licence not named'),
 'marwanmath/resnet-50-radimagenet-marwan': ('No description; CC-BY-NC-SA-4.0', 'RadImageNet ResNet-50 copy (NC share-alike on this copy)'),
 'shigechan/radimagenet-resnet50-official': ('Card: exact public RadImageNet ResNet-50 checkpoint (Lab-Rasool/RadImageNet on HF); "Original model is MIT licensed"', 'RadImageNet medical-image pretraining (public)'),
 'bliverrigpdoge/rsna-knee-leg-r9all': ('No description; CC0-1.0', 'Competition-data model (KCD / leg crops); training data not stated'),
 'isotriage/rsna-jev-oai-arm': ('No description; licence other', 'OAI arm by name: AVOID (undocumented)'),
 'glorysing/rsna-knee-offline-codecs-r09': ('Offline PyPI decoder wheels; upstream licences retained', 'Code only'),
 'beraterolelk/istanbul-real-estate-2024-anonymized': ('Unrelated real-estate CSV attached to a 0.956 notebook', 'No relevance'),
 'dinov2 (Kaggle Model metaresearch/dinov2 small/base/large/giant)': ('Model card: Meta DINOv2; instances small/base/large/giant; licence Apache 2.0; training data = facebookresearch/dinov2 (LVD-142M)', 'Public general-image pretraining. Clean'),
 'sofiaanjenje/rsna-knee-e11-train, e13-train (kernel outputs)': ('Notebook outputs used as weights (DINOv2 heads trained on competition data; log read 10-16)', 'Competition data; licence not stated'),
}
rows = []
for f in sorted(glob.glob('ds_meta/*.json')):
    m = json.load(open(f, encoding='utf8')).get('info', {})
    ref = os.path.basename(f)[:-5].replace('__', '/')
    d, e = FLAG.get(ref, ('', ''))
    u = use.get(ref, [0, 0, 0.0])
    rows.append([ref, m.get('title'), ','.join(l.get('name') for l in m.get('licenses', [])), m.get('totalDownloads'), m.get('totalVotes'), u[0], u[1], u[2], d, e])
for ref in ['dinov2 (Kaggle Model metaresearch/dinov2 small/base/large/giant)', 'sofiaanjenje/rsna-knee-e11-train, e13-train (kernel outputs)']:
    d, e = FLAG[ref]; rows.append([ref, '', 'Apache 2.0' if ref.startswith('dinov2') else 'n/a', '', '', '', '', '', d, e])
rows.sort(key=lambda r: -(r[5] if isinstance(r[5], int) else 0))
with open('dataset_provenance_2026-10-17.csv', 'w', newline='', encoding='utf8') as fh:
    w = csv.writer(fh); w.writerow(['dataset', 'title', 'licence_on_card', 'downloads', 'votes', 'n_notebooks_ge_0.955', 'n_notebooks_ge_0.957', 'best_notebook_score', 'training_data_declaration_on_card', 'rule_flag'])
    w.writerows(rows)
print(len(rows), 'rows;', sum(1 for r in rows if not r[8]), 'without annotation')
print([r[0] for r in rows if not r[8]])
for r in rows[:14]: print(r[0], r[5], r[6], r[7])
