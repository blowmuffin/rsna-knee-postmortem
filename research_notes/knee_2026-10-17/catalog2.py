"""Refreshed catalog of every public notebook (10-17 16:00 UTC listing) -> ../public_code_catalog_2026-10-17.csv
python -X utf8 catalog2.py kernels_web.json copies.csv lb_1601/<leaderboard>.csv ../public_code_catalog.csv ../public_code_catalog_2026-10-17.csv
Same first 11 columns as the 10-16 catalog, then extra columns. Reads only local files."""
import json, sys
import pandas as pd

raw = json.load(open(sys.argv[1]))
cp = pd.read_csv(sys.argv[2]).set_index('ref')
lb = pd.read_csv(sys.argv[3])
old = pd.read_csv(sys.argv[4])

FAM = [  # (substring of an attached source, family tag)  -- extended 10-17
    ('nartaa/rsna-knee-publication-swa', 'raptor-coatnet-nartaa(OAI)'), ('nartaa/rsna-knee-s6fall', 'S6-nartaa'),
    ('heliosli/rsna-knee-r6', 'helios-R6'), ('rostigor/rsna-knee-r6', 'helios-R6'), ('heliosli/rsna-knee-compact-r7', 'helios-R7(fastMRI)'),
    ('varianceofx/rsna-knee-compact-r7', 'helios-R7(fastMRI)'),
    ('heliosli/rsna-knee-compact-v1', 'helios-S6'), ('heliosli/rsna-knee-compact-r6', 'helios-S6'), ('varianceofx/rsna-knee-compact-r6', 'helios-S6'),
    ('goodjane/rsna-knee-compact-r6', 'helios-S6'),
    ('heliosli/rsna-orthofoundation', 'helios-code'), ('goodpjw2008/rsna-knee-2-5d-convnext-reader', 'goodpjw-reader'),
    ('bliverrigpdoge/', 'kcd-private'), ('dreaddevelopment/raptor-knee', 'raptor-coatnet-dread'),
    ('mattiaangeli/knee-mri-fold-weights', 'dino-mattia'), ('pilkwang/rsna-knee-weights', 'dinov2-pilkwang'),
    ('antoinegg1/', 'radimagenet-heads'), ('prvsiyan/rsna-knee-v52', 'radimagenet-heads'),
    ('mattiaangeli/rsna-knee-coat', 'coat-mattia'), ('tonylica/', 'tony-assets'),
    ('marwanmath/resnet-50-radimagenet', 'radimagenet'), ('shigechan/radimagenet', 'radimagenet'),
    ('isotriage/rsna-jev-oai-arm', 'jev-oai-arm(OAI)'), ('metaresearch/dinov2', 'dinov2-meta'),
    ('sofiaanjenje/rsna-knee-e1', 'e11-e13-heads')]

EXT = {  # family tag -> external-data / provenance flag (from dataset cards fetched 10-17, see report)
    'raptor-coatnet-nartaa(OAI)': 'OAI (NDA study 3407; 2,399 knees, masked supervision)',
    'helios-R7(fastMRI)': 'fastMRI aux loss (R7) - must not be used',
    'helios-code': 'SKM-TEA/KneeXNet anatomy anchors (upstream licence retained)',
    'helios-R6': 'SKM-TEA anchors via helios-code; no fastMRI declared (bundle fe055406)',
    'helios-S6': 'SKM-TEA anchors via helios-code; no fastMRI declared',
    'S6-nartaa': 'card: no OAI; SKM-TEA anchors; CC BY-NC 4.0',
    'jev-oai-arm(OAI)': 'OAI arm (undocumented)',
    'radimagenet': 'RadImageNet-pretrained ResNet-50 (public medical-image pretraining)',
    'radimagenet-heads': 'RadImageNet encoder; heads CC BY-NC-SA 4.0 (comp data only per cards)',
    'dino-mattia': 'undocumented training data (CC0 weights)', 'coat-mattia': 'undocumented training data (CC0 weights)',
    'dinov2-pilkwang': 'undocumented training data (CC0 weights)', 'dinov2-meta': 'Meta DINOv2 (Apache-2.0, LVD-142M)',
    'raptor-coatnet-dread': 'competition data only (CC0 weights)', 'goodpjw-reader': 'competition data only (Apache-2.0)',
    'kcd-private': 'competition data only (ImageNet init), dataset CC0 / undocumented'}

REN = {  # old slug -> new slug (renamed by the author; notes carried over)
    'goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-957': 'goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958',
    'heliosli/this-is-best-public-dataset-new-version-0-9576': 'heliosli/this-is-best-public-dataset-new-version-0-9586',
    'goodpjw2008/rsna-knee-coatnet-s6-2-5d-mil-lb-0-954': 'goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957'}

NEW = {
    'goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958':
        "10-17: v25/v26 = 0.958 (heliosli per-finding Raptor weights on R6 flip-TTA + Raptor 384/224 mirror + reader at 1/10); v26 added R7 at 0.4 of the compact slot (same 0.958). "
        "Latest v27 = v25 recipe, compact slot R6 alone from rostigor's hash-checked copy; R7 dataset still attached (metadata). 307 votes = most-voted 0.958. "
        "Own reader alone 0.929; at full weight on the 0.956 base 0.954; at 1/4 no gain; kept at 1/10.",
    'heliosli/this-is-best-public-dataset-new-version-0-9586':
        "10-17 04:21 rerun, renamed from ...-0-9576. V2 = blend-mine + retuned Raptor weights (Lat OA .15, PF OA .65, Synovitis .85, Baker's .30) + goodpjw 2.5D reader (Apache-2.0) at 0.004-0.035, 8.2 h guard. "
        "Mounts the un-suffixed R6 dataset whose latest version holds R7 (since 10-14 18:12): our fork base958v2 pins the -w1 copy. Code similarity 0.87 to heliosli 0-9573.",
    'goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957':
        "10-17 14:17 rerun, renamed from coatnet-s6-2-5d-mil-lb-0-954. v4: R6 + S6 (compact-v1 v13 = bundle 209a32c8) + reader + Raptor OAI = 0.957, i.e. S6 next to R6 costs 0.001 vs 0.958; v6/v7 retuned per-finding Raptor weights on the gold-58 and scored 0.957 / 0.955 vs heliosli's 0.958 (v8 text: do not add S6 to R6, do not tune weights on gold-58). Includes a per-finding gold-58 AUC table of the Raptor half vs R6 flip.",
    'lavinwins/rsna-gold-medal':
        "Until 10-16 this slug held an 'Apex dual-reader' heliosli-R6/R7 + nartaa copy (0.956); the 10-17 01:28 version replaced it and scores 0.958. Different family now: 'v6 combination layer on the 0.943 public pipeline, no new checkpoints': "
        "20 DINOv2-S members + A5 + RadImageNet heads (antoinegg1 CC BY-NC-SA, prvsiyan V52) + dread Raptor x4 + 4-member mattiaangeli CoAt family; no helios R6/R7, no nartaa OAI. "
        "14 datasets + 2 kernel outputs + DINOv2 base/small; visible run 262 s (hidden runtime not published; our test 57315208). Code similarity 0.88 to jiweiliu fast-2xt4.",
    'lavinwins/top-1-gold':
        "Current version (10-17 10:39) is a self-contained TRAINING notebook (competition data only, internet ON for backbone weights; report miner + 2-plane cache + 5-fold ConvNeXt-T 224); "
        "no submission, no datasets. The 0.958 badge is from an earlier version.",
    'kiyoshiohno/rsna-knee-template':
        "10-17 10:36 rerun (0.958, 826 s visible). Thin launcher that executes a versioned private 'logic' dataset (blank source) beside R7 + R6-w1 + nartaa Raptor + orthofoundation: opaque, mounts R7.",
    'toghrulaliyev/rsna-knee-aze-team-dd6d24':
        "New 10-15 (rerun 10-17 13:03, 2898 s): the team's own EDA/training (DINOv2-large + Qwen2.5-7B models, internet-off); no public score, team LB 0.542. Irrelevant.",
    'kaiwalyaatulraut/rsna-knee-competition-solution': "Rerun 10-17 12:38, 0.936. Irrelevant.",
}

notes = dict(zip(old.ref, old.notes.fillna('')))
for o, n in REN.items():
    if o in notes and not notes.get(n): notes[n] = notes[o]
notes.update(NEW)

# leaderboard rank by username
lb = lb.fillna({'TeamMemberUserNames': ''})
best = {}
for _, r in lb.iterrows():
    for u in str(r.TeamMemberUserNames).split(','):
        u = u.strip()
        if u and (u not in best or r.Rank < best[u][0]): best[u] = (int(r.Rank), r.Score)

rows = []
for k in raw:
    ref = k['scriptUrl'].removeprefix('/code/')
    srcs = [s.get('dataSourceUrl') or '' for s in (k.get('dataSources') or [])]
    fams = sorted({f for s in srcs for sub, f in FAM if sub in s})
    ext = '; '.join(EXT[f] for f in fams if f in EXT and f in ('raptor-coatnet-nartaa(OAI)', 'helios-R7(fastMRI)', 'helios-code', 'jev-oai-arm(OAI)', 'S6-nartaa', 'radimagenet', 'radimagenet-heads', 'dino-mattia', 'coat-mattia', 'dinov2-pilkwang', 'dinov2-meta'))
    auth = k['author']['userName']
    rk = best.get(auth)
    c = cp.loc[ref] if ref in cp.index else None
    rows.append(dict(ref=ref, title=k.get('title', ''), best_public_score=k.get('bestPublicScore'), votes=k.get('totalVotes') or 0,
                     created=str(k.get('dateCreated', ''))[:10], last_run=str(k.get('lastRunTime', ''))[:10],
                     run_seconds=k.get('lastRunExecutionTimeSeconds'), model_families=';'.join(fams),
                     private_inputs=sum(1 for s in srcs if not s), n_inputs=len(srcs), notes=notes.get(ref, ''),
                     author=auth, version_utc=str(k.get('scriptVersionDateCreated', ''))[:16], last_run_utc=str(k.get('lastRunTime', ''))[:16],
                     medal=k.get('medal') or '', gpu=k.get('isGpuEnabled'), total_lines=k.get('totalLines'),
                     author_lb_rank=rk[0] if rk else '', author_lb_score=rk[1] if rk else '',
                     external_data=ext, copy_of=(c.copy_of if c is not None and isinstance(c.copy_of, str) else ''),
                     copy_sim=(c.sim if c is not None else ''),
                     clone_cluster_root=(c.cluster_root if c is not None else ''), clone_cluster_size=(c.cluster_size if c is not None else '')))
d = pd.DataFrame(rows).sort_values(['best_public_score', 'votes'], ascending=[False, False])
d.to_csv(sys.argv[5], index=False)
print(len(d), 'rows;', (d.notes != '').sum(), 'annotated;', 'refs in old but not new:', len(set(old.ref) - set(d.ref)), 'new refs:', len(set(d.ref) - set(old.ref)))
