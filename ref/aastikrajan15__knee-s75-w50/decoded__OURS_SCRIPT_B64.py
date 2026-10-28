
import os, sys, glob, time, math
import numpy as np, pandas as pd, cv2, pydicom, torch
from concurrent.futures import ThreadPoolExecutor

# <<<S57_RESOLVE_BEGIN>>> ONE source of truth. src/push_s57_probe.py slices this exact text
# out of CELL_IMPORTS and embeds it in the probe kernel, so what is measured on Kaggle is
# literally what ships. Do not copy it anywhere; slice it.
def _resolve_inputs(comp_slug, ckpt_slug, base='/kaggle/input'):
    '''Return (ROOT, CKPT_DIR, ck_list, how_root, how_ckpt).

    S57. The two recursive globs this replaces cost 262.463 s + 97.790 s = 360.253 s of a
    368.901 s pre-work window, MEASURED on Kaggle 2026-09-10 -- 97.7% of everything that
    happens before the first DICOM is read (docs/S56-374S-ATTRIBUTED-2026-09-10.md). Both
    mount slugs are already known at push time (mkkernel.py writes them into
    competition_sources / dataset_sources), so both paths are directly constructible and the
    241k-file competition tree need not be walked at all.

    THE GLOBS ARE THE FALLBACK, NOT DEAD CODE. Every defence in them survives:
      * the two direct test.csv candidates are BOTH under the competition mount, so a stray
        test.csv inside an attached dataset cannot win -- STRICTER than the glob, which only
        PREFERS a /competitions/ path (round-1 defect 4: the wrong tree scores exactly 0.500);
      * *_best.pt only -- train.py also writes f{fold}_last.pt, which carries no 'args'
        (round-1 defect 6);
      * a renamed, re-slugged or re-laid-out mount is NOT fatal. It falls through to the
        original recursive glob and costs the original ~360 s. That path was broken on
        purpose once, on Kaggle -- docs/S57-GLOB-NARROWED-2026-09-11.md.
    '''
    import os, glob
    root, how_root = None, 'glob'
    for c in (base + '/competitions/' + comp_slug + '/test.csv',
              base + '/' + comp_slug + '/test.csv'):
        if os.path.exists(c):
            root, how_root = os.path.dirname(c), 'direct'
            break
    if root is None:
        cands = [p for p in glob.glob(base + '/**/test.csv', recursive=True)
                 if '/competitions/' in p.replace(os.sep, '/')]
        cands = cands or glob.glob(base + '/**/test.csv', recursive=True)
        assert cands, 'test.csv not found under ' + base
        root = os.path.dirname(cands[0])

    ck, how_ck = [], 'glob'
    # MEASURED on Kaggle 2026-09-11 by knee-s57-probe-glob v1: a dataset mounts at
    # /kaggle/input/datasets/<owner>/<slug>, NOT /kaggle/input/<slug>. v1 assumed the latter,
    # missed, and paid 247.89 s for the fallback glob -- which is how the real layout got
    # measured instead of guessed. `ckpt_slug` is therefore the FULL owner/slug.
    _short = ckpt_slug.split('/')[-1]
    for d in (base + '/datasets/' + ckpt_slug, base + '/' + _short, base + '/' + ckpt_slug):
        if not os.path.isdir(d):
            continue
        # depth <= 3 under the NAMED checkpoint mount only. Keeps 'do not assume the internal
        # layout' without walking the competition tree, and is narrower than the old glob,
        # which took the first sorted dir of ANY mount (STATE trap: a second mounted
        # checkpoint dataset REPLACES the fleet).
        for pat in ('/*_best.pt', '/*/*_best.pt', '/*/*/*_best.pt'):
            ck = sorted(glob.glob(d + pat))
            if ck:
                how_ck = 'direct'
                break
        if ck:
            break
    if not ck:
        ck = sorted(glob.glob(base + '/**/*_best.pt', recursive=True))
        assert ck, 'no *_best.pt checkpoint found under ' + base
    return root, os.path.dirname(ck[0]), ck, how_root, how_ck
# <<<S57_RESOLVE_END>>>

# Mount slugs, as mkkernel.py writes them into the kernel metadata at push time
# (COMP, and the dataset_sources entry). A wrong value here is not fatal: it falls back.
COMP_SLUG = 'rsna-knee-abnormality-detection'
CKPT_SLUG = 'aastikrajan15/knee-ckpt'
ROOT, CKPT_DIR, _ck, _HOW_ROOT, _HOW_CKPT = _resolve_inputs(COMP_SLUG, CKPT_SLUG)
print('S57 resolve: test.csv via', _HOW_ROOT, ROOT, '| ckpt via', _HOW_CKPT, CKPT_DIR,
      '|', len(_ck), 'checkpoints', flush=True)
# Optional subset of the staged checkpoints, by basename. The dataset is ~2 GB, so
# uploading a new version per candidate ensemble is slow; this lets several DIFFERENT
# ensembles be submitted against ONE dataset version. None = use everything staged.
ONLY = {'f0_convnext_tiny_320x24_pt_v5_cv9084_best.pt', 'f0_convnext_tiny_320x24_pt_v5_cv9088_best.pt', 'f1_convnext_tiny_320x24_pt_v5_cv9126_best.pt', 'f1_convnext_tiny_320x24_pt_v5_cv9154_best.pt', 'f2_convnext_tiny_320x24_pt_v5_cv9006_best.pt', 'f2_convnext_tiny_320x24_pt_v5_cv9031_best.pt', 'f3_convnext_tiny_320x24_pt_v5_cv8961_best.pt', 'f3_convnext_tiny_320x24_pt_v5_cv8991_best.pt', 'f4_convnext_tiny_320x24_pt_v5_cv9036_best.pt', 'f4_convnext_tiny_320x24_pt_v5_cv9058_best.pt'}
# PER-TARGET LEADERBOARD PROBE. macro-AUC is a MEAN over the 12 targets, so it decomposes:
# report real predictions for a subset and a CONSTANT 0.5 for the rest, and the LB reads
#     LB = ( (12-k)*0.5 + sum of the k real AUCs ) / 12
# because a constant column is all ties and scores exactly 0.5 -- VERIFIED on this competition
# by submission 55286582 ("pipeline check: constant 0.5 baseline"), which scored exactly 0.500.
# With k=4 that inverts to  mean AUC over the 4 = 3*LB - 1.
# WHY: every per-target number this project acts on is measured either OOF-vs-LABEL (a map of
# label noise, not model weakness) or on the enriched 58-study gold set where only 2 of 12
# targets clear a bootstrap. The public test set is ~7x larger AND the right distribution.
# None = normal submission, byte-identical to before.
PROBE_TARGETS = None
# PER-TARGET ENSEMBLE WEIGHTS. macro-AUC is the MEAN OF 12 SEPARATE RANKINGS -- target A's
# scores are never compared with target B's -- so each target may use its own mixture of
# members at zero cost to the others. We have always used ONE flat mean for all twelve.
# MEASURED out-of-sample on 84 on-disk OOF files with a leak-free split-half protocol
# (fit on half the studies, score the other half, 50 repeats x 2 directions x 5 folds):
#     flat mean            0.8988   optimism +0.0000  <- the protocol's own sanity check
#     per-target top-3     0.9022   optimism +0.0041
#     per-target greedy    0.9024   optimism +0.0078
# positive in 5/5 folds and 493/500 split-half units. The gain is concentrated in the
# targets we are worst at: MCL +0.0163, Synovitis +0.0068, ACL +0.0057 = 66% of the total.
# Format: {checkpoint_basename: [w0..w11]}, non-negative, any scale (renormalised per
# study x target over the members that actually produced a prediction). None = flat mean,
# byte-identical to every submission before this existed.
TARGET_WEIGHTS = None
SRC_SLICES = 32          # must equal N_SLICES in preproc_cells.py
STORE_RES  = 512         # must equal RES in preproc_cells.py
JPEG_Q     = 92
CROP_MM    = 130.0       # must equal crop_mm in dataset.py
TTA_OFFSETS = [0.0]
TTA_FULL = [-0.5, 0.0, 0.5]
TIME_BUDGET_S = 7.5 * 3600      # hard limit is 9 h; leave room for startup and writing
VIEWS   = [('Axial', 1), ('Sagittal', 0), ('Coronal', 1), ('Sagittal', 1)]
# must mirror dataset.VIEWS5. Models trained with --views5 read a 5th slot; a mixed
# ensemble decodes the union once and each model takes the ordered subset it was
# trained on.
VIEWS5  = VIEWS + [('Coronal', 0)]
TARGETS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
           'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

# How to pool the TTA passes WITHIN one model, per target. None = plain mean for all
# twelve, which is bit-identical to every submission made so far.
#
# Why a per-target pool is not just a knob: macro-AUC forbids a per-target MONOTONE
# transform of one model's score, but max/top-k OVER PASSES is not monotone in any single
# pass, so it CAN move AUC. And the dilution argument is measured -- our slice-attention
# pool attends 23.80 of 24 slices, so focal evidence is already averaged away once; a mean
# over offset windows that mostly MISS a 2-slice fracture line does it again.
#
# The split below is the public frontier's, which its author credits with 0.891 -> 0.894 on
# this competition: max for the focal findings, top-2 for the two that are focal but noisy
# enough that a single window is unreliable, and mean left alone for Synovitis (Hoffa fat
# pad / effusion-synovitis are diffuse) and for the OA compartments and Effusion, which are
# whole-volume properties.
# S39-B: this is NO LONGER the public dict. Measured on our own saved 3-offset arrays
# (work/oof/tta3_f{0,1,2}*.npy, n=2,646 over 3 folds) the public version is WRONG on two
# targets -- Contusion -0.0010 (P=.02) and Baker's -0.0009 (P=.04), both listed 'max'
# publicly. Their apparent focality is a probability artefact: the raw pos/neg spread ratio
# reads 1.82 for Contusion, but binned on predicted probability it collapses to 0.95.
# Medial Meniscus dropped too (+0.0001, prevalence .42, host needs surface contact on >=2
# images -- not a one-slice finding). Kept entries carry their measured delta.
TTA_POOL_FOCAL = {'Lateral Meniscus': 'max',   # +0.0010 P=.92, most focal (13.2/24 attended)
                  'Fracture': 'max',           # +0.0008 P=.83, 1-2 px line on T1, ~2 slices
                  'ACL': 'top2',               # +0.0005, focal but attention spends 19.2/24
                  'MCL': 'top2'}               # +0.0005, largest matched excess, thinnest n
TTA_TARGET_POOL = None
# Mirror TTA is a SEPARATE cost from the slice offsets, and a separate question. Non-lean
# turns both on at once (see where _mirror is assigned below), so a 'fat' accuracy entry
# costs 6x the fleet's inference rather than 3x -- and the kernel is KILLED at 9 h.
# Mirror is also the one TTA pass with a positive argument AGAINST it: dataset.py
# canonicalises every study to a right-knee orientation precisely so that medial/lateral
# is a FIXED IMAGE PROPERTY, and averaging a study with its mirror blurs exactly that cue
# -- on Medial/Lateral OA, the two targets that depend on it most.
# Default True keeps every existing kernel bit-identical.
TTA_MIRROR = True

# Kaggle hands out a Tesla P100 (sm_60) unless kernel-metadata sets
# machine_shape="NvidiaTeslaT4", and the image's torch carries sm_70+ only -- so
# torch.cuda.is_available() is True and every op then raises "no kernel image is
# available". That is why five submissions ran on CPU with TTA silently off. The metadata
# is fixed in mkkernel.py, but probe with a real matmul anyway: a wrong answer here costs
# a whole submission.
dev = 'cpu'
if torch.cuda.is_available():
    print('torch', torch.__version__, '| device', torch.cuda.get_device_name(0),
          '| capability', torch.cuda.get_device_capability(0),
          '| built for', torch.cuda.get_arch_list(), flush=True)
    try:
        _t = torch.randn(8, 8, device='cuda') @ torch.randn(8, 8, device='cuda')
        float(_t.sum())
        dev = 'cuda'
        print('GPU usable', flush=True)
    except Exception as e:
        print('GPU UNUSABLE, falling back to CPU:', str(e)[:120], flush=True)
        torch.set_num_threads(os.cpu_count() or 4)
print('device =', dev, flush=True)
# LEAN_MODE is the EFFICIENCY-PRIZE entry, and it is a separate prize with its own money.
# VERIFIED off the efficiency leaderboard 2026-08-13
# (kaggle kernels output ryanholbrook/rsna-knee-abnormalities-efficiency-lb):
#     rank 1   Siontist              0.928
#     rank 3   YassY_The_AlchemYst   0.928
#     rank 772 ourselves            0.928
# IDENTICAL accuracy to rank 1 -- all 771 places are RUNTIME. The metric is
# AUC/(Benchmark - maxAUC) + seconds/32400, minimised, and the denominator is negative, so
# 0.001 AUC is worth ~72 s. Our accuracy entry runs 5+ hours of the 9 h cap; a one-model,
# no-TTA entry runs well under one. Turning TTA on automatically here is therefore the wrong
# default for THAT submission -- and since a kernel cannot know which of the two selected
# submissions it is, it has to be told.
os.environ['KNEE_LEAN'] = '1'   # injected by cells(lean=True)
LEAN_MODE = bool(int(os.environ.get('KNEE_LEAN', '0')))
if dev == 'cuda' and not LEAN_MODE:
    TTA_OFFSETS = TTA_FULL
if LEAN_MODE:
    # mirror is switched off where _mirror is ASSIGNED, during checkpoint load -- GROUPS is
    # still empty here, so looping over it now would be a silent no-op.
    TTA_OFFSETS = [0.0]
    print('LEAN MODE: single-pass, no offset TTA, no mirror -- efficiency entry', flush=True)
print('TTA offsets', TTA_OFFSETS, flush=True)

test = pd.read_csv(f'{ROOT}/test.csv')
ts   = pd.read_csv(f'{ROOT}/test_series.csv')
ssub = pd.read_csv(f'{ROOT}/sample_submission.csv')
assert list(ssub.columns) == ['StudyInstanceUID'] + TARGETS, \
    f'column mismatch: {list(ssub.columns)}'
IMG_DIR = f'{ROOT}/test_series'
assert os.path.isdir(IMG_DIR), f'no image dir at {IMG_DIR}'
print('ROOT', ROOT, '| studies', len(test), '| series', len(ts), flush=True)



sys.path.insert(0, CKPT_DIR)
# model.py rides along in the checkpoint dataset and has silently gone stale before.
# A --per_target checkpoint against the old file raises TypeError and kills the run.
_msrc = open(f'{CKPT_DIR}/model.py').read()
assert 'MultiAttnPool' in _msrc and 'per_target' in _msrc, \
    'stale model.py in the checkpoint dataset -- re-upload src/model.py'
from model import KneeNet
# S51: the 3D arm. model3d.py rides along in the checkpoint dataset the same way model.py
# does. Imported LAZILY and guarded, because a dataset version staged before S51 has no
# model3d.py and every 2D-only submission must keep working against it -- an unguarded
# import would turn a missing file into a dead kernel for the 0.942 path. A 3D checkpoint
# present WITHOUT the module is a hard error, raised at the checkpoint (see _NO3D below),
# never a silent skip.
try:
    from model3d import Knee3DNet
    _NO3D = ''
except Exception as _e:                                          # noqa: BLE001
    Knee3DNet, _NO3D = None, f'{type(_e).__name__}: {_e}'
# dataset.py rides along for exactly one function: the TE/TR -> T1/PD/T2 rule that
# --contrast_emb models were trained against. Importing it (rather than re-deriving the
# thresholds here) is what stops the train and test rules drifting apart silently.
# FG_MIN/TARGET_P80 ride along for the same reason: the --norm_p80 landmark gain has to be
# the SAME arithmetic on both sides, and re-typing 10 and 110.0 here is exactly how the two
# rules drift apart. Same import-don't-redeclare rule as contrast_class.
from dataset import contrast_class, N_CONTRAST, FG_MIN, TARGET_P80

IMNET_M = torch.tensor([0.485,0.456,0.406]).view(1,1,3,1,1).to(dev)
IMNET_S = torch.tensor([0.229,0.224,0.225]).view(1,1,3,1,1).to(dev)
AMP = (dev == 'cuda')
def norm(x, gain=1.0):
    # `gain` is the norm_p80 landmark gain, applied HERE in float only for --norm_float
    # checkpoints. Legacy --norm_p80 checkpoints get it multiplied into the uint8 inside
    # sample_vol, because that is what they trained on. Default 1.0 = no-op, exact.
    x = x.float().div_(255.)
    if gain != 1.0:
        x = x * gain
    return (x - IMNET_M) / IMNET_S

# Channel-reversed ImageNet constants, for the sagittal mirror. See the call site.
IMNET_M_R = IMNET_M.flip(dims=[2])
IMNET_S_R = IMNET_S.flip(dims=[2])


# Sagittal mirror of an ALREADY-NORMALISED (B,S,C,H,W) tensor.
# The mirror reverses the slice axis and, with it, the 2.5D channel triplet. Training does that
# on the RAW uint8 volume (dataset.py:331) and normalises afterwards; here the tensor is already
# normalised, so a bare flip leaves channel c holding data scaled by mean[2-c]/std[2-c].
# Undo and reapply:  raw/255 = f*S_rev + M_rev,  then re-normalise with this channel's own
# constants. Exact, and needs no extra DICOM decode.
# NOTE: a ''' or triple-double docstring here would TERMINATE the enclosing CELL_MODEL r-string.
def _fwd(m, views, present, ctr, is3d):
    # THE CALL THAT BURNED A SLOT, 2026-09-08. `KneeNet.forward(views, present, ctr=None)`
    # takes three positional arguments; `Knee3DNet.forward(views, present=None, **_)` takes
    # TWO -- and `**_` collects only KEYWORD arguments, so it does not absorb a third
    # positional one. `m(vs, pr, ct)` therefore raised
    #     Knee3DNet.forward() takes from 2 to 3 positional arguments but 4 were given
    # for EVERY study. The per-study try/except caught it, the fallback wrote 0.5, and the
    # kernel exited 0 with a perfectly valid submission.csv of constant 0.5 -- the exact
    # "reads like success" failure this file already documents for a wrong image path.
    #
    # The 3D arm genuinely has no contrast embedding, so not passing `ctr` is also the
    # semantically right thing, not just the arity-safe one.
    return m(views, present) if is3d else m(views, present, ctr)


def _head(out):
    # The main (B, 12) logits, whichever model produced them.
    #
    # S51. KneeNet.forward ALWAYS returns a tuple, so this file has always written
    # m(...)[0]. Knee3DNet.forward returns (main, aux) only when it was built with
    # n_aux>0, and a BARE TENSOR otherwise (model3d.py) -- and tensor[0] is a VALID
    # index that silently returns row 0 of the batch, shape (12,) instead of (1, 12).
    # It raises nothing, it broadcasts, and the submission would be built from one
    # study's logits applied to every study. The shape assert below is what turns that
    # class of mistake from a plausible score into a stack trace.
    t = out[0] if isinstance(out, (tuple, list)) else out
    assert t.dim() == 2 and t.shape[1] == 12, (
        f'model head returned {tuple(t.shape)}, expected (B, 12) -- a bare-tensor return '
        f'was indexed as a tuple, or the head width is wrong')
    return t


def _mirror_sag(v):
    f = v.flip(dims=[1, 2])
    return (f * IMNET_S_R + IMNET_M_R - IMNET_M) / IMNET_S

# resolved once, so the run cell does no string work per study
_POOL_IDX = ({} if not TTA_TARGET_POOL else
             {TARGETS.index(t): m for t, m in TTA_TARGET_POOL.items() if t in TARGETS})
if TTA_TARGET_POOL:
    _bad = [t for t in TTA_TARGET_POOL if t not in TARGETS]
    assert not _bad, f'TTA_TARGET_POOL names unknown targets: {_bad}'
    print('per-target TTA pool:', TTA_TARGET_POOL, flush=True)

def tta_pool(P):
    # P: (n_pass, 12) of probabilities from ONE model -> (12,).
    # Mean everywhere unless a target asks for max / top-k. topk clamps to n_pass, so this
    # stays safe when the time valve has cut TTA to a single offset: every mode then
    # collapses to the same single value and the pooling becomes a no-op, which is the
    # correct degradation rather than an error.
    out = P.mean(0)
    if not _POOL_IDX:
        return out
    out = out.clone()
    n = P.shape[0]
    for j, mode in _POOL_IDX.items():
        if mode == 'max':
            out[j] = P[:, j].max()
        elif mode.startswith('top'):
            k = min(int(mode[3:] or 2), n)
            out[j] = P[:, j].topk(k).values.mean()
    return out

# Models are grouped by the input they need, so a 320/24 fold and a 384/32 fold can share
# one decode pass. GROUPS key, and it must list EVERY input-affecting flag or a model gets
# served a tensor it did not train on:
#   (res, slices, n_views, norm_p80, crop_xy, crop_clamp, norm_float, crop_mm) -> [model...]
GROUPS = {}
# parallel to GROUPS, same keys, same order -- recovers PER_MODEL index -> checkpoint name
GROUP_NAMES = {}
_seen = []


# Round-robin the checkpoints across folds instead of loading f0_*, then f1_*, ...
#
# S16: the model-drop valve keeps a PREFIX of the load order and its comment claims that
# prefix is fold-balanced. It was not. Load order was sorted(glob(...)), i.e. strictly
# alphabetical on names that all start with f{fold}_, so the prefix was every fold-0
# member, then every fold-1 member. If the valve had ever fired on a large ensemble it
# would have collapsed it onto one or two folds -- the exact opposite of the breadth the
# ensemble is built for, and the failure would have been invisible in the log.
# A flat mean is order-invariant, so interleaving changes nothing unless the valve fires.
def _fold_interleaved(paths):
    byfold = {}
    for q in paths:
        b = os.path.basename(q)
        f = b[1] if (b[:1] == 'f' and b[1:2].isdigit()) else '9'
        byfold.setdefault(f, []).append(q)
    out, i = [], 0
    while any(len(v) > i for v in byfold.values()):
        for f in sorted(byfold):
            if i < len(byfold[f]):
                out.append(byfold[f][i])
        i += 1
    return out


for p in _fold_interleaved(sorted(glob.glob(f'{CKPT_DIR}/*_best.pt'))):
    if ONLY is not None and os.path.basename(p) not in ONLY:
        continue
    _seen.append(os.path.basename(p))
    ck = torch.load(p, map_location='cpu', weights_only=False)
    a = ck['args']
    sd = ck['model']
    # `torch.compile(self.cnn)` returns an OptimizedModule, so a run launched with
    # KNEE_COMPILE=1 saves its backbone under `...cnn._orig_mod...` while the model object
    # built here is UNCOMPILED and expects `...cnn...`. Measured on ckptS22_F0 / ckptS22_F2:
    # 1080 of 1158 tensors carry the prefix, which lands as 1080 missing AND 1080 unexpected
    # keys. The asserts below do catch it -- loudly, not silently -- but that means those
    # three S22 checkpoints could not be SERVED AT ALL, while docs/STATE.md counts two of
    # them as ensemble additions worth +0.0022/+0.0026. Stripping the prefix here makes them
    # load with missing=0, unexpected=0 (verified S25 by set-differencing the key names
    # against an uncompiled checkpoint of the same architecture).
    # This is a rename only -- it cannot change any tensor -- and it is a no-op for every
    # checkpoint trained without compile. The ROOT fix belongs in train.py, which should save
    # the uncompiled state_dict; that file could not be touched while a run was live.
    if any('._orig_mod.' in k for k in sd):
        sd = {k.replace('._orig_mod.', '.'): v for k, v in sd.items()}
    # Laterality canonicalisation was REJECTED (CV 0.8786 / LB 0.909 vs 0.8805 / 0.915)
    # and a checkpoint trained with it but inferred without scored 0.519. Checkpoints
    # predating the flag carry no 'lat_csv' key, so `.get('lat_csv','')` cannot tell a
    # non-laterality run from an unknown one. Fail closed.
    # S51 THE 3D ARM. train3d.py writes ck['arch'] = 'Knee3DNet'; nothing else does, so the
    # discriminator is EXPLICIT and does not guess from the backbone name. Everything below
    # this block -- lat_csv, ZERO_OK, store_res, the KneeNet constructor -- is 2D-only, and
    # train3d.py's args namespace carries NONE of those keys, so the 2D asserts would fire
    # on a perfectly good 3D checkpoint. Branch before them, not inside them.
    _is3d = (ck.get('arch') == 'Knee3DNet')
    if _is3d:
        assert Knee3DNet is not None, (
            f'{os.path.basename(p)} is a 3D checkpoint but model3d.py is not in '
            f'{CKPT_DIR} -- re-version the checkpoint dataset with it. Import said: {_NO3D}')
        # Knee3DNet's aux head is `aux_head.*`, not `aux.*` as in model.py. Reading the
        # 2D key here would silently return n_aux=0 for an aux-trained 3D model and change
        # the head's output width -- so read the 3D name, and read it off the saved tensors
        # rather than the args, exactly as the 2D path does.
        _n_aux3 = int(sd['aux_head.weight'].shape[0]) if 'aux_head.weight' in sd else 0
        views = VIEWS5 if a.get('views5') else VIEWS
        m = Knee3DNet(backbone=a['backbone'], n_targets=12, n_views=len(views),
                      res3d=int(a.get('res3d', 224)), pretrained=False,
                      per_target=bool(a.get('per_target', False)),
                      vol_mode=a.get('vol_mode', 'center'), dropout=0.0,
                      n_aux=_n_aux3, grad_ckpt=False,
                      n_slices=int(a.get('slices', 24)), chunk=int(a.get('chunk3d', 0) or 0))
        _miss3, _unexp3 = m.load_state_dict(sd, strict=False)
        assert not _miss3, f'{os.path.basename(p)} missing 3D weights: {list(_miss3)[:5]}'
        assert not _unexp3, f'{os.path.basename(p)} unexpected 3D weights: {list(_unexp3)[:5]}'
        m = m.eval().to(dev)
        # The 3D path is the ONE place where the naive flip is CORRECT. The S19 bug was
        # mirroring AFTER normalisation, which permutes the 3 channels against the wrong
        # per-channel constants. Knee3DNet normalises INSIDE forward(), so the tensor this
        # kernel flips is still raw uint8 -- bit-identical to dataset.py:331's
        # `vol[::-1, ::-1]`. _mirror_sag must NOT be used here; it assumes normalised input.
        m._mirror = ((float(a.get('mirror_aug', 0) or 0) > 0)
                     and not LEAN_MODE and TTA_MIRROR)
        m._is3d = True
        # 3D never shares a decoded group with 2D: the tensor handed to a 3D model is raw
        # uint8 and the one handed to a 2D model is ImageNet-normalised float. The flag is
        # therefore part of the group KEY, not a per-model attribute, or one group would
        # have to be normalised two ways at once.
        _gkey = (int(a['res']), int(a['slices']), len(views),
                 bool(a.get('norm_p80', False)), bool(a.get('crop_xy', False)),
                 bool(a.get('crop_clamp', False)), bool(a.get('norm_float', False)),
                 float(a.get('crop_mm') or CROP_MM), True)
        GROUPS.setdefault(_gkey, []).append(m)
        GROUP_NAMES.setdefault(_gkey, []).append(os.path.basename(p))
        print('loaded', os.path.basename(p), '| 3D', a['backbone'],
              f"{a['res']}/{a['slices']}/{len(views)}v res3d {a.get('res3d', 224)}",
              '| aux', _n_aux3, '| cv', round(ck.get('cv', float('nan')), 4), flush=True)
        continue
    assert 'lat_csv' in a, (f'{os.path.basename(p)} predates the lat_csv flag -- '
                            'laterality unknown, refusing to guess')
    assert not a['lat_csv'], f'{os.path.basename(p)} is laterality-canonicalised (rejected)'
    # n_aux is not in args; read it off the saved head so a mismatch cannot happen
    n_aux = int(sd['aux.weight'].shape[0]) if 'aux.weight' in sd else 0
    views = VIEWS5 if a.get('views5') else VIEWS
    m = KneeNet(a['backbone'], n_aux=n_aux, pretrained=False, chunk=32,
                per_target=bool(a.get('per_target', False)),
                spatial=bool(a.get('spatial', False)), n_views=len(views),
                slice_pos=bool(a.get('slice_pos', False)),
                pool_prior=bool(a.get('pool_prior', False)),
                # --contrast_emb adds a `ctr_emb` PARAMETER, not a zero-init one that can
                # be whitelisted: constructing without it makes `ctr_emb` an UNEXPECTED
                # key and `assert not unexp` kills the load. The three best models in the
                # inventory (full0/full1/full2, incl. the best single ever at 0.8905)
                # were all unloadable for this reason. The token itself is fed in the run
                # cell -- constructing it alone would load fine and silently drop the
                # embedding at test time, which is the worse failure of the two.
                # S40 --slice_prior adds `pos_bias`. It is zero-init AND whitelisted below,
                # so an OLD checkpoint loads through the missing-key path -- but a NEW one
                # would otherwise hit `assert not unexp`, which is exactly how the three
                # best models in the inventory became unloadable over `ctr_emb`. Construct
                # it from the saved args, like every other recipe flag.
                contrast_emb=bool(a.get('contrast_emb', False)),
                slice_prior=bool(a.get('slice_prior', False)),
                n_slices=int(a.get('slices', 24)),
                img_size=int(a['res']))
    # BACKWARD COMPATIBILITY, and the reason two 8-model submissions died.
    # `logt` (attention temperature) was added to MultiAttnPool and QueryViewPool AFTER
    # most of these checkpoints were trained, so their state_dicts have no such key and a
    # strict load raises "Missing key(s)". It is initialised to zero and used as exp(logt),
    # so a missing logt means exp(0)=1 -- EXACTLY the pre-fix arithmetic. Filling it with
    # zeros reproduces the old model bit-for-bit. Anything else missing is a real error.
    # `vbias` (per-target anatomical view-attention prior) is the second such parameter and
    # was added the same way: it is an ADDITIVE bias on the view logits and is all zeros
    # unless the run asked for --view_prior, so a checkpoint without it loads to exactly
    # the arithmetic it was trained with. Whitelisting it HERE, in the same change that
    # introduced it, is the rule that came out of losing two submissions to `logt`.
    ZERO_OK = ("logt", "vbias", "pos_scale", "pool_mix", "pos_bias")
    miss, unexp = m.load_state_dict(sd, strict=False)
    bad = [k for k in miss if not k.endswith(ZERO_OK)]
    assert not bad, f"{os.path.basename(p)} missing non-zero-init weights: {bad[:5]}"
    assert not unexp, f"{os.path.basename(p)} has unexpected weights: {list(unexp)[:5]}"
    if miss:
        print(f"  {os.path.basename(p)}: filled {len(miss)} zero-init "
              f"({', '.join(sorted({k.split('.')[-1] for k in miss}))}) -- "
              f"identical to how it was trained", flush=True)
    # Moving a GRU to CUDA calls flatten_parameters() -> torch._cudnn_rnn_flatten_weight,
    # which raises "no kernel image available" when the image's cuDNN RNN kernels do not
    # cover the assigned card. Disable cuDNN only across the transfer, then restore it so
    # the convolutions (which dominate runtime) still use cuDNN.
    _c = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    m = m.eval().to(dev)
    torch.backends.cudnn.enabled = _c
    # Mirror TTA is only valid for a model that was TRAINED with random mirroring: it has
    # seen both orientations and is equivariant. A model trained without it has not, and
    # averaging it with its own mirror would just add noise, so the flag rides on the
    # checkpoint's own args rather than on a global switch.
    # LEAN_MODE forces this off: mirror TTA doubles every forward pass of a mirror-trained
    # model for a measured +0.0009 AUC, which on the efficiency metric is worth ~65 s and
    # costs far more than that.
    m._mirror = ((float(a.get('mirror_aug', 0) or 0) > 0)
                     and not LEAN_MODE and TTA_MIRROR)
    # The shard slice cap the checkpoint TRAINED on. train.py now records it as
    # args.src_slices (derived from the index, not a flag, so it cannot be forgotten).
    # SRC_SLICES here is a single global, so a checkpoint trained against a re-sharded store
    # with a different cap would be served a stack built by a different linspace -- and
    # linspace(0,N-1,32) is NOT a subset of linspace(0,N-1,64), so the slices genuinely
    # differ. That is the same silent train/test mismatch class as the norm_p80 bug, which
    # sat in this file undetected. Fail loudly instead. Supporting a MIXED ensemble means
    # decoding each capped series once per distinct cap and keying the stack cache by it;
    # build that only when a re-shard actually happens, not speculatively.
    # Same guard for the store's pixel resolution (S14). train.py now records store_res by
    # DECODING a real stored slice. 0 means a checkpoint from before that existed -- treat as
    # legacy and skip, because failing on every old checkpoint would be worse than the risk.
    _sr = int(a.get('store_res', 0) or 0)
    assert _sr in (0, STORE_RES), (
        f"{os.path.basename(p)} trained on a store of {_sr} px but this kernel has "
        f"STORE_RES={STORE_RES}. A re-shard changed the store under the checkpoints; set "
        f"STORE_RES to match or key the decode cache by resolution.")
    _cs = int(a.get('src_slices', SRC_SLICES))
    assert _cs == SRC_SLICES, (
        f"{os.path.basename(p)} trained on a store with src_slices={_cs} but this kernel "
        f"has SRC_SLICES={SRC_SLICES}. Set SRC_SLICES to match, or add per-checkpoint cap "
        f"support to load_slices/build_study. Refusing to run a silent mismatch.")
    # PRECISION. S14 found bf16-train/fp16-validate inside train.py and fixed it there
    # (EVAL_DTYPE), but left the IDENTICAL bug on this path: the autocast in CELL_RUN passes
    # no dtype, and torch defaults CUDA autocast to float16. So a --bf16 checkpoint has
    # always been SERVED in a precision it never trained in.
    # It cannot be fixed by matching bf16 here: bf16 needs compute capability >= 8.0 and
    # Kaggle gives T4 (7.5) or P100 (6.0). The fix is therefore on the TRAINING side --
    # drop --bf16, whose measured worth is ~1% of step time (S13, after the "~25%" claim was
    # retracted) -- and this warning is what stops a bf16 checkpoint entering the ensemble
    # unnoticed. Warn rather than assert: refusing to run would turn a small, unquantified
    # precision shift into a lost submission.
    if bool(a.get('bf16', False)):
        print(f"!! WARNING {os.path.basename(p)} was TRAINED with --bf16 but this kernel "
              f"can only infer in fp16 (T4/P100 are compute capability < 8.0). Train/serve "
              f"precision mismatch, size unmeasured.", flush=True)
    # norm_float is part of the KEY, not a per-model flag: it changes the uint8 tensor
    # itself (legacy multiplies the gain in and clips, --norm_float leaves it raw), so two
    # checkpoints that disagree on it cannot share one decoded volume.
    # crop_mm JOINS THE KEY (S14). It was a bare module global CROP_MM = 130.0 while the
    # shipped recipe trains at --crop_mm 150, and nothing here ever read it off the
    # checkpoint -- so a 150 mm model would have been served a 130 mm crop: 13.3% narrower
    # per axis on the 84.6% of series where both fractions are in range, i.e. ~1.154x
    # zoomed in, on EVERY study. Same bare-global exposure class as SRC_SLICES had, and it
    # would have silently invalidated the whole final run. `or CROP_MM` because
    # checkpoints trained before --crop_mm existed have no such key.
    # S51: the trailing False is the 3D flag. It MUST be present on both branches -- a
    # key of length 8 and one of length 9 would never collide, so the bug would look like
    # "3D got its own group", which is what we want anyway, and nothing would complain
    # until some later code unpacked a key positionally. Same length, explicit flag.
    _gkey = (int(a['res']), int(a['slices']), len(views),
             bool(a.get('norm_p80', False)),
             bool(a.get('crop_xy', False)),
             bool(a.get('crop_clamp', False)),
             bool(a.get('norm_float', False)),
             float(a.get('crop_mm') or CROP_MM),
             False)
    GROUPS.setdefault(_gkey, []).append(m)
    # PER_MODEL is indexed by position in the FLATTENED GROUPS iteration (see `k = 0` in
    # CELL_RUN), NOT by load order -- so a weight table keyed by checkpoint name needs this
    # parallel structure to recover index -> name. Appended in lockstep with GROUPS above
    # from the SAME key object, so the two cannot drift apart.
    GROUP_NAMES.setdefault(_gkey, []).append(os.path.basename(p))
    print('loaded', os.path.basename(p), '|', a['backbone'],
          f"{a['res']}/{a['slices']}/{len(views)}v",
          '| per_target', bool(a.get('per_target', False)),
          '| spatial', bool(a.get('spatial', False)),
          '| aux', n_aux, '| cv', round(ck.get('cv', float('nan')), 4), flush=True)
assert GROUPS, 'no checkpoints found in ' + CKPT_DIR
# A name in ONLY that matches nothing would SILENTLY ship a smaller ensemble than intended
# -- the same class of defect as the filename collision that once turned an 8-model set
# into 6 with nothing complaining. Name the missing files and refuse to continue.
if ONLY is not None:
    _missing = sorted(set(ONLY) - set(_seen))
    assert not _missing, f'ONLY names {len(_missing)} absent checkpoint(s): {_missing}'
    print(f'ONLY filter: {len(_seen)} of {len(ONLY)} requested loaded', flush=True)
# The physical crop USED to happen once per series before models were grouped, which
# forced the whole ensemble onto one --crop_xy setting and made the three full-stack
# models (our best fold-0/1/2 singles, including the 0.8905 that is the best single model
# in project history) unmixable with the other eleven. That is a pure packaging limit, not
# a modelling one: `load_slices` now returns BOTH crops from ONE decode and crop_xy joins
# the group key, so each model gets the geometry it trained on. Cost is near zero -- the
# two crops are the SAME object except on the ~7% of series whose physical FOV is
# non-square, which is the only case where the per-axis fraction differs.
# the exact (crop_xy, crop_clamp) geometries the loaded checkpoints ask for. load_slices
# builds only these, and identical ones share one array.
NEED_CROPS = sorted({(k[4], k[5], k[7]) for k in GROUPS})
print('crop geometries needed (crop_xy, crop_clamp, crop_mm):', NEED_CROPS, flush=True)
N_MODELS = sum(len(v) for v in GROUPS.values())
# The exact order PER_MODEL is indexed in. Asserting the length keeps a future change to the
# grouping from silently misaligning a per-target weight table against the wrong models.
MODEL_ORDER = [_n for _k in GROUPS for _n in GROUP_NAMES[_k]]
assert len(MODEL_ORDER) == N_MODELS, f'MODEL_ORDER {len(MODEL_ORDER)} != N_MODELS {N_MODELS}'
# decode the union of slots any model needs, once per study
SLOTS = VIEWS5 if max(k[2] for k in GROUPS) > len(VIEWS) else VIEWS
print('groups', {k: len(v) for k, v in GROUPS.items()}, '| models', N_MODELS,
      '| slots decoded', len(SLOTS), flush=True)



def order_slices(ds):
    try:
        o = np.array(ds[0].ImageOrientationPatient, float)
        nrm = np.cross(o[:3], o[3:])
        if nrm[np.argmax(np.abs(nrm))] < 0:
            nrm = -nrm
        key = [float(np.dot(np.array(d.ImagePositionPatient, float), nrm)) for d in ds]
    except Exception:
        key = [float(getattr(d, 'InstanceNumber', i)) for i, d in enumerate(ds)]
    return [ds[i] for i in np.argsort(key)]


def load_slices(sdir):
    # Replays the training pipeline literally, including the JPEG round-trip, and STOPS
    # at the cropped stored stack -- the last point that is common to every model config.
    # The per-model resize happens in sample_vol. Verified bit-identical to
    # dataset.KneeStudies._view_tensor on real data (maxdiff 0).
    fs = glob.glob(f'{sdir}/*.dcm')
    if not fs:
        return None
    # Stage 1 keeps at most SRC_SLICES of a series (median 30, max 320). On a LONG series
    # most pixel data is read and thrown away, and pixel data is ~78% of the read cost
    # (measured: 60 files, 77 ms full vs 17 ms headers-only). So on long series, order by
    # header first and decode pixels only for the slices that survive.
    # On a short series every slice is kept, so the header pass would be pure overhead --
    # hence the branch. Runtime is a scored quantity here (Efficiency leaderboard measures
    # wall clock including DICOM reads), and we are currently rank 1 on it.
    if len(fs) <= SRC_SLICES:
        ds = order_slices([pydicom.dcmread(f) for f in fs])
        arrs = [d.pixel_array.astype(np.float32) for d in ds]
    else:
        hdr = order_slices([pydicom.dcmread(f, stop_before_pixels=True) for f in fs])
        idx = np.linspace(0, len(hdr) - 1, SRC_SLICES).round().astype(int)
        ds = [hdr[i] for i in idx]
        arrs = [pydicom.dcmread(d.filename).pixel_array.astype(np.float32) for d in ds]
    # TRUE contrast class from the same header dict the geometry comes from. dataset.py
    # takes it off the SERIES it actually sampled (rec[8]=TE, rec[11]=TR); ds[0] here is a
    # slice of that same series, so this is the same quantity. Read off headers, so it is
    # free and works on the stop_before_pixels branch too. Anything unreadable -> 3
    # (unknown), which is exactly what dataset.py stores for an absent slot.
    ctr = contrast_class(getattr(ds[0], 'EchoTime', None),
                         getattr(ds[0], 'RepetitionTime', None))
    lo, hi = np.percentile(np.concatenate([a.ravel() for a in arrs]), [0.5, 99.5])
    stored = []
    for a in arrs:
        a = cv2.resize(np.clip((a - lo) / (hi - lo + 1e-6), 0, 1),
                       (STORE_RES, STORE_RES), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', (a * 255).astype(np.uint8),
                               [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
        stored.append(cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE))

    # --- the --norm_p80 landmark gain, computed HERE and not in sample_vol (B7) ---
    # dataset.py:294-303 takes the p80 off `blobs[...]`, i.e. the STORED slice, and the
    # crop below has not happened yet at that point. sample_vol used to recompute it from
    # `sl`, which build_study had ALREADY CROPPED -- a different set of pixels, so a
    # different p80 and a different gain, at test time only. Measured previously: 18.0% of
    # series off by more than 5%. Every --norm_p80 checkpoint in the shipped recipe carried
    # this, and nothing flagged it because both sides "computed a p80".
    #
    # Computing it once here also makes it ONE gain per series rather than one per crop
    # VARIANT, which is what training means by "one gain per series".
    # Index: dataset.py uses need[len(need)//2] where `need` covers 0..n-1 completely
    # (centres are linspace over the whole stack and each contributes c-1,c,c+1, so the
    # largest gap is 2 and is always bridged), hence need[len(need)//2] == n//2 exactly.
    p80_gain = None
    if stored:
        _mid = stored[len(stored) // 2]
        _fg = _mid[_mid > FG_MIN]
        if _fg.size >= 500:
            _p = float(np.percentile(_fg, 80.0))
            if _p > 1.0:
                p80_gain = float(np.clip(TARGET_P80 / _p, 0.25, 4.0))

    # --- stage 2a: physical-mm centre crop, identical to dataset.py ---
    # pixel pitch spans ~16x across series, so without this the model sees anatomy at a
    # scale it never trained on. The ROW fraction was historically applied to BOTH axes,
    # which is wrong for the 7.2% of series whose physical FOV is not square (column
    # window 90-260 mm instead of 130, 441 studies off by >20%). CROP_XY crops each axis
    # by its own fraction; it must match what the checkpoints trained with, so it is read
    # off them and every model in the ensemble has to agree.
    px_r = float(getattr(ds[0], 'PixelSpacing', [0, 0])[0] or 0)
    src_rows = float(getattr(ds[0], 'Rows', 0) or 0)
    try:
        px_c = float(getattr(ds[0], 'PixelSpacing', [0, 0])[1] or 0)
        src_cols = float(getattr(ds[0], 'Columns', 0) or 0)
    except Exception:
        px_c = src_cols = 0.0

    def _fracs(cmm):
        # The RAW crop fractions for ONE crop_mm, exactly dataset.py's arithmetic on the same
        # fields (rec[3]=px_r, rec[4]=rows, rec[9]=px_c, rec[10]=cols).
        #
        # Returns the column fraction RAW -- it must NOT be pre-collapsed to the row fraction
        # here. Doing that made the `cclamp` arm of the branch below unreachable whenever the
        # row fraction was usable, so a --crop_clamp model had its COLUMN cropped by the ROW
        # fraction while dataset.py:260-265 gives it keep_c = 1.0. Measured over the real
        # index at crop_xy+crop_clamp: 26 of 24,369 series diverge at crop_mm 130 but
        # 507 (2.08%, 361 studies) at the shipped crop_mm 150 -- mean column kept-fraction
        # 0.9223, min 0.6667. The old code had the same defect; widening crop_mm multiplied
        # its reach 19.5x, which is why it matters now.
        fr = cmm / (px_r * src_rows) if (px_r and src_rows) else 0.0
        fc = cmm / (px_c * src_cols) if (px_c and src_cols) else 0.0
        return fr, fc

    def _crop(fr, fc_):
        out = []
        for a in stored:
            h, w = a.shape[0], a.shape[1]
            kr = max(int(round(h * fr)), 32)
            kc = max(int(round(w * fc_)), 32)
            out.append(a[(h - kr) // 2:(h - kr) // 2 + kr,
                         (w - kc) // 2:(w - kc) // 2 + kc])
        return out

    # One decode, one crop per GEOMETRY the loaded checkpoints actually ask for. Keyed by
    # (crop_xy, crop_clamp) and built lazily, because identical keys share the same object:
    #  - crop_xy   : per-axis fraction. Differs from base only on the 7.2% of series whose
    #                physical FOV is non-square.
    #  - crop_clamp: clamp each fraction to 1.0 instead of dropping out of the crop. Fixes
    #                the scale bug (FOV_r <= crop_mm skipped the crop on BOTH axes: 41
    #                studies at up to 45.9% off scale) and the column fallback (FOV_c <=
    #                crop_mm made the column axis borrow the ROW fraction: 23 studies).
    # It must be per-model and not global: every checkpoint trained before 2026-08-12 saw
    # the unclamped geometry, so feeding it the clamped one is a train/test mismatch.
    variants, cache = {}, {}
    nocrop = False
    for cxy, cclamp, cmm in NEED_CROPS:
        frac, frac_c = _fracs(cmm)
        row_ok = 0 < frac < 1.0
        if not row_ok:
            # Two different causes land here and only one is benign. No usable PixelSpacing
            # is a real error (zero of 24,369 training series lacked it). frac >= 1.0 just
            # means the row FOV is already within crop_mm -- 41 studies -- and a crop_clamp
            # model still wants its COLUMN axis cropped, which the branches below handle.
            nocrop = True
        if not row_ok and not cclamp:
            # LEGACY, reproduced exactly: an unusable row fraction skipped the crop on BOTH
            # axes and the full stored frame went to the model. This is the scale bug, and
            # unclamped checkpoints were TRAINED on it, so it must be preserved for them.
            variants[(cxy, cclamp, cmm)] = stored
            continue
        fr = frac if row_ok else 1.0
        if not cxy:
            fc = fr
        elif 0 < frac_c < 1.0:
            fc = frac_c
        else:
            # column FOV already <= crop_mm: clamped -> no column crop; unclamped -> the
            # historic fallback of borrowing the ROW fraction, which squashed the axis.
            fc = 1.0 if cclamp else fr
        key = (round(fr, 9), round(fc, 9))
        if key not in cache:
            cache[key] = _crop(fr, fc)
        variants[(cxy, cclamp, cmm)] = cache[key]
    if nocrop:
        globals()['N_NOCROP'] = globals().get('N_NOCROP', 0) + 1
    return variants, ctr, p80_gain


def sample_vol(sl, res, n_slices, off=0.0, norm_p80=False, norm_float=False,
               p80_gain=None):
    # stage 2b: pick n_slices centres from the stored stack, shift by `off` slices, build
    # the 2.5D triplet, resize to this model's res. Only the ~26 distinct slices actually
    # referenced are resized.
    # Slice-offset is the right TTA for this task: a meniscal tear may be visible on two
    # slices out of twenty-four, so which centres we land on is a real source of variance.
    # Mirroring a right knee makes a LEFT knee, which is why mirror TTA IS valid and is
    # applied in the run cell for models trained with it -- the labels do not move. Scale
    # TTA would break the physical-mm crop.
    n = len(sl)
    gain = None
    if norm_p80:
        # must match dataset.py exactly: one gain per series, from the middle slice,
        # foreground p80 -> TARGET_P80. A train/inference mismatch here would rescale
        # every pixel at test time only, and nothing would flag it.
        #
        # BUG 1, fixed S13: `gain` was computed here and then NEVER APPLIED -- sample_vol
        # returned without using it, so every --norm_p80 checkpoint (the flag is in the
        # shipped recipe) was trained on p80-normalised input and served RAW input at test
        # time. Measured gain distribution: mean 0.89 overall, 0.63 on Coronal_0 and 1.11
        # on Coronal_1, so Coronal_0 arrived ~1.59x brighter than the model expected. The
        # parameter was also shadowed: it was named `norm`, which is the ImageNet normalise
        # FUNCTION at module scope -- renamed to norm_p80 so the shadowing cannot recur.
        #
        # BUG 2 (B7), fixed S15: the p80 was then taken from `sl`, which build_study has
        # ALREADY CROPPED, while dataset.py takes it from the UNCROPPED stored slice.
        # Different pixels -> different percentile -> different gain, on 18.0% of series by
        # more than 5%. It is now computed once in load_slices, off the uncropped stack,
        # and passed in. The fallback below only fires when a caller passes an uncropped
        # stack and no gain -- src/test_infer_equiv.py is the only one, and it does exactly
        # that -- so production always takes the value from load_slices.
        if p80_gain is not None:
            gain = p80_gain
        else:
            mid = sl[n // 2]
            fg = mid[mid > FG_MIN]
            if fg.size >= 500:
                pp = float(np.percentile(fg, 80.0))
                if pp > 1.0:
                    gain = float(np.clip(TARGET_P80 / pp, 0.25, 4.0))
    centres = np.clip((np.linspace(0, n - 1, n_slices) + off).round().astype(int), 0, n - 1)
    need = sorted({int(np.clip(c + d, 0, n - 1)) for c in centres for d in (-1, 0, 1)})
    cache = {}
    for i in need:
        a = sl[i]
        # Applied BEFORE the resize, matching dataset.py's decode -> gain -> crop -> resize.
        # The crop already happened upstream in build_study, but a scalar multiply commutes
        # with cropping exactly, so the order that matters is gain-before-resize.
        if gain is not None and not norm_float:
            a = np.clip(a.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        # BOTH dims, not just rows -- dataset.py:317 checks both and says so. Under
        # --crop_xy a non-square physical FOV gives (kr, kc) with kr == res and kc != res,
        # and a row-only test then SKIPS the resize and leaves a (res, kc) array. The
        # `vol[j, ch] = cache[i]` below then raises a broadcast ValueError, which the per
        # study try/except in CELL_RUN swallows by writing 0.5 to all 12 targets for the
        # WHOLE study. Measured on the real index: 0 series trigger it at crop_mm 130 /
        # res 480, but 338 of 24,369 (1.4%) do at the shipped crop_mm 150 / res 480 -- so
        # fixing CROP_MM without this would have newly exposed it.
        cache[i] = a if (a.shape[0] == res and a.shape[1] == res) else \
            cv2.resize(a, (res, res), interpolation=cv2.INTER_AREA)
    vol = np.empty((n_slices, 3, res, res), np.uint8)
    for j, c in enumerate(centres):
        for ch, d in enumerate((-1, 0, 1)):
            vol[j, ch] = cache[int(np.clip(c + d, 0, n - 1))]
    # second value is the gain still owed in float: 1.0 unless this is a --norm_float
    # checkpoint, which wants it applied after .float().div_(255) so nothing saturates
    return torch.from_numpy(vol), (gain if (norm_float and gain is not None) else 1.0)


def build_study(sid, sub, slots):
    # Decode each needed (plane, contrast) once. Returns, per slot, a (base_crop, xy_crop)
    # PAIR of cropped stored stacks rather than tensors, so one decode feeds every
    # (res, slices, views, norm_p80, crop_xy) group and every TTA offset.
    # Second return value is the per-slot contrast class for --contrast_emb models; it is
    # carried alongside rather than inside `out` so that `s is not None` (which builds the
    # presence mask) keeps meaning exactly what it meant.
    out, ctrs, gains = {}, {}, {}
    for plane, fluid in slots:
        m = sub[(sub.Anatomical_Plane == plane) & (sub.Fluid_Sensitive == fluid)]
        sl, ctr, gn = None, 3, None
        for _, r in m.iterrows():
            try:
                got = load_slices(f'{IMG_DIR}/{sid}/{r.SeriesInstanceUID}')
            except Exception:
                got = None          # bad series -> fall through to the next candidate
            if got is not None:
                sl, ctr, gn = got[0], got[1], got[2]  # sl is {(crop_xy, crop_clamp): stack}
                break
        out[(plane, fluid)] = sl
        # absent slot -> 3, matching dataset.py's `ctr.append(3)` on a missing slot
        ctrs[(plane, fluid)] = ctr if sl is not None else 3
        # the --norm_p80 gain off the UNCROPPED stack (B7); None -> sample_vol falls back
        gains[(plane, fluid)] = gn if sl is not None else None
    return out, ctrs, gains



sids = test.StudyInstanceUID.tolist()
by_study = {s: g for s, g in ts.groupby('StudyInstanceUID')}
preds = np.full((len(sids), 12), 0.5, np.float32)
# One array per MODEL, filled study by study, combined only at the end.
# Averaging raw probabilities across models assumes they are calibrated the same way, and
# ours are not: a convnext_tiny at 320 px and a convnext_small at 384 px trained on
# different folds put their probability mass in different places, so a plain mean lets
# whichever model is most confident dominate regardless of whether it is most correct.
# The metric is macro-AUC, which only cares about ORDER, so convert each model's column
# to ranks first and average those. Same trick the strongest public notebook uses.
PER_MODEL = [np.full((len(sids), 12), np.nan, np.float32) for _ in range(N_MODELS)]
done_ok = np.zeros(len(sids), bool)

def prep(i):
    # every failure returns None; an exception escaping here would abort the whole cell
    try:
        sid = sids[i]
        return i, build_study(sid, by_study.get(sid, ts.iloc[0:0]), SLOTS)
    except Exception as e:
        return i, None

CHUNK = 32            # bounds in-flight decoded studies to ~1 GB of RAM
t0, n_done, n_pres, n_fail = time.time(), 0, 0, 0
tta_cut = False
N_KEEP = None         # set by the third runtime valve; None = use every loaded model
pool = ThreadPoolExecutor(4)
with torch.no_grad():
    for base in range(0, len(sids), CHUNK):
        for i, built in pool.map(prep, range(base, min(base + CHUNK, len(sids)))):
            n_done += 1
            if built is None:
                n_fail += 1
                continue
            stacks, ctrs, gains = built
            n_pres += sum(v is not None for v in stacks.values())
            try:
                # dtype STATED, not defaulted. It was already float16 by default, but a
                # default is how the bf16 mismatch stayed invisible: nothing in this file
                # mentioned precision at all, so grepping "bf16" across src/ returned only
                # train.py and the reader concluded inference had no precision to get wrong.
                with torch.amp.autocast('cuda', enabled=AMP, dtype=torch.float16):
                    k = 0
                    for (res, ns, nv, nrm, cxy, ccl, nfl, cmm, is3d), ms in GROUPS.items():
                        slots = VIEWS5 if nv == len(VIEWS5) else VIEWS
                        pairs = [stacks.get(kk) for kk in slots]
                        pr = torch.tensor([[s is not None for s in pairs]]).to(dev)
                        # each group reads the crop geometry ITS checkpoints trained on,
                        # crop_mm included since S14 -- before that a --crop_mm 150 model
                        # silently received the hardcoded 130 mm crop
                        sl_list = [None if p is None else p[(cxy, ccl, cmm)] for p in pairs]
                        # one --norm_p80 gain per SERIES, taken off the uncropped stack in
                        # load_slices, so it is the same number for every crop variant and
                        # every TTA offset -- which is what dataset.py means by "one gain
                        # per series" (B7)
                        g_list = [gains.get(kk) for kk in slots]
                        # (B, n_views) long, SAME view order as `vs`. Models built without
                        # --contrast_emb have ctr_emb=None and forward() ignores this, so
                        # it is passed unconditionally rather than branched on.
                        ct = torch.tensor([[ctrs.get(kk, 3) for kk in slots]],
                                          dtype=torch.long).to(dev)
                        assert int(ct.max()) < N_CONTRAST, 'contrast class out of range'
                        # keep the passes rather than accumulating: per-target pooling
                        # needs the individual TTA passes, and a running sum destroys them
                        passes = [[] for _ in ms]
                        for off in TTA_OFFSETS:
                            vs = []
                            for sl, gn in zip(sl_list, g_list):
                                if sl is None:
                                    _v = torch.zeros(ns, 3, res, res, dtype=torch.uint8)
                                    _g = 1.0
                                else:
                                    _v, _g = sample_vol(sl, res, ns, off, nrm, nfl, gn)
                                # S51. Knee3DNet normalises INSIDE forward() with KINETICS
                                # statistics, not ImageNet's, and it resizes to res3d there
                                # too. Applying norm() here would feed it ImageNet-normalised
                                # float and then normalise that again -- a silent double
                                # transform with no exception and a plausible-looking score.
                                # The 3D group therefore gets the raw uint8 tensor, which is
                                # exactly what train3d.py hands the model.
                                vs.append(_v.unsqueeze(0).to(dev) if is3d
                                          else norm(_v.unsqueeze(0).to(dev), _g))
                            vs_m = None
                            for mi, m in enumerate(ms):
                                # skip models the third valve retired; k+mi is this model's
                                # global index in PER_MODEL, matching the write below
                                if N_KEEP is not None and k + mi >= N_KEEP:
                                    continue
                                passes[mi].append(torch.sigmoid(_head(_fwd(m, vs, pr, ct, is3d)).float()))
                                if not getattr(m, '_mirror', False):
                                    continue
                                if vs_m is None:
                                    # the mirror of a right knee is a left knee, so the
                                    # labels are unchanged. Costs NO extra DICOM decode --
                                    # inference here is decode-bound, so this is close to
                                    # free. Sagittal carries laterality on the SLICE axis
                                    # (reverse slices, and the 2.5D triplet with them);
                                    # axial/coronal carry it in-plane (flip width).
                                    # `slots` is VIEWS/VIEWS5, i.e. a list of (plane, fluid)
                                    # TUPLES -- not strings. Calling .startswith on the tuple
                                    # raised AttributeError for EVERY study, the per-study
                                    # try/except counted it as a failure, and the fallback
                                    # wrote a constant 0.5 to submission.csv. That is the
                                    # 0.500 on submission 55438579, the first one to include
                                    # a mirror-trained model. Guard: index the plane out.
                                    # S19 BUG FIX. `vs` is ALREADY NORMALISED (norm() above),
                                    # and flip(dims=[1,2]) reverses slices AND CHANNELS. The
                                    # channel reverse is correct in intent -- the 3 channels
                                    # are slices i-1,i,i+1, so reversing slice order must
                                    # reverse them too -- but training does it in the OTHER
                                    # ORDER: dataset.py:331 mirrors the RAW uint8 volume
                                    # (`vol[::-1, ::-1]`) and train.py normalises afterwards.
                                    # Doing it after normalisation leaves channel c carrying
                                    # data scaled by mean[2-c]/std[2-c]:
                                    #   train:  (v[S-1-s,2-c]/255 - M[c])   / S[c]
                                    #   infer:  (v[S-1-s,2-c]/255 - M[2-c]) / S[2-c]
                                    # Measured maxdiff 0.3911 on channels 0 and 2 and exactly
                                    # 0.0000 on channel 1 (the middle channel is unmoved by
                                    # the reverse) -- ~15% of the normalised range, injected
                                    # on every non-LEAN submission, on both Sagittal slots.
                                    # _mirror_sag re-maps it back onto the right constants;
                                    # the width flip needs no fix because it never touches
                                    # the channel axis. Proved bit-equal to the training path
                                    # in test_infer_equiv.py.
                                    # S51. For a 3D group `vs` is still RAW UINT8, so the
                                    # naive flip is the CORRECT one and _mirror_sag is the
                                    # wrong one: _mirror_sag exists only to undo the channel
                                    # permutation that mirroring AFTER ImageNet normalisation
                                    # introduces, and there is no such normalisation here.
                                    # `vol[::-1, ::-1]` (dataset.py:331) == flip(dims=[1,2])
                                    # on (B, S, 3, H, W) -- bit-identical to training.
                                    vs_m = [(v.flip(dims=[1, 2]) if is3d else _mirror_sag(v))
                                            if kk[0] == 'Sagittal'
                                            else v.flip(dims=[4])
                                            for v, kk in zip(vs, slots)]
                                # mirroring a right knee makes a left knee; the pulse
                                # sequence is unchanged, so `ct` is reused as-is
                                passes[mi].append(torch.sigmoid(_head(_fwd(m, vs_m, pr, ct, is3d)).float()))
                        # Pool the TTA passes WITHIN a model. A plain mean is the default
                        # and is right for a diffuse finding; for a FOCAL one it is the
                        # same dilution the slice pool already suffers, one level up --
                        # a fracture line visible in 2 of 24 slices is present in only
                        # some of the offset windows, and averaging over the windows that
                        # miss it washes it out. See TTA_TARGET_POOL.
                        for mi in range(len(ms)):
                            if N_KEEP is not None and k >= N_KEEP:
                                k += 1                            # stays NaN, see valve 3
                                continue
                            P = torch.cat(passes[mi], 0)          # (n_pass, 12)
                            PER_MODEL[k][i] = tta_pool(P).cpu().numpy()
                            k += 1
                    done_ok[i] = True
            except Exception as e:
                n_fail += 1
                print('infer failed', i, str(e)[:160], flush=True)
        # S16: the guard used to require len(TTA_OFFSETS) > 1, so in LEAN / no-TTA mode the
        # ENTIRE valve -- including the model-drop valve below, which has nothing to do with
        # TTA -- was unreachable. A no-TTA ensemble that overran 9 h therefore produced NO
        # submission at all, which is the one outcome every valve here exists to prevent, and
        # it is exactly the regime the accuracy ensemble now runs in (slice TTA is dominated
        # 7x by an extra member, so it is off by design). Arm the valve unconditionally; the
        # offset/mirror branches below are no-ops when they are already off.
        if not tta_cut and n_done >= 64:
            # extrapolate from a real measurement rather than trusting any estimate
            tta_cut = True
            proj = (time.time() - t0) / n_done * len(sids)
            if proj > TIME_BUDGET_S:
                TTA_OFFSETS = [0.0]
                print(f'projected {proj/3600:.1f} h > budget -- offset TTA disabled',
                      flush=True)
                # Second valve. Mirror TTA doubles the forward passes of the models that
                # use it, and dropping the offsets does not touch that. Decode dominates
                # runtime, so cutting offsets alone can be nowhere near enough. A kernel
                # that overruns produces NO submission, which is strictly worse than a
                # slightly weaker one, so give up the mirror as well when we are badly over.
                if proj > 2 * TIME_BUDGET_S:
                    for _g in GROUPS.values():
                        for _m in _g:
                            _m._mirror = False
                    print('still over by 2x -- mirror TTA disabled too', flush=True)
                # THIRD VALVE: DROP MODELS. Until S15 the valve could only give up TTA and
                # then the mirror, and if that was not enough the kernel simply overran --
                # producing NO submission, which is the worst possible outcome. That was
                # survivable at 8 members; the S15 campaign builds 20 (5 folds x 2 arms x
                # best+EMA), and 20 members x 5 views does not fit 9 h on the larger
                # plausible test set. Cost scales with the MODEL COUNT, so once TTA and
                # mirror are gone the only remaining lever is fewer models.
                # Keep a PREFIX of the load order, which stage_ckpts/mksubmit emit
                # fold-balanced, so the survivors stay spread across folds.
                # Crucially, blank the dropped models for the studies ALREADY done: AUC is a
                # ranking metric, so averaging early studies over 20 members and later ones
                # over 12 would score a mix of two different ensembles. nanmean then ignores
                # them and every study is averaged over exactly the same members.
                n_tot = len(PER_MODEL)
                over = proj / max(TIME_BUDGET_S, 1e-9)
                if over > 1.0 and n_tot > 1:
                    keep = max(1, int(n_tot / over * 0.9))       # 10% safety margin
                    if keep < n_tot:
                        for _j in range(keep, n_tot):
                            PER_MODEL[_j][:] = np.nan
                        globals()['N_KEEP'] = keep
                        print(f'projected {proj/3600:.1f} h still over -- keeping the first '
                              f'{keep} of {n_tot} models, dropping the rest for EVERY study',
                              flush=True)
        if n_done % 128 < CHUNK:
            el = time.time() - t0
            print(f'{n_done}/{len(sids)}  views/study {n_pres/max(n_done,1):.2f}  '
                  f'fail {n_fail}  {el/60:.1f} min  '
                  f'eta {(len(sids)-n_done)/max(n_done,1)*el/60:.1f} min', flush=True)

# combine: plain mean of the raw probabilities across models.
# Rank-averaging was tried and LOST, on two independent instruments: the same 6 models
# scored LB 0.927 raw vs 0.924 rank-averaged, and out-of-fold agreed (fold 1: raw 0.8848
# vs rank 0.8842; fold 0: 0.8866 vs 0.8857). It had been adopted on a SYNTHETIC
# calibration-mismatch test that did not transfer to real models. Studies that failed
# keep 0.5 everywhere and are left out of the mean.
ok = np.nonzero(done_ok)[0]
if len(ok) > 0:
    S = np.stack([P[ok] for P in PER_MODEL])              # (M, n_ok, 12)
    if TARGET_WEIGHTS is None:
        preds[ok] = np.nanmean(S, axis=0)
        print(f'raw-mean blended {len(PER_MODEL)} models over {len(ok)} studies', flush=True)
    else:
        # PER-TARGET WEIGHTED MEAN. Each of the 12 targets gets its own mixture of members;
        # the metric never compares one target's scores with another's, so this is free.
        # A model that FAILED on a study is NaN there and must drop OUT OF THE DENOMINATOR
        # for that (study, target) -- otherwise that study is scored on a different mixture
        # than its neighbours, which is precisely the bug the flat mean's nanmean avoids and
        # which would corrupt a ranking metric silently.
        _miss = [n for n in MODEL_ORDER if n not in TARGET_WEIGHTS]
        assert not _miss, f'TARGET_WEIGHTS has no entry for {len(_miss)} loaded model(s): {_miss[:3]}'
        W = np.asarray([TARGET_WEIGHTS[n] for n in MODEL_ORDER], np.float64)   # (M, 12)
        assert W.shape == (len(PER_MODEL), len(TARGETS)), f'weight shape {W.shape}'
        assert np.isfinite(W).all() and (W >= 0).all(), 'weights must be finite and >= 0'
        assert W.sum(0).min() > 0, 'every target needs at least one non-zero weight'
        if N_KEEP is not None:                 # the time-budget valve retired the tail
            W[N_KEEP:] = 0.0
            assert W.sum(0).min() > 0, 'time budget retired every weighted model for a target'
        Wb = np.where(np.isnan(S), 0.0, W[:, None, :])     # (M, n_ok, 12)
        den = Wb.sum(0)
        num = np.nansum(np.nan_to_num(S, nan=0.0) * Wb, axis=0)
        # where every weighted member is missing, fall back to the unweighted mean rather
        # than emitting NaN and tripping the submission assert below
        preds[ok] = np.where(den > 0, num / np.maximum(den, 1e-12), np.nanmean(S, axis=0))
        _eff = (W.sum(0) ** 2) / np.maximum((W ** 2).sum(0), 1e-12)
        print(f'PER-TARGET weighted blend, {len(PER_MODEL)} models over {len(ok)} studies',
              flush=True)
        for _t, _e, _nz in zip(TARGETS, _eff, (W > 0).sum(0)):
            print(f'   {_t:18} {_nz:3d} members, {_e:5.2f} effective', flush=True)

# if the image tree were wrong, every view would be absent and every row identical --
# a valid-looking CSV that scores exactly 0.500. Fail loudly instead.
assert n_pres / max(n_done, 1) > 0.5 * len(SLOTS) / 4, \
    f'only {n_pres/max(n_done,1):.2f} views per study loaded -- image path is wrong'
print('total minutes', round((time.time()-t0)/60, 1), '| failed', n_fail,
      '| series with no usable PixelSpacing', globals().get('N_NOCROP', 0), flush=True)



sub = pd.DataFrame({'StudyInstanceUID': sids})
for j, t in enumerate(TARGETS):
    sub[t] = preds[:, j]
assert len(sub) == len(test) and not sub[TARGETS].isna().any().any()
if PROBE_TARGETS is not None:
    # Blank every target OUTSIDE the probe group to a constant, so its AUC is exactly 0.5 and
    # the reported macro decomposes. Asserts rather than silently degrading: a name typo here
    # would produce a normal-looking submission whose arithmetic is wrong, which is precisely
    # the class of bug this repo keeps shipping.
    _bad = [t for t in PROBE_TARGETS if t not in TARGETS]
    assert not _bad, f'PROBE_TARGETS has unknown target(s): {_bad}'
    assert 0 < len(PROBE_TARGETS) < len(TARGETS), \
        f'PROBE_TARGETS must be a proper non-empty subset, got {len(PROBE_TARGETS)}'
    _blank = [t for t in TARGETS if t not in PROBE_TARGETS]
    for t in _blank:
        sub[t] = 0.5
    print(f'PROBE: reporting {sorted(PROBE_TARGETS)}; {len(_blank)} target(s) set to constant '
          f'0.5. Invert with  mean_AUC = (12*LB - {len(_blank)}/2) / {len(PROBE_TARGETS)}',
          flush=True)
    assert (sub[_blank].nunique() == 1).all(), 'blanking failed -- a blanked column varies'
assert list(sub.columns) == list(ssub.columns)
sub.to_csv('/kaggle/working/ours_s10.csv', index=False)
print(sub.shape)
print(sub[TARGETS].describe().loc[['mean','std','min','max']].round(3).to_string())
