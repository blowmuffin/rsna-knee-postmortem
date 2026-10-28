# RSNA Knee: one source for the prep / train / infer kernels. build.py sets MODE.
MODE = 'train'
SMOKE = 0  # >0: only this many studies (quick end-to-end check)
CACHE = 'knee-prep320a,knee-prep320b,knee-prep320c,knee-prep320d'  # train: kernel(s) whose output holds cache/*.npz, comma-separated for a sharded cache
SHARD = (0, 1)  # prep: this kernel caches studies[SHARD[0]::SHARD[1]] (Kaggle output is capped at 20GB)
ALLOW_SMOKE = 0  # infer: 1 only for plumbing tests (smoke weights are never fit for a real submission)
BLEND = 0.0  # infer: weight of an existing submission.csv (public stack) in the final rank blend
PSEUDO_A, PSEUDO_FROM = 0.7, ['knee-train-eff320', 'knee-train-eff320b', 'knee-train-eff31-mil', 'knee-train-rad31-mil']  # train: target = (1-a)*report label + a*mean OOF of these knee-train* kernels
LABEL_SRC = 'v4b'  # train: 'v4b' (stevenleehans v4 blend + SILENT_FIX), 'rubric' (our severity-rubric LLM labels) or 'consensus' (consensus.py)

import os, glob, json, time
import numpy as np, pandas as pd

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
          'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
SLOTS = [('Sagittal', 1), ('Coronal', 1), ('Axial', 1), ('Sagittal', 0), ('Coronal', 0)]
# 31 slices: the main site has 30-48 per series, 16 skipped every other one. pick_idx(31)[::2] == pick_idx(16),
# so models trained on the 16-slice cache still run on a 31-slice test prep (run_infer subsamples).
N_SLICE, SIZE, CROP_MM = 31, 256, 150.0
# Prep geometry. 1: band .1-.9, crop only where the image is larger than CROP_MM (non-square images got
# stretched, small fields of view magnified), exact (plane, fat-sat) slot match only.
# 2: band .04-.96 (keeps collaterals / meniscus edge), aspect-true crop zero-padded to CROP_MM, an empty slot
# falls back to an unused series of the same plane, one image per slice position (multi-echo series).
PREP = 2
BANDS = {1: (0.1, 0.9), 2: (0.04, 0.96)}
EPOCHS, FOLD_LIST = 6, [0, 1]  # train: FOLD_LIST=[] -> all 5 folds (split folds across kernels for long runs)
SILENT_FIX = {'ACL': .05, 'MCL': .05, 'Lateral Meniscus': .1, 'Medial OA': .03, 'Lateral OA': .05,
              'Effusion': .1, "Baker's": .04, 'Contusion': .08, 'Fracture': .05}  # ~gold rate when report is silent
T0 = time.time()


def uptime_h():
    """Hours since this Python process started (= notebook start when appended to the public stack)."""
    try:
        import psutil
        return (time.time() - psutil.Process().create_time()) / 3600
    except Exception:
        return (time.time() - T0) / 3600


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def find_input(name):
    """Kaggle mounts inputs at a few different depths depending on source type."""
    pats = [f'/kaggle/input/{name}', f'/kaggle/input/competitions/{name}',
            f'/kaggle/input/notebooks/*/{name}', f'/kaggle/input/datasets/*/{name}']
    for p in pats:
        hits = glob.glob(p)
        if hits:
            return hits[0]
    raise FileNotFoundError(name)


# ---------------------------------------------------------------- DICOM -> canonical volume
# In-plane target axes per plane (LPS patient coords): horizontal axis index, (vertical axis, wanted sign).
AXES = {'Sagittal': (1, (2, -1)), 'Coronal': (0, (2, -1)), 'Axial': (0, (1, 1))}
STACK_AXIS = {'Sagittal': 0, 'Coronal': 1, 'Axial': 2}
HDR = ['ImagePositionPatient', 'ImageOrientationPatient', 'InstanceNumber', 'PixelSpacing',
       'Laterality', 'Rows', 'Columns', 'EchoNumbers']


def orient(img, iop, plane):
    r, c = np.asarray(iop[:3], float), np.asarray(iop[3:6], float)
    h, (v, vs) = AXES[plane]
    if abs(r[h]) < abs(c[h]):
        img, r, c = img.T, c, r
    if r[h] < 0:
        img = img[:, ::-1]
    if c[v] * vs < 0:
        img = img[::-1]
    return img


def side_of(hdrs):
    for _, d in hdrs:
        lat = str(getattr(d, 'Laterality', '') or '').strip().upper()
        if lat[:1] in ('L', 'R'):
            return lat[0]
    xs = []
    for _, d in hdrs:  # fallback: image centre x; patient right is -x in LPS
        try:
            ipp, iop = np.asarray(d.ImagePositionPatient, float), np.asarray(d.ImageOrientationPatient, float)
            ps = float(d.PixelSpacing[0])
            xs.append((ipp + iop[:3] * ps * d.Columns / 2 + iop[3:] * ps * d.Rows / 2)[0])
        except Exception:
            pass
    return ('R' if np.median(xs) < 0 else 'L') if xs else 'L'


def series_headers(sdir, plane):
    """[(file, header)] sorted along the plane's stacking axis (InstanceNumber if geometry is missing)."""
    import pydicom
    hdrs = []
    for f in os.listdir(sdir):
        if f.endswith('.dcm'):
            try:
                hdrs.append((f, pydicom.dcmread(os.path.join(sdir, f), stop_before_pixels=True, specific_tags=HDR)))
            except Exception:
                pass
    ax = STACK_AXIS[plane]

    def num(h, tag):
        try:
            return float(getattr(h, tag, 0) or 0)
        except Exception:
            return 0.0

    def key(fd):
        try:
            return (0, float(fd[1].ImagePositionPatient[ax]), num(fd[1], 'EchoNumbers'), num(fd[1], 'InstanceNumber'))
        except Exception:
            return (1, num(fd[1], 'InstanceNumber'), 0.0, 0.0)
    hdrs = sorted(hdrs, key=key)
    if PREP >= 2:  # multi-echo / Dixon: several images per position -> keep the first echo at each position
        out, last = [], None
        for fd in hdrs:
            k = key(fd)
            p = (k[0], round(k[1], 1))
            if k[0] == 1 or p != last:
                out.append(fd)
            last = p
        hdrs = out
    return hdrs


def pick_idx(n, side, plane):
    """N_SLICE indices into the sorted slices. Sagittal is sorted by +x (patient left), which runs
    medial->lateral for a left knee and lateral->medial for a right one, so right knees read it backwards."""
    lo, hi = BANDS[PREP]
    idx = np.linspace(lo * (n - 1), hi * (n - 1), N_SLICE).round().astype(int)
    return n - 1 - idx if side == 'R' and plane == 'Sagittal' else idx  # exact mirror


def read_series(sdir, plane, hdrs, side):
    """-> uint8 [N_SLICE,SIZE,SIZE] in canonical left-knee layout, or None."""
    import pydicom, cv2
    if not hdrs:
        return None
    cache, out = {}, []
    for i in pick_idx(len(hdrs), side, plane):
        if i not in cache:
            f, h = hdrs[i]
            try:
                d = pydicom.dcmread(os.path.join(sdir, f))
                a = d.pixel_array.astype(np.float32)
                if a.ndim == 3:
                    a = a[0] if a.shape[0] < a.shape[-1] else a.mean(-1)
                a = a * float(getattr(d, 'RescaleSlope', 1) or 1) + float(getattr(d, 'RescaleIntercept', 0) or 0)
                a = orient(a, d.ImageOrientationPatient, plane)
                if side == 'R' and plane != 'Sagittal':
                    a = a[:, ::-1]
                want = int(round(CROP_MM / float(d.PixelSpacing[0])))
                hh, ww = a.shape
                if PREP >= 2 and want > 16:  # centre crop to CROP_MM per axis, zero-pad the short side: true scale
                    ch, cw = min(want, hh), min(want, ww)
                    y0, x0 = (hh - ch) // 2, (ww - cw) // 2
                    sq = np.zeros((want, want), np.float32)
                    py, px = (want - ch) // 2, (want - cw) // 2
                    sq[py:py + ch, px:px + cw] = a[y0:y0 + ch, x0:x0 + cw]
                    a = sq
                elif 16 < want < max(hh, ww):
                    y0, x0 = max(0, (hh - want) // 2), max(0, (ww - want) // 2)
                    a = a[y0:y0 + want, x0:x0 + want]
                cache[i] = cv2.resize(np.ascontiguousarray(a), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            except Exception:
                cache[i] = None
        out.append(cache[i])
    good = [x for x in out if x is not None]
    if not good:
        return None
    out = [x if x is not None else good[0] for x in out]
    vol = np.stack(out)
    lo, hi = np.percentile(vol[:, ::4, ::4], [1, 99.5])
    return (np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def prep_study(args):
    """-> (study, x uint8 [5,N_SLICE,SIZE,SIZE], mask [5], n_failed_slots, side). Never raises."""
    study, root, recs = args
    x = np.zeros((len(SLOTS), N_SLICE, SIZE, SIZE), np.uint8)
    m = np.zeros(len(SLOTS), np.uint8)
    fails, chosen, used = 0, {}, set()
    for strict in (True, False) if PREP >= 2 else (True,):  # 2nd pass: empty slot takes a spare same-plane series
        for k, (plane, fs) in enumerate(SLOTS):
            if k in chosen:
                continue
            try:
                cand = [r for r in recs if r['Anatomical_Plane'] == plane and (r['Fat_Suppression'] == fs or not strict)
                        and isinstance(r['SeriesInstanceUID'], str)]
                sdirs = [d for d in (os.path.join(root, study, r['SeriesInstanceUID']) for r in cand)
                         if os.path.isdir(d) and d not in used]
                if sdirs:
                    sdir = max(sdirs, key=lambda d: len(os.listdir(d)))
                    chosen[k] = (plane, sdir, series_headers(sdir, plane))
                    used.add(sdir)
            except Exception:
                fails += 1
    side = side_of([h for _, _, hd in chosen.values() for h in hd])  # one side per study, from every series
    for k, (plane, sdir, hd) in chosen.items():
        try:
            vol = read_series(sdir, plane, hd, side)
        except Exception:
            vol = None
        if vol is None:
            fails += 1
            continue
        x[k], m[k] = vol, 1
    return study, x, m, fails, side


def prep_all(split, studies, out_dir=None):
    """Preprocess studies in parallel; yields (study, x, m). Writes npz if out_dir.
    A stuck/dead worker can't hang the run: after a 10-min stall the rest are yielded as empty (masked)."""
    from multiprocessing import Pool
    comp = find_input('rsna-knee-abnormality-detection')
    ser = pd.read_csv(f'{comp}/{split}_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    by = {s: g.to_dict('records') for s, g in ser.groupby('StudyInstanceUID')}
    jobs = [(s, f'{comp}/{split}_series', by.get(s, [])) for s in studies]
    nfail, sides, done = 0, {'L': 0, 'R': 0}, set()
    from collections import deque
    with Pool(os.cpu_count()) as pool:
        todo, window = iter(jobs), deque()  # bounded in-flight window keeps memory flat

        def submit():
            j = next(todo, None)
            if j is not None:
                window.append(pool.apply_async(prep_study, (j,)))
        for _ in range(4 * os.cpu_count()):
            submit()
        i = 0
        while window:
            try:
                s, x, m, f, side = window.popleft().get(timeout=600)
            except Exception as e:
                log(f'{split}: prep stopped after {i} studies ({e!r}); rest left empty')
                break
            submit()
            done.add(s)
            nfail += f
            sides[side] += 1
            if out_dir:
                np.savez_compressed(f'{out_dir}/{s}.npz', x=x, m=m)
            if i % 200 == 0:
                log(f'{split} {i}/{len(jobs)} studies, failed slots so far {nfail}, sides {sides}')
            i += 1
            yield s, x, m
    for s, _, _ in jobs:
        if s not in done:
            yield s, np.zeros((len(SLOTS), N_SLICE, SIZE, SIZE), np.uint8), np.zeros(len(SLOTS), np.uint8)
    log(f'{split} done: {len(jobs)} studies, failed slots {nfail}, sides {sides}, missing {len(jobs) - len(done)}')


# ---------------------------------------------------------------- model
BACKBONE, IMG = 'efficientnet_b0.ra_in1k', 288
RAD = 'radimagenet_resnet50'  # centre slice x3, inputs scaled to [-1, 1] (how RadImageNet was trained)
HEAD = 'mil'  # train: 'mil' per-finding attention over all tokens + slice position; 'slot' = original head
POS_BINS = 31  # slice-position embedding bins over the [0,1] stack position
SLOT_DROP = 0.0  # train (mil head): drop each present series with this p (never all), robust to missing planes
HEAD_LR = 0.0  # train: >0 gives every non-encoder parameter this LR (public Raptor recipe: backbone 3e-5, head 1e-3)
ROT_DEG = 10  # train: in-plane rotation augmentation range (+/- degrees)
AUX_W = 0.0  # train (mil head): weight of an auxiliary max-over-tokens MIL loss on per-token logits
POS_W = 0  # train: 1 -> BCE pos_weight (1-prev)/prev clipped to [1,10] from the fold's training targets (Raptor)
MIRROR = 0.0  # train: p per study of the virtual contralateral knee (heliosli R6 --mirror); >0 also turns on flip TTA
# Mirror = coronal/axial flipped on width (patient left-right), sagittal slice order reversed (the cache stores it side-
# canonical); medial and lateral findings swap places and MCL has no lateral twin, so it is unlabelled when mirrored.
PERM = list(range(len(LABELS)))
for _a, _b in (('Medial Meniscus', 'Lateral Meniscus'), ('Medial OA', 'Lateral OA')):
    PERM[LABELS.index(_a)], PERM[LABELS.index(_b)] = LABELS.index(_b), LABELS.index(_a)
MCL = LABELS.index('MCL')


def mirror(x):
    """[B,K,n,H,W] study volumes -> the same studies as the contralateral knee (its own inverse)."""
    import torch
    return torch.stack([x[:, k].flip(1) if SLOTS[k][0] == 'Sagittal' else x[:, k].flip(-1) for k in range(len(SLOTS))], 1)


def mirror_targets(t):
    """Targets of mirrored studies: medial <-> lateral columns swapped, MCL NaN (no loss)."""
    t = t[:, PERM].clone()
    t[:, MCL] = float('nan')
    return t


def flip_merge(z, zm):
    """Flip TTA in logit space: mean of the plain pass and the mirrored pass mapped back by PERM; MCL from the plain pass."""
    out = (z + zm[:, PERM]) / 2
    out[:, MCL] = z[:, MCL]
    return out


def make_model(pretrained, backbone=None, head=None, img=None):
    """head 'mil': per-finding attention over every (slot, slice) token of the study, with slot and
    slice-position embeddings (Raptor/E11 style). 'slot': the original order-blind per-slot pooling,
    kept so models trained before HEAD existed still load."""
    import torch, torch.nn as nn, timm

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            bb = backbone or BACKBONE
            self.mil = (head or HEAD) == 'mil'
            if bb == RAD:  # RadImageNet ResNet-50 (torchvision layout, weights from marwanmath's public dataset)
                import torchvision
                body = nn.Sequential(*list(torchvision.models.resnet50(weights=None).children())[:-2])
                if pretrained:
                    w = f"{find_input('resnet-50-radimagenet-marwan')}/ResNet50.pt"
                    sd = torch.load(w, map_location='cpu', weights_only=True)
                    body.load_state_dict({k.removeprefix('backbone.'): v for k, v in sd.items()})  # saved as .backbone
                self.enc = nn.Sequential(body, nn.AdaptiveAvgPool2d(1), nn.Flatten())
                d = 2048
            else:  # DINOv2 ViTs default to 518 px; resample their position embedding to our input size
                kw = {'img_size': img or IMG} if 'dinov2' in bb else {}
                self.enc = timm.create_model(bb, pretrained=pretrained, num_classes=0, **kw)
                d = self.enc.num_features
            self.slot_emb = nn.Parameter(torch.zeros(len(SLOTS), d))
            if self.mil:
                n = len(LABELS)
                self.norm = nn.LayerNorm(d)
                self.pos_emb = nn.Embedding(POS_BINS, d)
                nn.init.zeros_(self.pos_emb.weight)
                self.att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, n))
                self.clsW = nn.Parameter(torch.zeros(n, d))
                self.clsb = nn.Parameter(torch.zeros(n))
                nn.init.trunc_normal_(self.clsW, std=0.02)
                self.d = d
            else:
                self.att = nn.Linear(d, 1)
                self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(len(SLOTS) * d, len(LABELS)))

        def forward(self, x, m, pos=None):  # x float [B,K,S,3,IMG,IMG], m [B,K], pos [S] in [0,1] (mil)
            if self.mil:
                return self.forward_mil(x, m, pos)
            B, K, S = x.shape[:3]
            # autocast here, not only in the caller: DataParallel runs replicas in threads that don't inherit it
            with torch.autocast('cuda', dtype=torch.float16, enabled=x.is_cuda):
                f = self.enc(x.flatten(0, 2))
            f = f.float().view(B, K, S, -1) + self.slot_emb[None, :, None]
            a = self.att(f)
            if self.training:  # drop ~15% of slice tokens per slot (never all of them)
                drop = torch.rand(B, K, S, 1, device=a.device) < 0.15
                drop[:, :, :1] &= drop[:, :, 1:].all(2, keepdim=True).logical_not()
                a = a.masked_fill(drop, -1e4)
            w = a.softmax(2)
            f = (w * f).sum(2) * m[..., None]
            return self.head(f.flatten(1))

        def forward_mil(self, x, m, pos):
            B, K, S = x.shape[:3]
            keep = m.bool()  # [B,K] present slots; missing ones are never encoded (BatchNorm sees real images only)
            if self.training and SLOT_DROP > 0:  # whole-series dropout, keeping at least one series per study
                drop = (torch.rand(B, K, device=x.device) < SLOT_DROP) & keep
                drop &= (keep & ~drop).any(1, keepdim=True)
                keep = keep & ~drop
            f = torch.zeros(B, K, S, self.d, device=x.device)
            if keep.any():
                with torch.autocast('cuda', dtype=torch.float16, enabled=x.is_cuda):
                    fp = self.enc(x[keep].flatten(0, 1))
                f[keep] = fp.float().view(-1, S, self.d)
            bins = (pos * (POS_BINS - 1)).round().long()
            h = self.norm(f) + self.slot_emb[None, :, None] + self.pos_emb(bins)[None, None]
            a = self.att(h).float()  # [B,K,S,12]: one attention logit per finding per token
            tok = keep[:, :, None].expand(B, K, S)
            if self.training:  # drop ~15% of tokens, but never every token of a study
                drop = (torch.rand(B, K, S, device=x.device) < 0.15) & tok
                drop &= ((tok & ~drop).flatten(1).any(1))[:, None, None]
                tok = tok & ~drop
            a = a.masked_fill(~tok[..., None], -1e4)
            w = a.flatten(1, 2).softmax(1)  # [B,T,12] softmax over all tokens of the study, per finding
            pooled = torch.einsum('btn,btd->bnd', w, h.flatten(1, 2))
            if self.training and AUX_W > 0:  # per-token classifier, max over the study's tokens (MIL max-pooling)
                t = torch.einsum('bksd,nd->bksn', h, self.clsW) + self.clsb
                self.aux = t.masked_fill(~tok[..., None], -1e4).flatten(1, 2).amax(1)
            return (pooled * self.clsW).sum(-1) + self.clsb
    return Net()


def to_input(x, train, backbone=None, img=None):
    """uint8 [B,K,n,SIZE,SIZE] on GPU -> (float [B,K,S,3,IMG,IMG], centre positions [S] in [0,1]) (2.5D stacks,
    GPU aug), S centres = every
    other slice (8 for the 16-slice cache, 15 for 31). Training on a 31-slice cache uses a random half of
    the centres: same cost as 16, every slice still seen; inference uses all of them.
    RadImageNet gets the centre slice repeated x3 and [-1,1] scaling instead of neighbours + ImageNet stats."""
    rad = (backbone or BACKBONE) == RAD
    img = img or IMG
    import torch, torch.nn.functional as F
    x = x.float() / 255
    B, K, n = x.shape[:3]
    c = torch.arange(1, n, 2, device=x.device)
    if train:  # shift which slices are centres by -1/0/+1 for the whole batch
        c = (c + int(torch.randint(-1, 2, (1,)))).clamp(1, n - 2)
        if n > 16:
            c = c[int(torch.randint(0, 2, (1,)))::2]
    pos = c.float() / (n - 1)
    if rad:
        x = x[:, :, c].unsqueeze(3).expand(-1, -1, -1, 3, -1, -1)
    else:
        x = torch.stack([x[:, :, c - 1], x[:, :, c], x[:, :, (c + 1).clamp(max=n - 1)]], 3)  # B,K,S,3,H,W
    x = x.flatten(0, 1)  # one affine per (study, slot), shared by its slices
    n = x.shape[0]
    theta = torch.zeros(n, 2, 3, device=x.device)
    if train:  # small rotation/scale/shift; no left-right flip (medial/lateral matters)
        ang = (torch.rand(n, device=x.device) - 0.5) * np.deg2rad(2 * ROT_DEG)
        sc = 1 + (torch.rand(n, device=x.device) - 0.5) * 0.2
        theta[:, 0, 0], theta[:, 0, 1] = torch.cos(ang) / sc, -torch.sin(ang) / sc
        theta[:, 1, 0], theta[:, 1, 1] = torch.sin(ang) / sc, torch.cos(ang) / sc
        theta[:, :, 2] = (torch.rand(n, 2, device=x.device) - 0.5) * 0.1
    else:
        theta[:, 0, 0] = theta[:, 1, 1] = 1
    S = x.shape[1]
    flat = x.flatten(1, 2)  # n, 8*3, H, W
    grid = F.affine_grid(theta, (n, flat.shape[1], img, img), align_corners=False)
    flat = F.grid_sample(flat, grid, align_corners=False)
    if train:  # gamma 0.8-1.25 then +-10% contrast, per (study, slot)
        g = torch.exp((torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.45)
        flat = flat.clamp(0, 1) ** g * (1 + (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.2)
    if rad:
        return (flat * 2 - 1).view(B, K, S, 3, img, img), pos
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).repeat(S)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).repeat(S)[None, :, None, None]
    return ((flat - mean) / std).view(B, K, S, 3, img, img), pos


# ---------------------------------------------------------------- modes
def run_prep():
    comp = find_input('rsna-knee-abnormality-detection')
    studies = pd.read_csv(f'{comp}/train.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
    studies = studies[SHARD[0]::SHARD[1]]
    if SMOKE:
        studies = studies[:SMOKE]
    out = '/kaggle/working/cache'
    os.makedirs(out, exist_ok=True)
    for _ in prep_all('train', studies, out):
        pass
    json.dump({'n': len(studies), 'N_SLICE': N_SLICE, 'SIZE': SIZE, 'CROP_MM': CROP_MM, 'PREP': PREP},
              open('/kaggle/working/prep_done.json', 'w'))


def load_targets():
    comp = find_input('rsna-knee-abnormality-detection')
    t = pd.read_csv(f'{comp}/train.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    gold = t[LABELS].dropna()
    y = pd.read_csv(f"{find_input('rsna-knee-llm-report-labels')}/llm_labels_v4_blend.csv",
                    dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[LABELS].astype(float)
    y = y.reindex(t.index).fillna(0.5)
    # Report silent on a finding radiologists reliably mention -> almost always absent in the 58 gold
    # (Baker's 3%, Medial OA 0%, ...); v4b encodes that silence as 0.25-0.5, so pull it down.
    raw = pd.read_csv(f"{find_input('rsna-knee-llm-report-labels')}/llm_labels_full.csv",
                      dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[LABELS].reindex(t.index)
    for lab, v in SILENT_FIX.items():
        y.loc[raw[lab] == 0.5, lab] = v
    y_eval = y.copy()  # v4b report labels: what every run is scored against, whatever it trains on
    if LABEL_SRC == 'rubric':  # P(image-read positive under the host rubric), one LLM read per unique report
        y = pd.read_csv(f"{find_input('knee-rubric-labels')}/rubric_labels.csv", dtype={'StudyInstanceUID': str}
                        ).set_index('StudyInstanceUID')[LABELS].astype(float).reindex(t.index).fillna(y)
    if LABEL_SRC == 'consensus':  # mean of 4 LLM reads; NaN = reads disagree, no loss on that cell (masked_bce)
        y = pd.read_csv(f"{find_input('knee-rubric-labels')}/consensus_labels.csv", dtype={'StudyInstanceUID': str}
                        ).set_index('StudyInstanceUID')[LABELS].astype(float).reindex(t.index)
    if PSEUDO_A > 0:
        oofs = [pd.read_csv(f'{find_input(k)}/oof.csv', index_col=0, dtype={0: str})[LABELS] for k in PSEUDO_FROM]
        o = pd.concat(oofs).groupby(level=0).mean().reindex(y.index)  # teachers averaged; split-fold kernels merged
        y = ((1 - PSEUDO_A) * y + PSEUDO_A * o).fillna(y)
        log(f'pseudo targets: a={PSEUDO_A} from {PSEUDO_FROM}, {o.notna().all(1).sum()} studies with OOF')
    y.loc[gold.index] = y_eval.loc[gold.index] = gold.values
    return y, y_eval, gold


def masked_bce(logits, t, pos_weight=None):
    """BCE-with-logits averaged over the non-NaN targets only (equals the plain mean when nothing is NaN)."""
    import torch, torch.nn.functional as F
    ok = ~torch.isnan(t)
    loss = F.binary_cross_entropy_with_logits(logits, t.nan_to_num(0.5), pos_weight=pos_weight, reduction='none')
    return (loss * ok).sum() / ok.sum().clamp(min=1)


def report_folds(k=5, seed=42):
    """study -> fold, grouping identical (normalised) reports so templates never straddle folds."""
    import re
    comp = find_input('rsna-knee-abnormality-detection')
    t = pd.read_csv(f'{comp}/train.csv', dtype={'StudyInstanceUID': str})
    key = t.Report.fillna('').str.lower().map(lambda r: re.sub(r'\W+', ' ', r).strip())
    groups = key.unique()
    np.random.default_rng(seed).shuffle(groups)
    g2f = {g: i % k for i, g in enumerate(groups)}
    return pd.Series(key.map(g2f).values, index=t.StudyInstanceUID)


def macro_auc(yt, p):
    from sklearn.metrics import roc_auc_score
    return np.mean([roc_auc_score(yt[:, j] > 0.5, p[:, j]) for j in range(len(LABELS))
                    if 0 < (yt[:, j] > 0.5).sum() < len(yt)])


def train_setup():
    """Everything a training process needs, rebuilt cheaply in each spawned GPU worker."""
    y, y_eval, gold = load_targets()
    dirs = [f"{find_input(c)}/cache" for c in CACHE.split(',')]
    path = {s[:-4]: f'{d}/{s}' for d in dirs for s in os.listdir(d)}
    ids = np.array(sorted(path))
    ids = ids[np.isin(ids, y.index)]
    if SMOKE:  # speed/plumbing check; repeats a small cache to SMOKE samples
        ids = np.resize(ids, SMOKE) if len(ids) < SMOKE else ids[:SMOKE]
    paths = np.array([path[i] for i in ids])
    done = [json.load(open(f'{os.path.dirname(d)}/prep_done.json')) for d in dirs]
    pv = {(j.get('PREP', 1), j.get('CROP_MM', 150.0)) for j in done}
    assert len(pv) == 1, f'cache shards from different prep versions / crops: {pv}'
    (prep, crop_mm), = pv
    shape = np.load(paths[0])['x'].shape
    return dict(paths=paths, n_slice=int(shape[1]), size=int(shape[-1]), prep=prep, crop_mm=crop_mm, ids=ids, gold=gold,
                fold_of=report_folds().reindex(ids).values,
                Y=y.loc[ids].values.astype(np.float32), YE=y_eval.loc[ids].values.astype(np.float32),
                folds=FOLD_LIST or ([0] if SMOKE else [0, 1, 2, 3, 4]), epochs=2 if SMOKE else EPOCHS)


def train_worker(rank, ngpu, t_start):
    """Train folds[rank::ngpu] on cuda:rank. One process per GPU: DataParallel was 11x slower for EfficientNet."""
    import torch, torch.nn.functional as F
    torch.backends.cudnn.benchmark = True
    S = train_setup()
    ids, Y, YE, fold_of, epochs, paths = S['ids'], S['Y'], S['YE'], S['fold_of'], S['epochs'], S['paths']
    dev = f'cuda:{rank}'
    torch.cuda.set_device(dev)
    bs = 2 if 'coatnet' in BACKBONE or 'convnext' in BACKBONE else 4  # per GPU; coatnet ~107MB activations/image; convnext-T at 288 OOMs at 4
    budget, fold_secs = 11.3 * 3600, []  # Kaggle kills at 12h and then saves nothing

    class DS(torch.utils.data.Dataset):
        def __init__(self, idx): self.idx = idx
        def __len__(self): return len(self.idx)
        def __getitem__(self, i):
            z = np.load(paths[self.idx[i]])
            return torch.from_numpy(z['x']), torch.from_numpy(z['m']).float(), torch.from_numpy(Y[self.idx[i]])

    def predict(model, idx):
        model.eval()
        out = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            for x, m, _ in torch.utils.data.DataLoader(DS(idx), batch_size=bs, num_workers=2, multiprocessing_context='fork'):
                x, m = x.to(dev), m.to(dev)
                xi, pos = to_input(x, False)
                z = model(xi, m, pos).float()
                if MIRROR > 0:  # flip TTA, exactly as run_infer does for this model
                    z = flip_merge(z, model(to_input(mirror(x), False)[0], m, pos).float())
                out.append(z.sigmoid().cpu().numpy())
        return np.concatenate(out)

    for fold in S['folds'][rank::ngpu]:
        if fold_secs and time.time() - t_start + np.mean(fold_secs) > budget:
            log(f'[gpu{rank}] time budget: skipping fold {fold} (train it in another kernel)')
            break
        t_fold = time.time()
        tr, va = np.where(fold_of != fold)[0], np.where(fold_of == fold)[0]
        model = make_model(True).to(dev)
        # attention backbones and the RadImageNet trunk want gentler LR; ImageNet CNNs underfit at 1e-4
        # (resnet50.a1_in1k: 0.848 OOF after 8 epochs, still rising)
        # DINOv2: full fine-tune at a ViT-friendly 3e-5 (pair with headlr=1e-3 so the new head still learns fast)
        lr = 3e-5 if 'dinov2' in BACKBONE else 1e-4 if BACKBONE == RAD or 'coatnet' in BACKBONE or 'convnext' in BACKBONE else 3e-4
        named = list(model.named_parameters())
        groups = [{'params': [p for n, p in named if n.startswith('enc.')], 'lr': lr},
                  {'params': [p for n, p in named if not n.startswith('enc.')], 'lr': HEAD_LR or lr}]
        opt = torch.optim.AdamW(groups, weight_decay=1e-2)
        dl = torch.utils.data.DataLoader(DS(tr), batch_size=bs, shuffle=True, num_workers=2, drop_last=True,
                                         persistent_workers=True, multiprocessing_context='fork')  # workers never touch CUDA
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g['lr'] for g in groups], total_steps=epochs * len(dl),
                                                    pct_start=0.1)
        pw = None
        if POS_W:
            prev = np.clip(np.nanmean(Y[tr], 0), 0.03, 0.7)
            pw = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev)
        scaler = torch.amp.GradScaler()
        over = False
        for ep in range(epochs):
            model.train()
            tot, t_wait = 0, 0.0
            t0_ep = tick = time.time()
            for x, m, t in dl:
                t_wait += time.time() - tick
                x, t = x.to(dev), t.to(dev)
                if MIRROR > 0:  # a random part of the batch becomes the contralateral knee
                    sel = torch.rand(len(x), device=dev) < MIRROR
                    x[sel], t[sel] = mirror(x[sel]), mirror_targets(t[sel])
                with torch.autocast('cuda', dtype=torch.float16):
                    xi, pos = to_input(x, True)
                    loss = masked_bce(model(xi, m.to(dev), pos).float(), t, pw)
                    if AUX_W > 0:
                        loss = loss + AUX_W * masked_bce(model.aux.float(), t)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot += loss.item()
                tick = time.time()
            p = predict(model, va)
            log(f'[gpu{rank}] fold {fold} ep {ep} loss {tot / len(dl):.4f} val macroAUC(vs labels) '
                f'{macro_auc(YE[va], p):.4f} | data-wait {t_wait:.0f}s of {time.time() - t0_ep:.0f}s')
            if time.time() - t_start > budget:
                over = True
                break
        if over:
            log(f'[gpu{rank}] time budget hit inside fold {fold}; discarding it')
            break
        torch.save({k: v.half() for k, v in model.state_dict().items()}, f'/kaggle/working/fold{fold}.pt')
        pd.DataFrame(p, index=ids[va], columns=LABELS).to_csv(f'/kaggle/working/oof_fold{fold}.csv')
        fold_secs.append(time.time() - t_fold)
        log(f'[gpu{rank}] fold {fold} saved in {fold_secs[-1] / 60:.0f} min')
        del model, opt, dl
        torch.cuda.empty_cache()


def run_train():
    import torch
    ngpu = max(1, torch.cuda.device_count())
    S = train_setup()
    log(f'{BACKBONE}: {len(S["ids"])} studies, folds {S["folds"]}, {S["epochs"]} epochs, {ngpu} GPUs, '
        f'pseudo {PSEUDO_A} {PSEUDO_FROM}')
    t_start = time.time()
    if ngpu > 1 and len(S['folds']) > 1:
        import torch.multiprocessing as mp
        mp.spawn(train_worker, args=(ngpu, t_start), nprocs=min(ngpu, len(S['folds'])), join=True)
    else:
        train_worker(0, 1, t_start)
    parts = sorted(glob.glob('/kaggle/working/oof_fold*.csv'))
    trained = sorted(int(f[-5]) for f in parts)
    json.dump({'BACKBONE': BACKBONE, 'IMG': IMG, 'SMOKE': SMOKE, 'FOLDS': trained, 'EPOCHS': S['epochs'],
               'PSEUDO': [PSEUDO_A, PSEUDO_FROM], 'N_SLICE': S['n_slice'], 'SIZE': S['size'], 'HEAD': HEAD, 'PREP': S['prep'], 'CROP_MM': S['crop_mm'], 'LABEL_SRC': LABEL_SRC,
               'SLOT_DROP': SLOT_DROP, 'HEAD_LR': HEAD_LR, 'POS_W': POS_W, 'ROT_DEG': ROT_DEG, 'AUX_W': AUX_W, 'MIRROR': MIRROR},
              open('/kaggle/working/config.json', 'w'))
    if not parts:
        log('no fold finished')
        return
    oof = pd.concat([pd.read_csv(f, index_col=0) for f in parts])
    oof = oof[~oof.index.duplicated()]  # SMOKE repeats studies
    oof.to_csv('/kaggle/working/oof.csv')
    pos = pd.Series(np.arange(len(S['ids'])), index=S['ids'])
    pos = pos[~pos.index.duplicated()]
    rows = pos.reindex(oof.index).values
    log(f'trained folds {trained}; OOF macroAUC vs labels: {macro_auc(S["YE"][rows], oof.values):.4f}')
    g = [i for i, s in enumerate(oof.index) if s in S['gold'].index]
    if g:
        log(f'OOF macroAUC vs gold ({len(g)} studies): {macro_auc(S["YE"][rows[g]], oof.values[g]):.4f}')


def run_infer():
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    comp = find_input('rsna-knee-abnormality-detection')
    studies = pd.read_csv(f'{comp}/test.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
    cfgs = sorted(set(glob.glob('/kaggle/input/*/config.json') + glob.glob('/kaggle/input/notebooks/*/*/config.json')))
    models = []  # every fold of every attached training kernel of ours, equally weighted
    for cf in cfgs:
        wdir, cfg = os.path.dirname(cf), json.load(open(cf))
        if not {'BACKBONE', 'FOLDS'} <= set(cfg):  # some other notebook's config
            continue
        bb = cfg['BACKBONE']
        if cfg.get('SMOKE', 1) and not ALLOW_SMOKE:
            log(f'skipping smoke-test kernel {wdir}')
            continue
        ns = cfg.get('N_SLICE', 16)  # kernels from before the 31-slice cache don't record it
        pv = cfg.get('PREP', 1)  # ... nor the prep geometry version
        assert ns in (N_SLICE, (N_SLICE + 1) // 2) and (pv == 1 or ns == N_SLICE), f'{wdir}: {ns} slices, prep {pv}'
        sz, im = cfg.get('SIZE', 256), cfg.get('IMG', 224)  # cache px (absent before 384-cache runs) -> network px
        cm = cfg.get('CROP_MM', 150.0)  # absent before the zoomed cache
        for f in sorted(glob.glob(f'{wdir}/fold*.pt')):
            mdl = make_model(False, bb, cfg.get('HEAD', 'slot'), im)
            mdl.load_state_dict({k: v.float() for k, v in torch.load(f, map_location='cpu', weights_only=True).items()})
            mdl.flip_tta = cfg.get('MIRROR', 0) > 0  # trained with the mirror -> scored on both views, as its OOF was
            models.append(((bb, ns, pv, sz, cm, im), mdl.to(dev).eval()))
        log(f'{wdir}: {bb}, {ns} slices, prep v{pv}, {sz}px/{cm}mm cache -> {im}px, flip TTA {cfg.get("MIRROR", 0) > 0}')
    log(f'{len(models)} fold models')
    assert models, 'no trained knee-train* models attached'
    global PREP, SIZE, CROP_MM
    psum, t0 = {s: 0.0 for s in studies}, time.time()
    for pv, sz, cm in sorted({k[2:5] for k, _ in models}):  # one test prep per (geometry, cache px, crop); forked workers read the globals
        PREP, SIZE, CROP_MM = pv, sz, cm
        group = [(k, mdl) for k, mdl in models if k[2:5] == (pv, sz, cm)]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=dev == 'cuda'):
            for s, x, m in prep_all('test', studies):
                if BLEND > 0 and uptime_h() > 8.7:  # Kaggle kills at 9h; the public submission.csv is still intact
                    raise TimeoutError(f'uptime {uptime_h():.2f}h, abandoning our part')
                xb = torch.from_numpy(x[None]).to(dev)
                mi = torch.from_numpy(m[None]).float().to(dev)
                keys = {(bb, ns, im, f) for (bb, ns, _, _, _, im), mdl in group for f in {False, mdl.flip_tta}}
                xin = {(bb, ns, im, f): to_input((mirror(xb) if f else xb)[:, :, ::1 if ns == N_SLICE else 2], False, bb, im)
                       for bb, ns, im, f in keys}
                for k, mdl in group:
                    xi, pos = xin[(k[0], k[1], k[5], False)]
                    z = mdl(xi, mi, pos).float()
                    if mdl.flip_tta:
                        z = flip_merge(z, mdl(xin[(k[0], k[1], k[5], True)][0], mi, pos).float())
                    psum[s] = psum[s] + z.sigmoid().cpu().numpy()[0]
        log(f'prep v{pv} {sz}px {cm}mm: {len(group)} models done at {time.time() - t0:.0f}s on {dev}')
    ours = pd.DataFrame([psum[s] / len(models) for s in studies], columns=LABELS, index=studies)
    assert np.isfinite(ours.values).all(), 'non-finite predictions'  # blend mode: raising keeps the public csv
    ours = ours.rank(pct=True)
    path = '/kaggle/working/submission.csv'
    if BLEND > 0 and os.path.exists(path):  # rank-blend with whatever the earlier cells wrote
        pub = pd.read_csv(path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[LABELS]
        ours = BLEND * pub.loc[studies].rank(pct=True) + (1 - BLEND) * ours
        log(f'blended with existing submission, weight {BLEND}')
    ours.index.name = 'StudyInstanceUID'
    ours.reset_index().to_csv(path, index=False)
    log('submission written', ours.shape)


if __name__ == '__main__':
    if MODE == 'infer' and BLEND > 0 and os.path.exists('/kaggle/working/submission.csv'):
        log(f'public stack finished at {uptime_h():.2f}h uptime')
        try:  # appended to the public notebook: on any failure keep its submission.csv untouched
            if uptime_h() > 8.2:
                raise TimeoutError('not enough time left before the 9h limit')
            run_infer()
        except Exception as e:
            log('knee infer FAILED, keeping the public submission.csv:', repr(e))
    else:
        {'prep': run_prep, 'train': run_train, 'infer': run_infer}[MODE]()
