
# ======== CELL 0 [markdown] ========
# RSNA Knee | DINOsaur V5? 🦖

## ConvNeXt diversity-arm trainer

# ======== CELL 1 [code] ========
from __future__ import annotations
import os, gc, json, math, random, glob, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 20260921
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

TARGETS = [
    "ACL","MCL","Medial Meniscus","Lateral Meniscus",
    "Medial OA","Lateral OA","PF OA","Effusion","Synovitis",
    "Baker's","Contusion","Fracture",
]
N_TARGETS = len(TARGETS)

WORK = Path("/kaggle/working")
FEATURE_CACHE = WORK / "dinov5_convnext_features"
FEATURE_CACHE.mkdir(parents=True, exist_ok=True)

def find_comp_root():
    for p in (
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ):
        if (p/"train.csv").is_file() and (p/"train_series.csv").is_file():
            return p
    raise FileNotFoundError("RSNA Knee competition input is not mounted")

ROOT = find_comp_root()
TRAIN = pd.read_csv(ROOT/"train.csv", dtype={"StudyInstanceUID":str})
SERIES = pd.read_csv(
    ROOT/"train_series.csv",
    dtype={"StudyInstanceUID":str, "SeriesInstanceUID":str},
)
TRAIN["StudyInstanceUID"] = TRAIN["StudyInstanceUID"].astype(str)
SERIES["StudyInstanceUID"] = SERIES["StudyInstanceUID"].astype(str)
SERIES["SeriesInstanceUID"] = SERIES["SeriesInstanceUID"].astype(str)
UIDS = TRAIN["StudyInstanceUID"].tolist()
UID_TO_INDEX = {u:i for i,u in enumerate(UIDS)}

print("studies:", len(TRAIN), "series:", len(SERIES))
print("GPU count:", torch.cuda.device_count())


# ======== CELL 2 [code] ========
# ----------------------- public soft-label consensus --------------------------
ALIASES = {
    "acl":"ACL", "mcl":"MCL",
    "medial meniscus":"Medial Meniscus", "medial_meniscus":"Medial Meniscus",
    "lateral meniscus":"Lateral Meniscus", "lateral_meniscus":"Lateral Meniscus",
    "medial oa":"Medial OA", "medial_oa":"Medial OA",
    "lateral oa":"Lateral OA", "lateral_oa":"Lateral OA",
    "pf oa":"PF OA", "pf_oa":"PF OA", "patellofemoral oa":"PF OA",
    "effusion":"Effusion", "synovitis":"Synovitis",
    "baker's":"Baker's", "bakers":"Baker's", "baker cyst":"Baker's",
    "contusion":"Contusion", "fracture":"Fracture",
}

def target_name(col):
    raw = str(col).strip()
    if raw in TARGETS:
        return raw
    key = raw.lower().replace("-", " ").replace("__","_")
    return ALIASES.get(key)

def read_label_csv(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    uid_col = next(
        (c for c in df.columns if str(c).strip().lower() in {
            "studyinstanceuid","study_instance_uid","studyuid","study_id","studyid"
        }),
        None,
    )
    if uid_col is None:
        return None
    rename = {uid_col:"StudyInstanceUID"}
    for c in df.columns:
        t = target_name(c)
        if t:
            rename[c] = t
    df = df.rename(columns=rename)
    have = [t for t in TARGETS if t in df.columns]
    if len(have) < 8:
        return None
    out = df[["StudyInstanceUID"] + have].copy()
    out["StudyInstanceUID"] = out["StudyInstanceUID"].astype(str)
    for t in have:
        out[t] = pd.to_numeric(out[t], errors="coerce")
        out.loc[~out[t].between(0,1), t] = np.nan
    return out


# Group CSVs by Kaggle dataset root before taking the final consensus.
# This avoids giving one provider 2-3x the vote just because the dataset ships
# multiple closely related CSV variants.
family_sources = {}
source_paths = []
competition_train = str((ROOT/"train.csv").resolve())

def dataset_family(path):
    parts=Path(path).parts
    try:
        k=parts.index("datasets")
        return "/".join(parts[k:k+2])
    except Exception:
        # Fallback: parent directory is still better than treating every CSV
        # as an independent annotator.
        return str(Path(path).parent)

raw_by_family={}
for path in glob.glob("/kaggle/input/**/*.csv",recursive=True):
    low=path.lower()
    if not any(k in low for k in ("label","llm","pseudo","report","target","soft")):
        continue
    try:
        if str(Path(path).resolve()) == competition_train:
            continue
    except Exception:
        pass

    df=read_label_csv(path)
    if df is None:
        continue

    overlap=df["StudyInstanceUID"].isin(UIDS).mean()
    if overlap<0.50:
        continue

    fam=dataset_family(path)
    raw_by_family.setdefault(fam,[]).append(df)
    source_paths.append(path)

idx=pd.Index(UIDS,name="StudyInstanceUID")
family_frames=[]

for fam,frames in raw_by_family.items():
    fam_out=pd.DataFrame({"StudyInstanceUID":UIDS})
    fam_out=fam_out.set_index("StudyInstanceUID")

    for t in TARGETS:
        cols=[]
        for frame in frames:
            if t in frame.columns:
                s=frame.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
                cols.append(s.reindex(idx)[t].to_numpy(np.float32))
        if cols:
            fam_out[t]=np.nanmedian(np.stack(cols,axis=1),axis=1)

    family_frames.append((fam,fam_out))
    print("label family:",fam,"csvs=",len(frames))

print("label CSVs discovered:",len(source_paths))
print("independent label families:",len(family_frames))

pseudo=np.full((len(UIDS),N_TARGETS),np.nan,np.float32)
source_count=np.zeros((len(UIDS),N_TARGETS),np.int16)

for j,t in enumerate(TARGETS):
    cols=[]
    for fam,frame in family_frames:
        if t in frame.columns:
            cols.append(frame[t].to_numpy(np.float32))
    if cols:
        arr=np.stack(cols,axis=1)
        pseudo[:,j]=np.nanmedian(arr,axis=1)
        source_count[:,j]=np.isfinite(arr).sum(axis=1)

gold = np.full_like(pseudo, np.nan)
for j,t in enumerate(TARGETS):
    if t in TRAIN.columns:
        gold[:,j] = pd.to_numeric(TRAIN[t], errors="coerce").to_numpy(np.float32)

gold_mask = np.isfinite(gold)
targets = np.where(gold_mask, gold, pseudo).astype(np.float32)

# Keep pseudo labels soft, but gently sharpen indecisive probabilities.
valid = np.isfinite(targets)
p = np.clip(np.where(valid, targets, 0.5), 1e-4, 1-1e-4)
logit = np.log(p/(1-p))
soft_sharp = 1/(1+np.exp(-logit/0.85))
targets = np.where(gold_mask, targets, soft_sharp).astype(np.float32)
targets[~valid] = np.nan

max_sources = max(len(family_frames), 1)
coverage = np.clip(source_count / max_sources, 0, 1)
certainty = np.clip(2*np.abs(np.nan_to_num(targets, nan=0.5)-0.5), 0, 1)
weights = (0.25 + 0.75*(0.55*coverage + 0.45*certainty)).astype(np.float32)
weights[~np.isfinite(targets)] = 0.0
weights[gold_mask] = 2.0

gold_study_mask = np.any(gold_mask, axis=1)
print("expert-labelled studies:", int(gold_study_mask.sum()))
print("pseudo-covered studies:", int(np.any(np.isfinite(pseudo), axis=1).sum()))

if np.any(np.isfinite(pseudo), axis=1).sum() < 3000:
    warnings.warn(
        "Pseudo-label coverage is low. Attach Steven/GPT public report-label datasets."
    )


# ======== CELL 3 [code] ========
# ---------- stable weak-label MRI adaptation -> ConvNeXt study features -------
import timm
import cv2

PLANE_TO_ID = {"Sagittal":0, "Coronal":1, "Axial":2}
MAX_SERIES = 8
CLIP_Q = (0.15, 0.325, 0.50, 0.675, 0.85)
CROP_MM = 150.0
IMAGE_SIZE = 224
FEATURE_DIM = 768

series_by_study = {k:v.copy() for k,v in SERIES.groupby("StudyInstanceUID")}

def choose_series(frame, max_series=MAX_SERIES):
    f = frame.copy()
    for c in ("Fluid_Sensitive","Fat_Suppression"):
        if c not in f.columns:
            f[c] = 0
        f[c] = pd.to_numeric(f[c], errors="coerce").fillna(0).astype(int)
    f["Anatomical_Plane"] = f["Anatomical_Plane"].astype(str)

    chosen, used = [], set()
    for plane in ("Sagittal","Coronal","Axial"):
        z = f[f["Anatomical_Plane"].eq(plane)]
        if len(z):
            z = z.sort_values(
                ["Fluid_Sensitive","Fat_Suppression"], ascending=False
            )
            r = z.iloc[0]
            chosen.append(r)
            used.add(str(r["SeriesInstanceUID"]))

    for fluid,fat in ((1,1),(1,0),(0,1),(0,0)):
        z = f[
            (f["Fluid_Sensitive"]==fluid)
            & (f["Fat_Suppression"]==fat)
        ]
        for _,r in z.iterrows():
            sid = str(r["SeriesInstanceUID"])
            if sid not in used:
                chosen.append(r)
                used.add(sid)
                break

    for _,r in f.iterrows():
        if len(chosen) >= max_series:
            break
        sid = str(r["SeriesInstanceUID"])
        if sid not in used:
            chosen.append(r)
            used.add(sid)

    return pd.DataFrame(chosen[:max_series]) if chosen else f.head(0)

def ordered_files(series_dir):
    values = []
    for fp in Path(series_dir).glob("*.dcm"):
        try:
            h = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
            iop = getattr(h, "ImageOrientationPatient", None)
            ipp = getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop)==6:
                row = np.asarray(iop[:3], np.float64)
                col = np.asarray(iop[3:6], np.float64)
                normal = np.cross(row,col)
                pos = float(
                    np.dot(np.asarray(ipp[:3],np.float64), normal)
                )
            else:
                pos = float(getattr(h,"InstanceNumber",0) or 0)
            values.append((pos, str(fp)))
            del h
        except Exception:
            pass
    values.sort(key=lambda z:z[0])
    return [fp for _,fp in values]

def read_slice_resized_uint8(fp):
    ds = pydicom.dcmread(str(fp), force=True)
    arr = ds.pixel_array.astype(np.float32, copy=False)
    slope = float(getattr(ds,"RescaleSlope",1.0) or 1.0)
    intercept = float(getattr(ds,"RescaleIntercept",0.0) or 0.0)
    if slope != 1.0 or intercept != 0.0:
        arr = arr*slope + intercept
    if str(getattr(ds,"PhotometricInterpretation","")) == "MONOCHROME1":
        arr = arr.max() - arr

    ps = getattr(ds,"PixelSpacing",None)
    spacing = float(ps[0]) if ps is not None else None
    if spacing and spacing > 0:
        side = int(round(CROP_MM/spacing))
        side = max(16,min(side,arr.shape[0],arr.shape[1]))
        cy,cx = arr.shape[0]//2, arr.shape[1]//2
        y0=max(0,cy-side//2)
        x0=max(0,cx-side//2)
        arr=arr[y0:y0+side,x0:x0+side]

    lo,hi=np.percentile(arr,[1,99])
    arr=np.clip((arr-lo)/(hi-lo+1e-6),0,1)
    arr=cv2.resize(
        arr,
        (IMAGE_SIZE,IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    out=np.clip(arr*255.0+0.5,0,255).astype(np.uint8)
    del ds, arr
    return out

def triplet_uint8(files, center):
    n=len(files)
    ids=[max(0,min(n-1,center+d)) for d in (-1,0,1)]
    return np.stack(
        [read_slice_resized_uint8(files[i]) for i in ids],
        axis=0,
    )

IM_MEAN = torch.tensor([0.485,0.456,0.406],dtype=torch.float32).view(3,1,1)
IM_STD = torch.tensor([0.229,0.224,0.225],dtype=torch.float32).view(3,1,1)

def normalize_uint8(x):
    x=x.float().div_(255.0)
    return (x-IM_MEAN)/IM_STD

def safe_float(v, default=0.0):
    z=pd.to_numeric(v,errors="coerce")
    return float(default if pd.isna(z) else z)

def meta_vector(row,q):
    plane=np.zeros(3,np.float32)
    plane[PLANE_TO_ID.get(str(row.get("Anatomical_Plane","")),2)] = 1.0
    return np.array(
        [
            *plane,
            safe_float(row.get("Fluid_Sensitive",0)),
            safe_float(row.get("Fat_Suppression",0)),
            float(q),
        ],
        np.float32,
    )

# ----------------------------- offline backbone -------------------------------
_CONVNEXT_MODEL_NAME = "convnext_tiny.fb_in22k_ft_in1k"

def find_offline_convnext_weights():
    exact = [
        Path("/kaggle/input/models/ambrosm/convnext-tiny-fb-in22k-ft-in1k/pytorch/default/1/model.safetensors"),
        Path("/kaggle/input/convnext-tiny-fb-in22k-ft-in1k/model.safetensors"),
    ]
    for p in exact:
        if p.is_file():
            return p

    hits=[]
    for pat in (
        "/kaggle/input/models/**/convnext-tiny-fb-in22k-ft-in1k/**/model.safetensors",
        "/kaggle/input/**/convnext*tiny*in22k*ft*in1k*/**/model.safetensors",
    ):
        hits.extend(Path(p) for p in glob.glob(pat,recursive=True))
    hits=sorted({p for p in hits if p.is_file()},key=lambda p:str(p))
    if hits:
        return hits[0]

    raise FileNotFoundError(
        "Add Kaggle Model: ambrosm/convnext-tiny-fb-in22k-ft-in1k"
    )

CONVNEXT_PRETRAINED_PATH=find_offline_convnext_weights()
print("offline ConvNeXt:",CONVNEXT_PRETRAINED_PATH)

encoder=timm.create_model(
    _CONVNEXT_MODEL_NAME,
    pretrained=True,
    pretrained_cfg_overlay={"file":str(CONVNEXT_PRETRAINED_PATH)},
    num_classes=0,
    global_pool="avg",
)
FEATURE_DIM=int(encoder.num_features)
if FEATURE_DIM != 768:
    raise RuntimeError(f"Unexpected ConvNeXt feature dim: {FEATURE_DIM}")

# ----------------------- balanced pseudo-label source audit -------------------
# Stage-A uses non-gold studies only, preserving Gold58 as a clean validation gate.
adapt_indices=np.where(
    (~gold_study_mask)
    & (np.sum(np.isfinite(targets),axis=1)>=8)
    & (np.sum(weights>0,axis=1)>=8)
)[0]
if len(adapt_indices)<3000:
    raise RuntimeError(
        f"Only {len(adapt_indices)} pseudo-labelled non-gold studies are available."
    )

selected_rows={}
for ii in adapt_indices:
    uid=UIDS[int(ii)]
    frame=series_by_study.get(uid)
    if frame is not None and len(frame):
        selected_rows[uid]=choose_series(frame).to_dict("records")

# -------------------------- disk-backed adaptation set ------------------------
# Four deterministic views per study, stored once as uint8:
# 4349*4*3*224*224 ~= 2.6 GB on disk, not in host RAM.
ADAPT_REPEATS=4
ADAPT_Q=(0.18,0.38,0.62,0.82)
N_ADAPT=len(adapt_indices)*ADAPT_REPEATS
ADAPT_BIN=WORK/"dinov5_adapt_triplets_uint8.dat"
ADAPT_MAP=WORK/"dinov5_adapt_sample_to_study.npy"
ADAPT_OK=WORK/"dinov5_adapt_valid.npy"

sample_to_study=np.repeat(adapt_indices.astype(np.int32),ADAPT_REPEATS)
valid_sample=np.zeros(N_ADAPT,np.uint8)

mm=np.memmap(
    ADAPT_BIN,
    mode="w+",
    dtype=np.uint8,
    shape=(N_ADAPT,3,IMAGE_SIZE,IMAGE_SIZE),
)

t0=time.time()
fallbacks=0
for local_i,base_i in enumerate(adapt_indices):
    uid=UIDS[int(base_i)]
    rows=selected_rows.get(uid,[])
    rng=np.random.default_rng(SEED+int(base_i)*1009)

    for rep in range(ADAPT_REPEATS):
        out_i=local_i*ADAPT_REPEATS+rep
        success=False
        if rows:
            order=rng.permutation(len(rows))
            for ridx in order:
                row=rows[int(ridx)]
                sid=str(row["SeriesInstanceUID"])
                files=ordered_files(ROOT/"train_series"/uid/sid)
                if not files:
                    continue
                q=float(np.clip(
                    ADAPT_Q[rep]+rng.uniform(-0.06,0.06),
                    0.08,0.92,
                ))
                c=int(round(q*(len(files)-1)))
                try:
                    mm[out_i]=triplet_uint8(files,c)
                    valid_sample[out_i]=1
                    success=True
                    break
                except Exception:
                    continue
        if not success:
            mm[out_i].fill(0)
            fallbacks+=1

    if (local_i+1)%100==0 or local_i+1==len(adapt_indices):
        mm.flush()
        gc.collect()
        print(
            f"precache {local_i+1}/{len(adapt_indices)} "
            f"fallback_triplets={fallbacks} "
            f"elapsed={(time.time()-t0)/60:.1f}m",
            flush=True,
        )

mm.flush()
del mm
np.save(ADAPT_MAP,sample_to_study)
np.save(ADAPT_OK,valid_sample)
gc.collect()

valid_fraction=float(valid_sample.mean())
print(
    "adapt memmap ready:",
    ADAPT_BIN,
    "size_gb=",round(ADAPT_BIN.stat().st_size/(1024**3),2),
    "valid_fraction=",round(valid_fraction,4),
)
if valid_fraction < 0.97:
    raise RuntimeError(
        f"Too many adaptation triplet failures: valid_fraction={valid_fraction:.4f}"
    )

# --------------------------- memmap-only adaptation ---------------------------
class MemmapAdaptDataset(Dataset):
    def __init__(self, bin_path, sample_to_study, valid):
        self.bin_path=str(bin_path)
        self.sample_to_study=np.asarray(sample_to_study,dtype=np.int32)
        self.valid=np.asarray(valid,dtype=np.uint8)
        self.mm=np.memmap(
            self.bin_path,
            mode="r",
            dtype=np.uint8,
            shape=(len(self.sample_to_study),3,IMAGE_SIZE,IMAGE_SIZE),
        )
        self.keep=np.where(self.valid>0)[0]

    def __len__(self):
        return len(self.keep)

    def __getitem__(self,k):
        sample_i=int(self.keep[k])
        base_i=int(self.sample_to_study[sample_i])

        # Copy one fixed-size uint8 sample; no DICOM/pydicom/cv2 here.
        arr=np.array(self.mm[sample_i],copy=True)
        x=torch.from_numpy(arr)
        y=torch.from_numpy(targets[base_i].astype(np.float32))
        w=torch.from_numpy(weights[base_i].astype(np.float32))
        return x,y,w

class AdaptModel(nn.Module):
    def __init__(self,encoder):
        super().__init__()
        self.encoder=encoder
        self.head=nn.Linear(FEATURE_DIM,N_TARGETS)

    def forward(self,x):
        return self.head(self.encoder(x))

for p in encoder.parameters():
    p.requires_grad=False
if not hasattr(encoder,"stages") or len(encoder.stages)<1:
    raise RuntimeError("Unexpected timm ConvNeXt structure")
for p in encoder.stages[-1].parameters():
    p.requires_grad=True
for name in ("norm_pre","head"):
    mod=getattr(encoder,name,None)
    if mod is not None:
        for p in mod.parameters():
            p.requires_grad=True

# Deliberately use ONE GPU for stage A: no DataParallel replicas, simpler and stable.
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
adapt_model=AdaptModel(encoder).to(device)

adapt_ds=MemmapAdaptDataset(ADAPT_BIN,sample_to_study,valid_sample)
adapt_dl=DataLoader(
    adapt_ds,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
)

trainable=[p for p in adapt_model.parameters() if p.requires_grad]
opt=torch.optim.AdamW(trainable,lr=8e-5,weight_decay=1e-4)
scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda"))
GRAD_ACCUM=2

try:
    import psutil
    proc=psutil.Process(os.getpid())
    def rss_gb():
        return proc.memory_info().rss/(1024**3)
except Exception:
    def rss_gb():
        return float("nan")

def weighted_bce(logits,y,w):
    valid=torch.isfinite(y)&(w>0)
    yy=torch.nan_to_num(y,nan=.5)
    loss=F.binary_cross_entropy_with_logits(logits,yy,reduction="none")
    ww=torch.where(valid,w,torch.zeros_like(w))
    return (loss*ww).sum()/ww.sum().clamp_min(1.0)

print(
    "MRI adaptation from memmap:",
    "samples=",len(adapt_ds),
    "batch=32 grad_accum=2 effective_batch=64",
    flush=True,
)

for epoch in range(3):
    adapt_model.train()
    opt.zero_grad(set_to_none=True)
    losses=[]

    for step,(x,y,w) in enumerate(adapt_dl):
        # uint8 -> normalized float only for the current fixed-size batch.
        x=normalize_uint8(x).to(device,non_blocking=False)
        y=y.to(device,non_blocking=False)
        w=w.to(device,non_blocking=False)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type=="cuda"),
        ):
            logits=adapt_model(x)
            loss=weighted_bce(logits,y,w)
            loss_scaled=loss/GRAD_ACCUM

        scaler.scale(loss_scaled).backward()

        do_step=((step+1)%GRAD_ACCUM==0) or ((step+1)==len(adapt_dl))
        if do_step:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(trainable,1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        losses.append(float(loss.detach().cpu()))
        del x,y,w,logits,loss,loss_scaled

        if (step+1)%100==0 or step+1==len(adapt_dl):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"adapt epoch {epoch+1}/3 step {step+1}/{len(adapt_dl)} "
                f"loss={np.mean(losses[-100:]):.4f} rss={rss_gb():.2f}GB",
                flush=True,
            )

    mean_loss=float(np.mean(losses))
    core=adapt_model
    torch.save(
        {
            "epoch":epoch+1,
            "model":core.state_dict(),
            "optimizer":opt.state_dict(),
            "scaler":scaler.state_dict(),
            "mean_loss":mean_loss,
            "model_name":_CONVNEXT_MODEL_NAME,
        },
        WORK/f"dinov5_adapt_epoch{epoch+1}.pt",
    )
    print(
        f"adapt epoch {epoch+1}/3 mean_loss={mean_loss:.4f} "
        f"rss={rss_gb():.2f}GB checkpoint_saved",
        flush=True,
    )

# Export adapted backbone only.
encoder=adapt_model.encoder.cpu()
torch.save(encoder.state_dict(),WORK/"dinov5_convnext_backbone.pt")
del adapt_model,opt,scaler,adapt_dl,adapt_ds,trainable
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Temporary 2.6 GB training memmap is no longer needed.
try:
    ADAPT_BIN.unlink(missing_ok=True)
    ADAPT_MAP.unlink(missing_ok=True)
    ADAPT_OK.unlink(missing_ok=True)
except Exception as exc:
    print("temporary memmap cleanup warning:",repr(exc))

# ----------------------- stage B: cached study features -----------------------
# DICOM is decoded again only ONCE for the final study token bank, never per epoch.
encoder.eval()
for p in encoder.parameters():
    p.requires_grad=False

gpu_count=torch.cuda.device_count()
device=torch.device("cuda:0" if gpu_count else "cpu")
if gpu_count>1:
    encoder=nn.DataParallel(encoder,device_ids=list(range(gpu_count)))
encoder=encoder.to(device)

@torch.inference_mode()
def extract_study(uid):
    frame=series_by_study.get(uid)
    if frame is None or len(frame)==0:
        return (
            np.zeros((1,FEATURE_DIM),np.float16),
            np.zeros((1,6),np.float16),
        )

    frame=choose_series(frame)
    images=[]
    metas=[]

    for _,row in frame.iterrows():
        sid=str(row["SeriesInstanceUID"])
        files=ordered_files(ROOT/"train_series"/uid/sid)
        if not files:
            continue

        for q in CLIP_Q:
            center=int(round(q*(len(files)-1)))
            try:
                arr=triplet_uint8(files,center)
                images.append(torch.from_numpy(arr))
                metas.append(meta_vector(row,q))
            except Exception:
                continue

    if not images:
        return (
            np.zeros((1,FEATURE_DIM),np.float16),
            np.zeros((1,6),np.float16),
        )

    feats=[]
    for s in range(0,len(images),64):
        x=torch.stack(images[s:s+64])
        x=normalize_uint8(x).to(device,non_blocking=False)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type=="cuda"),
        ):
            z=encoder(x).float()
        feats.append(z.cpu().numpy())
        del x,z

    feat=np.concatenate(feats,axis=0).astype(np.float16)
    meta=np.stack(metas,axis=0).astype(np.float16)
    del images,metas,feats
    return feat,meta

t0=time.time()
feature_fallbacks=0

for i,uid in enumerate(UIDS):
    fp=FEATURE_CACHE/f"{uid}.npz"
    if fp.is_file():
        continue

    try:
        feat,meta=extract_study(uid)
    except Exception as exc:
        feature_fallbacks+=1
        if feature_fallbacks<=10:
            print(
                "feature fallback",
                uid,
                type(exc).__name__,
                str(exc)[:140],
            )
        feat=np.zeros((1,FEATURE_DIM),np.float16)
        meta=np.zeros((1,6),np.float16)

    np.savez_compressed(fp,feat=feat,meta=meta)
    del feat,meta

    if (i+1)%50==0 or i+1==len(UIDS):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if (i+1)%100==0 or i+1==len(UIDS):
        print(
            f"feature bank {i+1}/{len(UIDS)} "
            f"fallbacks={feature_fallbacks} "
            f"rss={rss_gb():.2f}GB "
            f"elapsed={(time.time()-t0)/60:.1f}m",
            flush=True,
        )

del encoder
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("feature extraction done; fallback studies:",feature_fallbacks)

# ======== CELL 4 [code] ========
# ----------------------- label-aware attention head ---------------------------
class FeatureDataset(Dataset):
    def __init__(self, indices, y, w):
        self.indices=list(map(int,indices))
        self.y=y
        self.w=w

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,k):
        i=self.indices[k]
        z=np.load(FEATURE_CACHE/f"{UIDS[i]}.npz")
        return (
            torch.from_numpy(z["feat"].astype(np.float32)),
            torch.from_numpy(z["meta"].astype(np.float32)),
            torch.from_numpy(self.y[i].astype(np.float32)),
            torch.from_numpy(self.w[i].astype(np.float32)),
            i,
        )

def collate(batch):
    B=len(batch)
    S=max(item[0].shape[0] for item in batch)
    feat=torch.zeros(B,S,FEATURE_DIM)
    meta=torch.zeros(B,S,6)
    mask=torch.zeros(B,S,dtype=torch.bool)
    yy=[]; ww=[]; ids=[]
    for b,(f,m,y,w,i) in enumerate(batch):
        n=f.shape[0]
        feat[b,:n]=f
        meta[b,:n]=m
        mask[b,:n]=True
        yy.append(y); ww.append(w); ids.append(i)
    return feat,meta,mask,torch.stack(yy),torch.stack(ww),torch.tensor(ids)

class LabelAwareHead(nn.Module):
    def __init__(self,d=384):
        super().__init__()
        self.proj=nn.Sequential(
            nn.Linear(FEATURE_DIM,d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(.15),
        )
        self.meta=nn.Sequential(
            nn.Linear(6,64),nn.GELU(),nn.Linear(64,d)
        )
        self.query=nn.Parameter(torch.randn(N_TARGETS,d)*0.02)
        self.key=nn.Linear(d,d,bias=False)
        self.value=nn.Linear(d,d,bias=False)
        self.label_emb=nn.Parameter(torch.randn(N_TARGETS,d)*0.02)

        layer=nn.TransformerEncoderLayer(
            d_model=d,nhead=8,dim_feedforward=768,
            dropout=.12,activation="gelu",
            batch_first=True,norm_first=True,
        )
        self.cross_label=nn.TransformerEncoder(layer,num_layers=2)
        self.norm=nn.LayerNorm(d)
        self.h1=nn.Linear(d,192)
        self.out=nn.Parameter(torch.randn(N_TARGETS,192)*0.02)
        self.bias=nn.Parameter(torch.zeros(N_TARGETS))

    def forward(self,feat,meta,mask):
        h=self.proj(feat)+self.meta(meta)
        k=self.key(h)
        v=self.value(h)
        scores=torch.einsum("ld,bsd->bls",self.query,k)/math.sqrt(k.shape[-1])
        scores=scores.masked_fill(~mask[:,None,:],-1e4)
        attn=torch.softmax(scores,dim=-1)
        z=torch.einsum("bls,bsd->bld",attn,v)
        z=z+self.label_emb[None]
        z=self.cross_label(z)
        z=F.gelu(self.h1(self.norm(z)))
        return torch.einsum("blh,lh->bl",z,self.out)+self.bias[None]

def soft_bce(logits,y,w):
    valid=torch.isfinite(y)&(w>0)
    yy=torch.nan_to_num(y,nan=.5)
    loss=F.binary_cross_entropy_with_logits(logits,yy,reduction="none")
    ww=torch.where(valid,w,torch.zeros_like(w))
    return (loss*ww).sum()/ww.sum().clamp_min(1.0)

def pair_rank_loss(logits,y,w):
    pieces=[]
    for j in range(N_TARGETS):
        valid=torch.isfinite(y[:,j])&(w[:,j]>0)
        if valid.sum()<4:
            continue
        yy=y[valid,j]
        zz=logits[valid,j]
        pos=zz[yy>=.65]
        neg=zz[yy<=.35]
        if len(pos)==0 or len(neg)==0:
            continue
        n=min(64,max(len(pos),len(neg)))
        p=pos[torch.randint(len(pos),(n,),device=zz.device)]
        q=neg[torch.randint(len(neg),(n,),device=zz.device)]
        pieces.append(F.softplus(-(p-q)).mean())
    return torch.stack(pieces).mean() if pieces else logits.sum()*0

def macro_auc(y,p):
    vals=[]
    for j in range(N_TARGETS):
        m=np.isfinite(y[:,j])
        if m.sum()<3 or len(np.unique(y[m,j]))<2:
            continue
        vals.append(roc_auc_score(y[m,j],p[m,j]))
    return float(np.mean(vals)) if vals else float("nan")

def per_target_auc(y,p):
    result={}
    for j,t in enumerate(TARGETS):
        m=np.isfinite(y[:,j])
        if m.sum()>=3 and len(np.unique(y[m,j]))>=2:
            result[t]=float(roc_auc_score(y[m,j],p[m,j]))
        else:
            result[t]=None
    return result


# ======== CELL 5 [code] ========
# --------------------------- 5-fold Gold58 OOF -------------------------------
gold_indices=np.where(gold_study_mask)[0]
if len(gold_indices)<40:
    raise RuntimeError(f"Expected roughly 58 expert studies; found {len(gold_indices)}")

# Stable low-cardinality stratifier from positive-count burden.
positive_count=np.nansum(gold[gold_indices],axis=1)
strata=np.clip(positive_count.astype(int),0,5)
vc=pd.Series(strata).value_counts()
for value in list(vc[vc<5].index):
    replacement=max(0,min(5,int(value)-1))
    strata[strata==value]=replacement

skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
fold_of=np.full(len(UIDS),-1,np.int8)
for fold,(_,va_local) in enumerate(skf.split(gold_indices,strata)):
    fold_of[gold_indices[va_local]]=fold

print(pd.Series(fold_of[gold_indices]).value_counts().sort_index())

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
non_gold=np.where(
    (~gold_study_mask) & np.any(np.isfinite(targets),axis=1)
)[0]
oof=np.full((len(UIDS),N_TARGETS),np.nan,np.float32)
fold_metrics={}

for fold in range(5):
    val_idx=np.where(fold_of==fold)[0]
    train_gold=np.where(gold_study_mask & (fold_of!=fold))[0]
    train_idx=np.concatenate([non_gold,train_gold])
    rng=np.random.default_rng(SEED+fold)
    rng.shuffle(train_idx)

    train_ds=FeatureDataset(train_idx,targets,weights)
    val_weights=np.where(np.isfinite(gold),1.0,0.0).astype(np.float32)
    val_ds=FeatureDataset(val_idx,gold,val_weights)

    train_dl=DataLoader(
        train_ds,batch_size=40,shuffle=True,num_workers=0,
        pin_memory=False,collate_fn=collate,
    )
    val_dl=DataLoader(
        val_ds,batch_size=32,shuffle=False,num_workers=0,
        pin_memory=False,collate_fn=collate,
    )

    model=LabelAwareHead().to(device)
    optimizer=torch.optim.AdamW(
        model.parameters(),lr=2e-4,weight_decay=2e-4
    )
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,T_max=16,eta_min=2e-5
    )

    best_score=-1.0
    best_state=None
    no_improve=0

    for epoch in range(16):
        model.train()
        train_loss=[]
        for feat,meta,mask,y,w,_ in train_dl:
            feat=feat.to(device,non_blocking=True)
            meta=meta.to(device,non_blocking=True)
            mask=mask.to(device,non_blocking=True)
            y=y.to(device,non_blocking=True)
            w=w.to(device,non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits=model(feat,meta,mask)
            loss=.88*soft_bce(logits,y,w)+.12*pair_rank_loss(logits,y,w)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            train_loss.append(float(loss.detach().cpu()))

        scheduler.step()

        model.eval()
        pred=[]
        truth=[]
        with torch.inference_mode():
            for feat,meta,mask,y,w,_ in val_dl:
                logits=model(
                    feat.to(device),
                    meta.to(device),
                    mask.to(device),
                )
                pred.append(torch.sigmoid(logits).cpu().numpy())
                truth.append(y.numpy())

        pred=np.concatenate(pred)
        truth=np.concatenate(truth)
        score=macro_auc(truth,pred)
        print(
            f"fold={fold} epoch={epoch+1:02d} "
            f"loss={np.mean(train_loss):.4f} GoldAUC={score:.4f}"
        )

        if np.isfinite(score) and score>best_score+1e-5:
            best_score=score
            no_improve=0
            best_state={
                k:v.detach().cpu().clone()
                for k,v in model.state_dict().items()
            }
        else:
            no_improve+=1

        if epoch>=7 and no_improve>=5:
            break

    if best_state is None:
        raise RuntimeError(f"fold {fold}: no valid model checkpoint")

    model.load_state_dict(best_state)
    model.eval()

    pred=[]
    ids=[]
    with torch.inference_mode():
        for feat,meta,mask,y,w,ii in val_dl:
            logits=model(
                feat.to(device),
                meta.to(device),
                mask.to(device),
            )
            pred.append(torch.sigmoid(logits).cpu().numpy())
            ids.append(ii.numpy())

    pred=np.concatenate(pred)
    ids=np.concatenate(ids)
    oof[ids]=pred

    fold_metrics[str(fold)]={
        "macro_auc":float(best_score),
        "n_gold":int(len(val_idx)),
    }

    torch.save(
        {"state":best_state,"fold":fold,"targets":TARGETS},
        WORK/f"dinov5_convnext_head_fold{fold}.pt",
    )

    del model,best_state,train_dl,val_dl,train_ds,val_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

gold_pred=oof[gold_indices]
gold_true=gold[gold_indices]
oof_macro=macro_auc(gold_true,gold_pred)
target_auc=per_target_auc(gold_true,gold_pred)

print("Gold58 OOF macro:",oof_macro)
print(json.dumps(target_auc,indent=2))


# ======== CELL 6 [code] ========
# ----------------------- manifest + blend recommendation ----------------------
# Novelty is not enough; targets with weak Gold58 OOF get zero ensemble weight.
blend_beta={}
for t in TARGETS:
    auc=target_auc.get(t)
    if auc is None or not np.isfinite(auc) or auc<0.66:
        beta=0.0
    else:
        beta=float(np.clip(0.04+0.20*(auc-0.66),0.04,0.12))
    blend_beta[t]=beta

manifest={
    "schema":"dinov5_convnext_diversity_v1",
    "title":"RSNA Knee | DINOsaur V5? 🦖",
    "seed":SEED,
    "targets":TARGETS,
    "backbone":"timm convnext_tiny.fb_in22k_ft_in1k -> non-gold soft-label MRI adaptation",
    "backbone_model_name":_CONVNEXT_MODEL_NAME,
    "offline_pretrained_source":str(CONVNEXT_PRETRAINED_PATH),
    "input":{
        "crop_mm":CROP_MM,
        "image_size":IMAGE_SIZE,
        "max_series":MAX_SERIES,
        "clip_quantiles":list(CLIP_Q),
        "order":"physical ImagePositionPatient projection",
    },
    "labels":{
        "sources":source_paths,
        "consensus":"two-level median: within Kaggle dataset family, then across independent families",
        "gold_override":True,
        "pseudo_temperature":0.85,
    },
    "validation":{
        "gold_n":int(len(gold_indices)),
        "oof_macro_auc":float(oof_macro),
        "per_target_auc":target_auc,
        "fold_metrics":fold_metrics,
    },
    "recommended_main_beta":blend_beta,
    "weights":{
        "backbone":"dinov5_convnext_backbone.pt",
        "heads":[f"dinov5_convnext_head_fold{i}.pt" for i in range(5)],
    },
}

(WORK/"dinov5_convnext_manifest.json").write_text(
    json.dumps(manifest,indent=2,sort_keys=True)+"\n",
    encoding="utf-8",
)

oof_df=pd.DataFrame({"StudyInstanceUID":np.array(UIDS)[gold_indices]})
for j,t in enumerate(TARGETS):
    oof_df[t]=gold_pred[:,j]
oof_df.to_csv(WORK/"dinov5_convnext_gold58_oof.csv",index=False)

print(json.dumps(
    {"oof_macro_auc":oof_macro,"recommended_main_beta":blend_beta},
    indent=2,
))
print("\nSave this notebook version and attach its OUTPUT to the final V5 inference notebook.")

# Feature cache is a training intermediate and is intentionally not exported.
import shutil as _d5_shutil
_d5_shutil.rmtree(FEATURE_CACHE, ignore_errors=True)
print("Removed training-only feature cache:", FEATURE_CACHE)


# ======== CELL 7 [markdown] ========
## Use in the final submission

After this notebook finishes, **Save Version**. Then in the submission V5 use:

**Add Input → Notebook Output → this trainer**

The final V5 searches for `dinov5_convnext_manifest.json`, the frozen backbone,
and the five fold heads automatically.

The original 0.942 submission remains available as an exact anchor. If this
new arm has weak Gold58 OOF on a target, that target receives zero blend weight.

