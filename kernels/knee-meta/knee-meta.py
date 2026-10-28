# One DICOM header per study -> patient / scanner metadata for train and test (CPU, ~minutes).
import os, glob, pandas as pd, pydicom
from multiprocessing import Pool
comp = glob.glob('/kaggle/input/**/rsna-knee-abnormality-detection', recursive=True)
comp = comp[0] if comp else '/kaggle/input/rsna-knee-abnormality-detection'
TAGS = ['PatientSex', 'PatientAge', 'PatientSize', 'PatientWeight', 'PatientBirthDate', 'StudyDate',
        'Manufacturer', 'ManufacturerModelName', 'MagneticFieldStrength', 'InstitutionName', 'BodyPartExamined']

def one(args):
    split, study = args
    row = {'StudyInstanceUID': study, 'split': split}
    try:
        sd = os.path.join(comp, f'{split}_series', study)
        ser = sorted(os.listdir(sd))[0]
        f = sorted(os.listdir(os.path.join(sd, ser)))[0]
        h = pydicom.dcmread(os.path.join(sd, ser, f), stop_before_pixels=True)
        for t in TAGS:
            v = getattr(h, t, None)
            row[t] = None if v is None else str(v)
    except Exception as e:
        row['error'] = repr(e)
    return row

jobs = []
for split in ('train', 'test'):
    ids = pd.read_csv(f'{comp}/{split}.csv', dtype=str).StudyInstanceUID
    jobs += [(split, s) for s in ids]
with Pool(os.cpu_count()) as p:
    rows = p.map(one, jobs, chunksize=16)
df = pd.DataFrame(rows)
df.to_csv('/kaggle/working/meta.csv', index=False)
print(df.notna().mean().round(3).to_string())
for t in TAGS:
    if t in df:
        print(t, df[t].value_counts(dropna=False).head(8).to_dict())
