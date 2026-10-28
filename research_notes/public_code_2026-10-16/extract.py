"""Static scan of downloaded notebooks (never executed): code text -> features. Usage: python -I extract.py <dir>... > out.csv"""
import sys, os, re, json, glob, hashlib, csv
def code_of(path):
    if path.endswith('.ipynb'):
        nb = json.load(open(path, encoding='utf-8', errors='replace'))
        cells = nb.get('cells') or nb.get('worksheets', [{}])[0].get('cells', [])
        return '\n'.join(''.join(c.get('source', '')) if isinstance(c.get('source'), list) else c.get('source', '')
                         for c in cells if c.get('cell_type') == 'code')
    return open(path, encoding='utf-8', errors='replace').read()
def norm(s):
    s = re.sub(r'#.*', '', s); s = re.sub(r'\s+', '', s); return s
pat = {
 'train': r'\.backward\(|optimizer\.step|opt\.step|scaler\.step',
 'backbones': r"(coatnet[\w\.]*|convnext[\w\.]*|efficientnet[\w\.]*|tf_efficientnet[\w\.]*|resnet\d+[\w\.]*|dinov[23][\w\.]*|vit_\w+|swin\w*|regnet\w*|maxvit\w*|eva\w*|radimagenet|nfnet\w*|seresnext\w*|densenet\d*)",
 'img': r"(?:IMG|img_size|IMG_SIZE|image_size|RES|res|SIZE|size)\s*[=:]\s*\(?(\d{3})",
 'epochs': r"(?:EPOCHS|epochs|n_epochs|NUM_EPOCHS|num_epochs)\s*[=:]\s*(\d+)",
 'lr': r"(?:\bLR\b|\blr\b|learning_rate|BACKBONE_LR|HEAD_LR|backbone_lr|head_lr)\s*[=:]\s*([\d\.e\-]+)",
 'slices': r"(?:N_SLICES|n_slices|NSLICE|N_SLICE|num_slices|SLICES|slices_per|K_WIN|n_windows|WINDOWS)\s*[=:]\s*(\d+)",
 'crop_mm': r"(?:CROP_MM|crop_mm|FOV_MM|fov_mm|CROP)\s*[=:]\s*([\d\.]+)",
}
w = csv.writer(sys.stdout)
w.writerow(['dir', 'file', 'nlines', 'hash', 'train', 'backbones', 'img', 'epochs', 'lr', 'slices', 'crop_mm'])
for d in sys.argv[1:]:
    for p in sorted(glob.glob(os.path.join(d, '*.ipynb')) + glob.glob(os.path.join(d, '*.py'))):
        try: c = code_of(p)
        except Exception as e: w.writerow([d, os.path.basename(p), 'ERR', str(e)[:50]]); continue
        f = lambda k: sorted(set(m.lower() if isinstance(m, str) else m for m in re.findall(pat[k], c)))
        w.writerow([os.path.basename(d), os.path.basename(p), c.count('\n'), hashlib.md5(norm(c).encode()).hexdigest()[:10],
                    bool(re.search(pat['train'], c)), ' '.join(f('backbones'))[:200], ' '.join(f('img'))[:60],
                    ' '.join(f('epochs'))[:40], ' '.join(f('lr'))[:60], ' '.join(f('slices'))[:40], ' '.join(f('crop_mm'))[:30]])
