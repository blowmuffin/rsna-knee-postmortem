"""Near-duplicate detection among pulled notebooks (static, never executed). python -I copies.py kernels_web.json > copies.csv
copy_of = notebook whose CURRENT VERSION is the earliest-dated one with line-set Jaccard >= 0.8 to this one (ordering by scriptVersionDateCreated, not notebook creation); sim = that Jaccard."""
import json, os, re, sys, csv
raw = json.load(open(sys.argv[1]))
base = '../../ref/'
dirs = set(os.listdir(base)); alt = {}
for p in ('_new_2026-10-17', '_new_2026-10-17b', '_new_2026-10-17c'):
    for d in os.listdir(base + p): alt[d] = base + p + '/' + d
def lines_of(path):
    nb = json.load(open(path, encoding='utf-8', errors='replace'))
    src = []
    for c in nb.get('cells', []):
        if c.get('cell_type') == 'code':
            s = c.get('source', ''); src.append(''.join(s) if isinstance(s, list) else s)
    out = set()
    for l in '\n'.join(src).split('\n'):
        l = re.sub(r'#.*', '', l).strip()
        l = re.sub(r'\s+', ' ', l)
        if len(l) > 3: out.add(l)
    return out
items = []
for k in raw:
    sc = k.get('bestPublicScore') or 0
    if sc < 0.94: continue
    ref = k['scriptUrl'].removeprefix('/code/'); a = ref.replace('/', '__'); b = ref.replace('/', '_')
    d = alt.get(b) or (base + a if a in dirs else None)
    if not d: continue
    fs = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.ipynb')]
    if not fs: continue
    try: items.append((ref, str(k.get('scriptVersionDateCreated'))[:19], sc, lines_of(fs[0])))
    except Exception as e: print('ERR', ref, e, file=sys.stderr)
items.sort(key=lambda x: x[1])
par = list(range(len(items)))
def find(i):
    while par[i] != i:
        par[i] = par[par[i]]; i = par[i]
    return i
best = []
for i, (ref, cr, sc, ls) in enumerate(items):
    b, bs = '', 0.0
    for j in range(i):
        ls2 = items[j][3]
        if not ls or not ls2: continue
        jac = len(ls & ls2) / len(ls | ls2)
        if jac > bs: b, bs = items[j][0], jac
        if jac >= 0.9: par[find(i)] = find(j) if find(j) < find(i) else find(i); par[find(j)] = min(find(i), find(j))
    best.append((b, bs))
root = {}
for i in range(len(items)):
    r = find(i); root.setdefault(r, i)
size = {}
for i in range(len(items)): size[find(i)] = size.get(find(i), 0) + 1
w = csv.writer(sys.stdout); w.writerow(['ref', 'version_utc', 'score', 'nlineset', 'copy_of', 'sim', 'cluster_root', 'cluster_size'])
for i, (ref, cr, sc, ls) in enumerate(items):
    b, bs = best[i]
    r = min(k for k in range(len(items)) if find(k) == find(i))
    w.writerow([ref, cr, sc, len(ls), b if bs >= 0.8 else '', round(bs, 3), items[r][0], size[find(i)]])
