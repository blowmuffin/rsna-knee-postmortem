"""Static keyword scan of pulled notebooks (never executed) for external-data mentions. python -I scan_ext.py kernels_web.json > scan_ext.csv"""
import json, os, re, sys, csv, hashlib
raw = json.load(open(sys.argv[1]))
base = '../../ref/'
dirs = {d for d in os.listdir(base)}
alt = {}
for p in ('_new_2026-10-17', '_new_2026-10-17b', '_new_2026-10-17c'):
    for d in os.listdir(base + p): alt[d] = base + p + '/' + d
KW = {'oai': r'\bOAI\b|osteoarthritis initiative|moaks', 'fastmri': r'fast[-_ ]?mri|fm[_-]aux|fastmri', 'skmtea': r'skm[-_ ]?tea', 'mrnet': r'mrnet', 'kneemri': r'kneemri|knee[_ ]mri[_ ]dataset|ISMRM', 'oai_dataset': r'nda\.nih|10\.15154',
      'internet': r'pip install|!pip|huggingface|from_pretrained\(|hf_hub', 'qwen_llm': r'qwen|llama|gemma|vllm'}
def code_of(path):
    if path.endswith('.ipynb'):
        nb = json.load(open(path, encoding='utf-8', errors='replace'))
        cells = nb.get('cells') or []
        return '\n'.join(''.join(c.get('source', '')) if isinstance(c.get('source'), list) else c.get('source', '') for c in cells if c.get('cell_type') == 'code'), \
               '\n'.join(''.join(c.get('source', '')) if isinstance(c.get('source'), list) else c.get('source', '') for c in cells if c.get('cell_type') == 'markdown')
    t = open(path, encoding='utf-8', errors='replace').read(); return t, ''
w = csv.writer(sys.stdout); w.writerow(['ref', 'score', 'nlines', 'hash'] + list(KW) + ['md_oai', 'md_fastmri'])
for k in raw:
    sc = k.get('bestPublicScore') or 0
    if sc < 0.94: continue
    ref = k['scriptUrl'].removeprefix('/code/'); a = ref.replace('/', '__'); b = ref.replace('/', '_')
    d = alt.get(b) or (base + a if a in dirs else None)
    if not d: continue
    files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(('.ipynb', '.py'))]
    if not files: continue
    c, md = code_of(files[0])
    norm = re.sub(r'\s+', '', re.sub(r'#.*', '', c))
    w.writerow([ref, sc, c.count('\n'), hashlib.md5(norm.encode()).hexdigest()[:10]] + [len(re.findall(p, c, flags=re.I)) for p in KW.values()] +
               [len(re.findall(KW['oai'], md, flags=re.I)), len(re.findall(KW['fastmri'], md, flags=re.I))])
