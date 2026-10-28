"""Dataset cards (metadata incl. description + licence) of the datasets used by notebooks scoring >= MINSCORE (read-only).
python fetch_ds_meta.py kernels_web.json ds_meta 0.950   -> ds_meta/<owner>__<slug>.json"""
import sys, json, os, subprocess, time, collections, tempfile, shutil
raw = json.load(open(sys.argv[1])); out = sys.argv[2]; minscore = float(sys.argv[3])
os.makedirs(out, exist_ok=True)
refs = collections.Counter()
for k in raw:
    if (k.get('bestPublicScore') or 0) >= minscore or k['scriptUrl'].endswith(('toghrulaliyev/rsna-knee-aze-team-dd6d24',)):
        for s in k.get('dataSources') or []:
            u = s.get('dataSourceUrl') or ''
            if u.startswith('/datasets/'): refs[u[len('/datasets/'):]] += 1
for extra in sys.argv[4:]: refs[extra] += 0
print(len(refs), 'datasets', flush=True)
for ref in sorted(refs, key=lambda r: -refs[r]):
    dst = os.path.join(out, ref.replace('/', '__') + '.json')
    if os.path.exists(dst): continue
    for attempt in range(8):
        tmp = tempfile.mkdtemp()
        p = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'kaggle', 'datasets', 'metadata', ref, '-p', tmp], capture_output=True, text=True)
        f = os.path.join(tmp, 'dataset-metadata.json')
        if os.path.exists(f):
            shutil.copy(f, dst); shutil.rmtree(tmp); print('ok', ref, flush=True); break
        shutil.rmtree(tmp)
        msg = (p.stderr or p.stdout).strip()[-120:]
        print('retry', ref, attempt, msg, flush=True)
        if '404' in msg or '403' in msg:
            open(dst, 'w').write(json.dumps({'error': msg})); break
        time.sleep(15 * (attempt + 1))
    time.sleep(3)
