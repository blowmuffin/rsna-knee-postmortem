"""Markdown sentences/table rows that state a score or delta, per notebook (data only). python -I claims.py <dirs...>"""
import sys, json, glob, os, re
sys.stdout.reconfigure(encoding='utf-8')
num = re.compile(r'(0\.9\d{1,4}|0\.8\d{2,4}|[+\u2212-]0\.0\d{1,4})')
seen = set()
for d in sys.argv[1:]:
    for p in glob.glob(os.path.join(d, '*.ipynb')):
        try: nb = json.load(open(p, encoding='utf-8', errors='replace'))
        except Exception: continue
        md = '\n'.join(''.join(c['source']) if isinstance(c['source'], list) else c['source'] for c in nb.get('cells', []) if c['cell_type'] == 'markdown')
        md = re.sub(r'<[^>]+>', ' ', md); md = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', md)
        lines = [l.strip() for l in re.split(r'\n|(?<=\.)\s', md) if num.search(l) and len(l.strip()) > 15]
        lines = [l for l in lines if l not in seen]; seen.update(lines)
        if lines:
            print('=====', os.path.basename(d))
            for l in lines[:40]: print('  ', l[:300])
