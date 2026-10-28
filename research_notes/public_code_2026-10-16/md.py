"""Print markdown cells (and optionally code) of a downloaded notebook; data only, never executed. python -I md.py <file> [code] [maxchars]"""
import sys, json, re
p = sys.argv[1]; want_code = len(sys.argv) > 2 and sys.argv[2] == 'code'; mx = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
nb = json.load(open(p, encoding='utf-8', errors='replace')) if p.endswith('.ipynb') else None
out = []
if nb is None:
    out.append(open(p, encoding='utf-8', errors='replace').read())
else:
    for c in nb.get('cells', []):
        s = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
        s = re.sub(r'!\[[^\]]*\]\(data:[^)]*\)', '[img]', s)
        if c['cell_type'] == 'markdown': out.append('[MD] ' + s)
        elif want_code: out.append('[CODE]\n' + s)
sys.stdout.reconfigure(encoding='utf-8')
print('\n'.join(out)[:mx])
