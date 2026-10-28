"""Kaggle run log (JSON list of {stream_name,time,data}) -> plain text with timestamps; drops debugger noise. python -I logtxt.py <log>"""
import sys, json, re
sys.stdout.reconfigure(encoding='utf-8')
raw = open(sys.argv[1], encoding='utf-8', errors='replace').read()
try: ev = json.loads(raw)
except Exception:
    ev = [json.loads(l.strip().lstrip(',')) for l in raw.splitlines() if l.strip().lstrip(',').startswith('{')]
for e in ev:
    d = e.get('data', '')
    if re.search(r'Debugger warning|frozen modules|PYDEVD|Xfrozen|NbConvertApp|SyntaxWarning|^\s*$', d): continue
    t = e.get('time', 0)
    for line in d.rstrip('\n').split('\n'):
        print(f'{t:8.0f} {line[:400]}')
