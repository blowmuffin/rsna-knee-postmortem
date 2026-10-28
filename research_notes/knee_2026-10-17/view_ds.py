import json, glob, sys, os
SEEN = 'seen_ds.txt'
seen = set(open(SEEN).read().split()) if os.path.exists(SEEN) else set()
n = int(sys.argv[1]) if len(sys.argv) > 1 else 1800
for f in sorted(glob.glob('ds_meta/*.json')):
    if f in seen: continue
    m = json.load(open(f, encoding='utf8')).get('info', {})
    print('=====', os.path.basename(f), '| title:', m.get('title'), '| lic:', [l.get('name') for l in m.get('licenses', [])], '| dl', m.get('totalDownloads'), '| votes', m.get('totalVotes'))
    print((m.get('description') or '(no description)')[:n].replace('\n\n', '\n'))
    seen.add(f)
open(SEEN, 'w').write('\n'.join(seen))
