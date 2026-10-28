"""Extract string-literal source dicts (SOURCES = {...}, CODE = {...}) from a notebook with ast.literal_eval only (no execution)."""
import sys, json, ast, os
nb = json.load(open(sys.argv[1], encoding='utf-8'))
out = sys.argv[2]; os.makedirs(out, exist_ok=True)
for c in nb['cells']:
    if c['cell_type'] != 'code': continue
    src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
    try: tree = ast.parse(src)
    except SyntaxError: continue
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            try: d = ast.literal_eval(node.value)
            except Exception: continue
            if all(isinstance(k, str) and isinstance(v, str) and len(v) > 500 for k, v in d.items()) and d:
                name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else 'x'
                for k, v in d.items():
                    fn = os.path.join(out, f'{name}__{os.path.basename(k)}')
                    if not os.path.exists(fn): open(fn, 'w', encoding='utf-8').write(v)
                    print(name, k, len(v.splitlines()), 'lines')
