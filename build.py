"""Write kernels/<name>/ (code + kernel-metadata.json) from knee.py, then push with
   python -m kaggle kernels push -p kernels/<name>

usage: python build.py prep [name=knee-prep2a] [nslice=31] [size=256] [cropmm=150] [shard=0:2] [prep=2] [smoke=N]
       python build.py train [name=knee-train-x] [backbone=timm_name|radimagenet_resnet50] [epochs=N]
                             [folds=0,1,2] [smoke=N] [pseudo=a:kernelA,kernelB] [cache=knee-prep2a,knee-prep2b] [labels=rubric|consensus]
                             [headlr=1e-3] [posw=1] [slotdrop=0.15] [img=224] [rot=25] [aux=0.3] [mirror=0.5]
       python build.py infer [blend=w] [trains=knee-train,knee-train-coat] [nslice=16]
       python build.py blend blend=w trains=... base=subdir [ours_min=60]   (our infer first, the public notebook, then a
                             rank blend with base weight w; the base needs one NB_T0 = time.time() clock start)
       python build.py base name=knee-base957 base=goodpjw-v21   (a public notebook in baseline/<base>/, unchanged)
       python build.py union name=knee-union958lav base=helios958 base2=lavin958 [blend=0.5] [union_h=8.6]
                             (base unchanged, then base2 unchanged in its own kernel until union_h hours after the base's
                             NB_T0, then a rank blend with base weight blend; any failure keeps the base's submission.csv)"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = 'rsna-knee-abnormality-detection'
mode = sys.argv[1]
opt = dict(a.split('=', 1) for a in sys.argv[2:])
KNOWN = {'allow_smoke', 'aux', 'backbone', 'base', 'base2', 'blend', 'cache', 'cpu', 'cropmm', 'epochs', 'first', 'folds', 'headlr',
         'img', 'labels', 'lav_h', 'mirror', 'name', 'nb_t0', 'nslice', 'ours_min', 'posw', 'prep', 'pseudo', 'rot', 'shard', 'size', 'slotdrop',
         'smoke', 'trains', 'union_h'}
assert not set(opt) - KNOWN, f'build: unknown options {sorted(set(opt) - KNOWN)} (a typo would be silently ignored)'
name = opt.get('name', f'knee-{mode}') + ('-smoke' if int(opt.get('smoke', 0)) else '')  # smoke never shadows a real kernel
trains = [f'ayushkasare/{k}' for k in opt.get('trains', 'knee-train').split(',')]
src = open(os.path.join(HERE, 'knee.py'), encoding='utf-8').read()


def put(pattern, line):
    """Replace one top-level assignment line in knee.py; fail loudly if it isn't there."""
    global src
    src, n = re.subn(pattern, line, src, count=1, flags=re.M)
    assert n == 1, f'build: pattern not found in knee.py: {pattern}'


put(r"^MODE = .*$", f"MODE = '{'infer' if mode == 'blend' else mode}'")
put(r"^SMOKE = \d+", f"SMOKE = {int(opt.get('smoke', 0))}")
if mode == 'blend':
    assert 0 < float(opt.get('blend', 0)) < 1, 'blend needs blend=w with the public-stack weight in (0,1)'
if mode in ('infer', 'blend'):  # an unfinished/missing train kernel would be dropped silently by the CLI
    import subprocess
    for k in trains:
        st = subprocess.run([sys.executable, '-m', 'kaggle', 'kernels', 'status', k], capture_output=True, text=True).stdout
        assert 'COMPLETE' in st, f'{k} is not a completed kernel: {st.strip()}'
if 'blend' in opt and mode == 'infer':  # blend mode mixes in its own last cell; our part keeps BLEND 0
    put(r"^BLEND = [\d.]+", f"BLEND = {float(opt['blend'])}")
if 'backbone' in opt:
    put(r"^BACKBONE, IMG = '[^']*'", f"BACKBONE, IMG = '{opt['backbone']}'")
if 'epochs' in opt or 'folds' in opt:
    folds = [int(f) for f in opt['folds'].split(',')] if 'folds' in opt else []
    put(r"^EPOCHS, FOLD_LIST = [^#\n]*", f"EPOCHS, FOLD_LIST = {int(opt.get('epochs', 10))}, {folds}  ")
if 'allow_smoke' in opt:
    put(r"^ALLOW_SMOKE = \d", f"ALLOW_SMOKE = {int(opt['allow_smoke'])}")
if 'cache' in opt:
    put(r"^CACHE = '[^']*'", f"CACHE = '{opt['cache']}'")
if 'size' in opt:  # prep: cached in-plane pixels over CROP_MM (default 256)
    put(r"^(N_SLICE, SIZE, CROP_MM = \d+, )\d+", rf"\g<1>{int(opt['size'])}")
if 'img' in opt:  # train: network input pixels (default 224)
    put(r"^(BACKBONE, IMG = '[^']*', )\d+", rf"\g<1>{int(opt['img'])}")
if 'nslice' in opt:  # prep: slices cached per series; infer/blend: test prep depth (16 for old-cache-only models)
    put(r"^N_SLICE, SIZE, CROP_MM = \d+", f"N_SLICE, SIZE, CROP_MM = {int(opt['nslice'])}")
if 'cropmm' in opt:  # prep: field of view cropped around the image centre, mm (default 150; ~110 = knee-centred zoom)
    put(r"^(N_SLICE, SIZE, CROP_MM = \d+, \d+, )[\d.]+", rf"\g<1>{float(opt['cropmm'])}")
if 'prep' in opt:  # prep geometry version for a prep kernel (1 = original, 2 = band/crop/fallback fixes)
    put(r"^PREP = \d", f"PREP = {int(opt['prep'])}")
if 'shard' in opt:  # prep: shard=i:k caches studies[i::k]
    i, k = opt['shard'].split(':')
    put(r"^SHARD = \([^)]*\)", f"SHARD = ({int(i)}, {int(k)})")
kernel_sources = {'prep': [], 'train': [f'ayushkasare/{c}' for c in opt.get('cache', 'knee-prep').split(',')],
                  'infer': trains}.get(mode, [])
if 'labels' in opt:  # train: label source; 'rubric' or 'consensus' mounts our private label dataset
    put(r"^LABEL_SRC = '[^']*'", f"LABEL_SRC = '{opt['labels']}'")
if 'headlr' in opt:  # train: separate LR for the head (Raptor recipe 1e-3)
    put(r"^HEAD_LR = [\d.e-]+", f"HEAD_LR = {float(opt['headlr'])}")
if 'posw' in opt:  # train: 1 -> prevalence-based BCE pos_weight
    put(r"^POS_W = \d", f"POS_W = {int(opt['posw'])}")
if 'rot' in opt:  # train: rotation augmentation range in degrees (default 10)
    put(r"^ROT_DEG = [\d.]+", f"ROT_DEG = {float(opt['rot'])}")
if 'aux' in opt:  # train: weight of the auxiliary max-over-tokens MIL loss (default 0)
    put(r"^AUX_W = [\d.]+", f"AUX_W = {float(opt['aux'])}")
if 'slotdrop' in opt:  # train: whole-series dropout p (0 = off, as in runs before 2026-10-03)
    put(r"^SLOT_DROP = [\d.]+", f"SLOT_DROP = {float(opt['slotdrop'])}")
if 'mirror' in opt:  # train: p of the contralateral-knee mirror (medial<->lateral labels swapped, MCL unlabelled) + flip TTA
    put(r"^MIRROR = [\d.]+", f"MIRROR = {float(opt['mirror'])}")
if 'pseudo' in opt:
    a, ks = opt['pseudo'].split(':')
    put(r"^PSEUDO_A, PSEUDO_FROM = [^#\n]*", f"PSEUDO_A, PSEUDO_FROM = {float(a)}, {ks.split(',')}  ")
    kernel_sources += [f'ayushkasare/{k}' for k in ks.split(',')]

out = os.path.join(HERE, 'kernels', name)
os.makedirs(out, exist_ok=True)
if mode in ('blend', 'base', 'union'):
    bdir = os.path.join(HERE, 'baseline', opt.get('base', ''))  # default: the jiweiliu stack at baseline/ itself
    meta = json.load(open(os.path.join(bdir, 'kernel-metadata.json'), encoding='utf-8'))
    nb = json.load(open(os.path.join(bdir, meta['code_file']), encoding='utf-8'))
    cell = lambda s: {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': s.splitlines(keepends=True)}
    if mode == 'union':  # the base FIRST and unchanged (its NB_T0 and deadlines untouched), the second notebook after it
        b2dir = os.path.join(HERE, 'baseline', opt['base2'])
        meta2 = json.load(open(os.path.join(b2dir, 'kernel-metadata.json'), encoding='utf-8'))
        nb2 = json.load(open(os.path.join(b2dir, meta2['code_file']), encoding='utf-8'))
        assert meta2.get('docker_image') == meta.get('docker_image'), 'union: the notebooks pin different docker images'
        assert 0 < float(opt.get('blend', 0.5)) < 1, 'union: blend is the base weight in (0,1)'
        n_t0 = sum(len(re.findall(r'^NB_T0 = time\.time\(\)', ''.join(c['source']), flags=re.M)) for c in nb['cells'])
        assert n_t0 == 1, f'union: the base needs exactly one "NB_T0 = time.time()" clock start, found {n_t0}'
        snap = '''# [union] snapshot the docker's own timm and pydicom before the base pip-installs its pins over them (timm 1.0.29 ->
# 1.0.25, pydicom 3.0.2 -> 3.0.1); the second notebook gets them back through PYTHONPATH, as it was scored with them
import os, glob, shutil, importlib.util
os.makedirs('/tmp/union/site', exist_ok=True)
for _m in ('timm', 'pydicom'):
    try:
        _d = importlib.util.find_spec(_m).submodule_search_locations[0]  # a top-level find_spec does not import it
        shutil.copytree(_d, f'/tmp/union/site/{_m}', dirs_exist_ok=True)
        for _di in glob.glob(os.path.join(os.path.dirname(_d), f'{_m}-*.dist-info')):
            shutil.copytree(_di, os.path.join('/tmp/union/site', os.path.basename(_di)), dirs_exist_ok=True)
        print('UNION_SNAPSHOT', _m, _d, sorted(os.listdir('/tmp/union/site')), flush=True)
    except Exception as e:
        print('UNION_SNAPSHOT_FAILED', _m, repr(e), flush=True)
'''
        save = '''# [union] keep the base's submission outside /kaggle/working: the second notebook deletes submission.csv at start
import os, shutil, time
U_DIR, U_T0, U_BASE_OK = '/tmp/union', globals().get('NB_T0', time.time()), False
os.makedirs(U_DIR, exist_ok=True)
try:
    shutil.copyfile('/kaggle/working/submission.csv', U_DIR + '/base.csv')
    shutil.copyfile(U_DIR + '/base.csv', '/kaggle/working/union_base.csv')  # diagnostics: compare with the base run alone
    U_BASE_OK = True
    for _p in ('/kaggle/temp/probe', '/tmp/probe'):  # the base's scratch (bundle copy, predictions), done with
        shutil.rmtree(_p, ignore_errors=True)
    print(f'UNION_BASE_SAVED at {time.time() - U_T0:.0f}s', flush=True)
except Exception as e:
    print('UNION_BASE_SAVE_FAILED, the base runs alone:', repr(e), flush=True)
'''
        run = f'''# [union] the second public notebook (unchanged except its TIME_BUDGET) in its own kernel; on timeout its process tree is killed so
# nothing can rewrite submission.csv later; whatever happens, submission.csv is the base's again before the mix
import os, sys, time, signal, shutil, subprocess
U_LAV_OK = False
def _u_kill(p):
    try:
        import psutil
        kids = psutil.Process(p.pid).children(recursive=True)  # the kernel and its workers may sit in new sessions
        for k in kids:
            k.kill()
    except Exception as e:  # never pkill by name: the notebook's own kernel is an ipykernel too
        print('UNION_KILL psutil:', repr(e), flush=True)
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except Exception:
        pass
    p.wait()
if U_BASE_OK:
    try:
        budget = {float(opt.get('union_h', 8.6))} * 3600 - (time.time() - U_T0)
        assert budget > 1800, f'only {{budget:.0f}}s left for the second notebook'
        env = dict(os.environ, UNION_BUDGET_S=str(budget - 600),  # its own deadline, inside ours
                   PYTHONPATH=os.pathsep.join(x for x in ('/tmp/union/site', os.environ.get('PYTHONPATH', '')) if x))
        with open(U_DIR + '/lav_nbconvert.log', 'w') as log:
            p = subprocess.Popen([sys.executable, '-m', 'nbconvert', '--to', 'notebook', '--execute',
                                  '--ExecutePreprocessor.timeout=-1', '--ExecutePreprocessor.kernel_name=python3',
                                  '--output-dir', U_DIR, '--output', 'lav_out', U_DIR + '/lav.ipynb'],
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
            try:
                rc = p.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                _u_kill(p)
                raise TimeoutError(f'second notebook stopped after {{budget:.0f}}s')
        assert rc == 0, f'nbconvert exit code {{rc}}'
        shutil.copyfile('/kaggle/working/submission.csv', U_DIR + '/lav.csv')  # it deleted the base file at start
        shutil.copyfile(U_DIR + '/lav.csv', '/kaggle/working/union_lav.csv')  # diagnostics: compare with that notebook alone
        U_LAV_OK = True
        print(f'UNION_LAV_DONE at {{time.time() - U_T0:.0f}}s', flush=True)
    except Exception as e:
        print('UNION_FALLBACK second notebook:', repr(e), flush=True)
    finally:
        try:
            shutil.copyfile(U_DIR + '/base.csv', '/kaggle/working/submission.csv.tmp')
            os.replace('/kaggle/working/submission.csv.tmp', '/kaggle/working/submission.csv')
        except Exception as e:
            print('UNION_RESTORE_FAILED:', repr(e), flush=True)
        try:
            print(''.join(open(U_DIR + '/lav_nbconvert.log').readlines()[-20:]), flush=True)
        except Exception:
            pass
'''
        mix = f'''# [union] rank-blend the base and the second notebook; any problem keeps the base's submission.csv
if U_LAV_OK:
    try:
        import numpy as np, pandas as pd
        W = {float(opt.get('blend', 0.5))}  # base weight
        base = pd.read_csv(U_DIR + '/base.csv', dtype={{'StudyInstanceUID': str}}).set_index('StudyInstanceUID')
        lav = pd.read_csv(U_DIR + '/lav.csv', dtype={{'StudyInstanceUID': str}}).set_index('StudyInstanceUID')
        assert base.index.is_unique and lav.index.is_unique, 'duplicate study ids'
        lav = lav.loc[base.index, base.columns]  # KeyError on any missing study or finding
        assert np.isfinite(base.values).all() and np.isfinite(lav.values).all(), 'non-finite predictions'
        assert not np.allclose(base.values, lav.values), 'the second output equals the base'
        mix = W * base.rank(pct=True) + (1 - W) * lav.rank(pct=True)
        rho = np.mean([base[c].corr(lav[c], method='spearman') for c in base.columns])  # before the replace
        p = '/kaggle/working/submission.csv'
        mix.reset_index().to_csv(p + '.tmp', index=False)
        os.replace(p + '.tmp', p)
        print(f'UNION_MIX_DONE base {{W}} + second {{1 - W:.2f}}, {{len(mix)}} studies, mean Spearman {{rho:.3f}}', flush=True)
    except Exception as e:
        print('UNION_FALLBACK mix, keeping the base submission.csv:', repr(e), flush=True)
'''
        n_tb = 0  # its own 8 h budget counts from its own start; make it the real remaining time (unchanged when run alone)
        for c in nb2['cells']:
            s2, k = re.subn(r'^(\s*)TIME_BUDGET = 8\.0 \* 3600\b', r"\1TIME_BUDGET = float(__import__('os').environ.get('UNION_BUDGET_S', 8.0 * 3600))",
                            ''.join(c['source']), flags=re.M)
            c['source'], n_tb = s2.splitlines(keepends=True), n_tb + k
        assert n_tb == 2, f'union: expected 2 "TIME_BUDGET = 8.0 * 3600" lines in base2, found {n_tb}'
        lav_cell = '%%writefile /tmp/union/lav.ipynb\n' + json.dumps(nb2)
        assert json.loads(lav_cell.split('\n', 1)[1]) == nb2  # the writefile body is the (budget-patched) second notebook
        nb['cells'] = [cell(snap)] + nb['cells'] + [cell(save), cell(lav_cell), cell(run), cell(mix)]
        if opt.get('first') == 'base2':  # the all-or-nothing second notebook FIRST (pristine docker, no snapshot needed), then
            # the base with NB_T0 = ours, so its own R6 deadline (8.3 h from NB_T0) counts our time and only cuts its late studies
            run1 = f'''# [union] the second public notebook FIRST (unchanged except its TIME_BUDGET) in its own kernel; on timeout its process
# tree is killed; then the base runs with its clock started here
import os, sys, time, signal, shutil, subprocess
NB_T0 = time.time()
U_DIR, U_T0, U_LAV_OK, U_BASE_OK = '/tmp/union', NB_T0, False, False
''' + run[run.index('def _u_kill'):run.index('if U_BASE_OK:')] + f'''try:
    budget = {float(opt.get('lav_h', 3.3))} * 3600
    env = dict(os.environ, UNION_BUDGET_S=str(budget - 600))  # its own deadline, inside ours
    with open(U_DIR + '/lav_nbconvert.log', 'w') as log:
        p = subprocess.Popen([sys.executable, '-m', 'nbconvert', '--to', 'notebook', '--execute',
                              '--ExecutePreprocessor.timeout=-1', '--ExecutePreprocessor.kernel_name=python3',
                              '--output-dir', U_DIR, '--output', 'lav_out', U_DIR + '/lav.ipynb'],
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        try:
            rc = p.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            _u_kill(p)
            raise TimeoutError(f'second notebook stopped after {{budget:.0f}}s')
    assert rc == 0, f'nbconvert exit code {{rc}}'
    shutil.copyfile('/kaggle/working/submission.csv', U_DIR + '/lav.csv')
    shutil.copyfile(U_DIR + '/lav.csv', '/kaggle/working/union_lav.csv')  # diagnostics: compare with that notebook alone
    U_LAV_OK = True
    print(f'UNION_LAV_DONE at {{time.time() - U_T0:.0f}}s', flush=True)
except Exception as e:
    print('UNION_FALLBACK second notebook:', repr(e), flush=True)
finally:
    if os.path.exists('/kaggle/working/submission.csv'):
        os.remove('/kaggle/working/submission.csv')  # the base writes its own
    try:
        print(''.join(open(U_DIR + '/lav_nbconvert.log').readlines()[-20:]), flush=True)
    except Exception:
        pass
'''
            save1 = '''# [union] the base's submission, for the mix and as diagnostics
import shutil
try:
    shutil.copyfile('/kaggle/working/submission.csv', U_DIR + '/base.csv')
    shutil.copyfile(U_DIR + '/base.csv', '/kaggle/working/union_base.csv')
    U_BASE_OK = True
    print(f'UNION_BASE_SAVED at {time.time() - U_T0:.0f}s', flush=True)
except Exception as e:
    print('UNION_BASE_SAVE_FAILED:', repr(e), flush=True)
'''
            base = nb['cells'][1:-4]
            for c in base:  # the base's clock must count from our first cell, as in blend mode
                c['source'] = re.sub(r'^NB_T0 = time\.time\(\)', "NB_T0 = globals().get('NB_T0', time.time())", ''.join(c['source']),
                                     flags=re.M).splitlines(keepends=True)
            nb['cells'] = [cell("import os\nos.makedirs('/tmp/union', exist_ok=True)\n"), cell(lav_cell), cell(run1)] + base + [
                cell(save1), cell(mix.replace('if U_LAV_OK:', 'if U_LAV_OK and U_BASE_OK:'))]
            assert sum("NB_T0 = globals().get('NB_T0', time.time())" in ''.join(c['source']) for c in nb['cells']) == 1
        for k in ('dataset_sources', 'kernel_sources', 'model_sources'):
            meta[k] = list(dict.fromkeys([s for s in meta.get(k, []) + meta2.get(k, []) if s]))
    if mode == 'blend':  # ours FIRST (appended last, the base's 9 h guards would starve it on the hidden test), blend LAST
        n = 0
        for c in nb['cells']:  # the base's clock must count from our cell, so keep the NB_T0 we already set
            s, k = re.subn(r'^NB_T0 = time\.time\(\)', "NB_T0 = globals().get('NB_T0', time.time())", ''.join(c['source']), flags=re.M)
            c['source'], n = s.splitlines(keepends=True), n + k
        # nb_t0=0: a base with its own clock (e.g. lavinwins' T0/TIME_BUDGET); then L + ours_min must fit in 9 h (check by hand)
        assert n == int(opt.get('nb_t0', 1)), f'blend: expected {opt.get("nb_t0", 1)} "NB_T0 = time.time()" clock start(s), found {n}'
        ours = f'''# [knee] our models run first in their own process (GPU memory freed on exit); never raises
import os, sys, time, subprocess
NB_T0 = time.time()
try:
    rc = subprocess.run([sys.executable, '-u', '/kaggle/working/knee_ours.py'], timeout={int(opt.get('ours_min', 60))} * 60).returncode
    assert rc == 0, f'exit code {{rc}}'
    os.replace('/kaggle/working/submission.csv', '/kaggle/working/ours.csv')
    print(f'KNEE_OURS_DONE {{time.time() - NB_T0:.0f}}s', flush=True)
except Exception as e:
    print('KNEE_OURS_FAILED, the base runs alone:', repr(e), flush=True)
'''
        mix = f'''# [knee] rank-blend ours.csv into the base's submission.csv; any problem keeps the base file
try:
    import os, numpy as np, pandas as pd
    W = {float(opt['blend'])}  # base weight
    p = '/kaggle/working/submission.csv'
    base = pd.read_csv(p, dtype={{'StudyInstanceUID': str}}).set_index('StudyInstanceUID')
    ours = pd.read_csv('/kaggle/working/ours.csv', dtype={{'StudyInstanceUID': str}}).set_index('StudyInstanceUID')
    ours = ours.loc[base.index, base.columns]  # KeyError on any missing study or finding
    assert np.isfinite(ours.values).all() and np.isfinite(base.values).all(), 'non-finite predictions'
    mix = W * base.rank(pct=True) + (1 - W) * ours.rank(pct=True)
    mix.reset_index().to_csv(p + '.tmp', index=False)
    os.replace(p + '.tmp', p)
    print(f'KNEE_MIX_DONE base {{W}} + ours {{1 - W:.2f}}, {{len(mix)}} studies', flush=True)
except Exception as e:
    print('KNEE_MIX_FAILED, keeping the base submission.csv:', repr(e), flush=True)
'''
        nb['cells'] = [cell('%%writefile /kaggle/working/knee_ours.py\n' + src), cell(ours)] + nb['cells'] + [cell(mix)]
    json.dump(nb, open(os.path.join(out, f'{name}.ipynb'), 'w', encoding='utf-8'), indent=1)
    meta.pop('id_no', None)  # a pulled notebook's id_no would make the push target its author's kernel
    meta.update(id=f'ayushkasare/{name}', title=name, code_file=f'{name}.ipynb', is_private=True,
                kernel_sources=meta['kernel_sources'] + (trains if mode == 'blend' else []))
else:
    open(os.path.join(out, f'{name}.py'), 'w', encoding='utf-8').write(src)
    meta = dict(id=f'ayushkasare/{name}', title=name, code_file=f'{name}.py', language='python',
                kernel_type='script', is_private=True, competition_sources=[COMP],
                enable_gpu=mode != 'prep', enable_internet=mode == 'train', kernel_sources=kernel_sources,
                dataset_sources=['stevenleehans/rsna-knee-llm-report-labels',
                                 'marwanmath/resnet-50-radimagenet-marwan'] if mode == 'train' else [])
    if mode == 'train' and opt.get('labels') in ('rubric', 'consensus'):
        meta['dataset_sources'].append('ayushkasare/knee-rubric-labels')
    if 'cpu' in opt:
        meta['enable_gpu'] = False
    if meta['enable_gpu']:
        meta['machine_shape'] = 'NvidiaTeslaT4'
json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=2)
print(f'built {out}: {mode} {opt}')
