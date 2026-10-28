"""Remote worker for the hyper-parameter screen: ask hpo_server for a trial, train it with train_knee.py (8-epoch compressed
schedule, OOF300 scored every epoch), report each epoch (server decides pruning), tell the result. Loops until the session
budget is nearly spent. Runs on a Kaggle TPU kernel (HPO_XLA=1) or any machine with the corpus.

Env: HPO_URL, HPO_TOKEN, HPO_WHO, HPO_BUDGET_H (default 8.3), HPO_EST_TRIAL_H (2.5), HPO_EPOCHS (8), HPO_OOF_N (300),
     HPO_XLA (0/1), HPO_BS (1), HPO_WORKERS (4), HPO_TRAIN_DIR (train_knee.py + labels + folds), HPO_OOF_MASK, RAPTOR_INPUT,
     HPO_SMOKE=1 -> no ask: time the public recipe for 2 epochs on 240 studies and exit.
"""
import json, os, re, shutil, signal, subprocess, sys, time, urllib.request
E = os.environ.get
URL, TOKEN, WHO = E("HPO_URL", "").rstrip("/"), E("HPO_TOKEN", ""), E("HPO_WHO", "worker")
BUDGET_H, EST_H = float(E("HPO_BUDGET_H", "8.3")), float(E("HPO_EST_TRIAL_H", "2.5"))
EPOCHS, OOF_N = int(E("HPO_EPOCHS", "8")), int(E("HPO_OOF_N", "300"))
XLA, BS, WORKERS = E("HPO_XLA", "0") == "1", int(E("HPO_BS", "1")), int(E("HPO_WORKERS", "4"))
T = E("HPO_TRAIN_DIR", os.path.dirname(os.path.abspath(__file__))); OUT = E("HPO_OUT", os.getcwd())
MASK = E("HPO_OOF_MASK", f"{T}/train_labels_v0.parquet"); PY = sys.executable
T0 = time.time()
def log(*x): print(time.strftime("%F %T ") + " ".join(str(v) for v in x), flush=True)
def call(path, body, tries=40, wait=15):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "X-Token": TOKEN})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r: return json.loads(r.read())
        except Exception as e:
            log(f"[http] {path} failed ({type(e).__name__}: {str(e)[:80]}), retry {i+1}/{tries} in {wait}s"); time.sleep(wait)
    return None
def train_cmd(P, tag, epochs, extra=()):
    acc = max(1, int(P["accum"]) // BS)               # accum in the study = effective batch (bs 1 locally)
    cmd = [PY, f"{T}/train_knee.py", "--arch", "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "--res", "384", "--norm", "imagenet",
           "--workers", str(WORKERS), "--k", str(P["k"]), "--k_eval", E("HPO_KEVAL", "24"), "--bs", str(BS), "--accum", str(acc), "--epochs", str(epochs),
           "--bb_lr", f"{P['bb_lr']:.3g}", "--head_lr", f"{P['head_lr']:.3g}", "--wd", f"{P['wd']:.3g}", "--folds", "5", "--fold", "0",
           "--labels", f"{T}/labels_v0raw.parquet", "--fold_file", f"{T}/folds_v0.json", "--oof_every", str(OOF_N), "--oof_mask", MASK, "--tag", tag]
    cmd += ["--xla"] if XLA else ["--win_chunk", "4"]
    return cmd + list(extra)
def run(cmd, number=None):
    """Run one training; stream OOF lines to the server; return (last_value, epochs_seen, pruned)."""
    lf = f"{OUT}/train_{cmd[cmd.index('--tag')+1]}.log"; f = open(lf, "w")
    p = subprocess.Popen(cmd, cwd=T, stdout=f, stderr=subprocess.STDOUT); seen, last, pruned = -1, None, False
    while p.poll() is None:
        time.sleep(30)
        for m in re.finditer(r"^ep(\d+) OOF\d+ masked macro-AUC ([0-9.]+)", open(lf).read(), re.M):
            ep, v = int(m.group(1)), float(m.group(2))
            if ep > seen:
                seen, last = ep, v; log(f"trial {number} ep{ep} oof={v:.4f}")
                if number is not None:
                    r = call("/report", {"number": number, "step": ep, "value": v}, tries=4, wait=10)
                    if r and r.get("prune"):
                        p.send_signal(signal.SIGTERM); time.sleep(5); p.kill(); pruned = True; log(f"trial {number} PRUNED at ep{ep}")
    f.close()
    tag = cmd[cmd.index('--tag') + 1]                 # this trial's checkpoints only (the winner is retrained in full)
    for fn in os.listdir(T):
        if fn.endswith(".pt") and fn.startswith(f"raptor_ft_{tag}"): os.remove(os.path.join(T, fn))
    return last, seen + 1, pruned, p.returncode
if E("HPO_SMOKE", "0") == "1":
    P = {"k": 12, "bb_lr": 3e-5, "head_lr": 1e-3, "wd": 0.02, "accum": 8}
    for bs in [int(v) for v in E("HPO_SMOKE_BS", str(BS)).split(",")]:
        BS = bs; tag = f"smoke_bs{bs}"; cmd = train_cmd(P, tag, 2, extra=["--limit", "240"])
        log("SMOKE", " ".join(cmd)); t = time.time(); last, n, _, rc = run(cmd)
        txt = open(f"{OUT}/train_{tag}.log").read(); ts = [int(m.group(1)) for m in re.finditer(r"GOLD macro-AUC [0-9.]+ \(best [0-9.]+\) \| (\d+)s", txt)]
        log(f"SMOKE bs{bs} done rc={rc} epochs={n} oof={last} cumulative_s={ts} -> steady epoch {ts[1]-ts[0] if len(ts)>1 else '?'} s for 240 studies")
        print("\n".join(l for l in txt.splitlines() if l.startswith("  [xla]") or l.startswith("ep") or "Counter" in l or "Metric" in l or "Error" in l)[-4000:], flush=True)
    sys.exit(0)
if E("HPO_CONFIRM", "0") == "1":
    if E("HPO_PARAMS"):                                   # pinned recipe (label arms): no dependence on the study's state
        P = json.loads(E("HPO_PARAMS")); num, val = "pinned", float("nan")
    else:
        r = call("/best", {}); assert r and r.get("params"), f"no best trial from server: {r}"
        P, num, val = r["params"], r["number"], r["value"]
    folds = [int(x) for x in E("HPO_FOLDS", "0").split(",")]       # e.g. "1,2": several folds back to back in one session
    epochs = int(E("HPO_CONFIRM_EPOCHS", "16")); rc_all = 0
    for fold in folds:
        tag = (E("HPO_TAG") or f"confirm_t{num}").replace("{fold}", str(fold))
        cmd = train_cmd(P, tag, epochs); cmd[cmd.index("--fold") + 1] = str(fold)
        if E("HPO_LABELS"):                                   # alternative training labels (same parquet layout as labels_v0raw)
            cmd[cmd.index("--labels") + 1] = os.path.join(T, E("HPO_LABELS"))
        log(f"CONFIRM trial {num} (oof300={val:.4f}) {P} labels={E('HPO_LABELS') or 'labels_v0raw'} k_eval={E('HPO_KEVAL', '24')} -> {epochs} epochs fold {fold}, tag {tag}")
        lf = f"{OUT}/train_{tag}.log"; f = open(lf, "w"); t1 = time.time()
        rc = subprocess.call(cmd, cwd=T, stdout=f, stderr=subprocess.STDOUT); f.close(); rc_all |= rc
        for fn in os.listdir(T):
            if tag in fn and (fn.endswith(".pt") or fn.endswith(".npz") or fn.endswith(".json")): shutil.copy(os.path.join(T, fn), OUT)
        log(f"CONFIRM fold {fold} done rc={rc} in {(time.time()-t1)/3600:.2f} h; outputs: {[fn for fn in os.listdir(OUT) if tag in fn]}")
        print("\n".join(l for l in open(lf).read().splitlines() if l.startswith("ep") or "OOF" in l or "SWA" in l or "TOPK" in l or "[cv]" in l), flush=True)
    sys.exit(rc_all)
est = EST_H; n_done = 0; durs = []
while True:
    left = BUDGET_H - (time.time() - T0) / 3600
    if left < est: log(f"stop: {left:.2f} h left < est trial {est:.2f} h; {n_done} trials done"); break
    r = call("/ask", {"who": WHO})
    if r is None: log("no server; stopping"); break
    num, P = r["number"], r["params"]; log(f"trial {num} START {P}"); t1 = time.time()
    last, n, pruned, rc = run(train_cmd(P, f"r{num}", EPOCHS), number=num)
    dur = (time.time() - t1) / 3600
    if pruned: body = {"number": num, "state": "pruned"}
    elif last is None or rc != 0 and n == 0: body = {"number": num, "state": "fail"}
    else: body = {"number": num, "value": last}
    if not pruned: durs.append(dur); est = max(durs) * 1.05
    log(f"trial {num} END {body} in {dur:.2f} h")
    if call("/tell", body) is None:
        open(f"{OUT}/untold_results.jsonl", "a").write(json.dumps(body) + "\n"); log("tell failed; saved to untold_results.jsonl")
    n_done += 1
