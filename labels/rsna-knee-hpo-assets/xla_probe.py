"""Where do the 69 ms/img of XLA training go? Times, on one TPU core with a fixed (2,12,3,384,384) batch:
loader alone | fwd (eval, bf16) | fwd+bwd (bf16) | full step AdamW+clip with OneCycle LR | full step constant LR | fwd+bwd fp32.
Prints ms/img per phase and torch_xla compile counts, so recompilation shows up as a count that keeps growing."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import train_knee as tk
from torch.utils.data import DataLoader
import torch_xla, torch_xla.core.xla_model as xm, torch_xla.debug.metrics as met
ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"; dev = torch_xla.device()
def stamp(m): print(time.strftime("%T"), m, flush=True)
def ncomp():
    d = met.metric_data("CompileTime"); return int(d[0]) if d else 0
stamp(f"device {dev} | torch_xla {torch_xla.__version__} | mem {xm.get_memory_info(dev)}")
ids = tk._open_ids(); id2row = {u: i for i, u in enumerate(ids)}; sub = list(ids[:64]); labels = {u: np.zeros(12, np.float32) for u in sub}
ds = tk.StudyWindows(tk.HERE, sub, id2row, labels, 384, 12, train=True, norm="imagenet")
dl = DataLoader(ds, batch_size=2, num_workers=8, collate_fn=tk.collate, drop_last=True)
t = time.time(); n = 0
for x, y in dl: n += x.shape[0]
stamp(f"loader: {n} studies in {time.time()-t:.1f}s -> {(time.time()-t)/n*1000:.0f} ms/study (8 workers, first read)")
t = time.time(); n = 0
for x, y in dl: n += x.shape[0]
stamp(f"loader: {n} studies in {time.time()-t:.1f}s -> {(time.time()-t)/n*1000:.0f} ms/study (second pass, cached)")
x = x.to(dev); y = y.to(dev); xm.mark_step(); NI = x.shape[0] * x.shape[1]
bb = tk.build_backbone(ARCH, pretrained=True); model = tk.RaptorClassifier(bb, F_dim=bb.num_features).to(dev)
lossf = torch.nn.BCEWithLogitsLoss()
hp = [p for n_, p in model.named_parameters() if not n_.startswith("backbone.")]; bp = [p for n_, p in model.named_parameters() if n_.startswith("backbone.")]
opt = torch.optim.AdamW([{"params": bp, "lr": 3e-5}, {"params": hp, "lr": 1e-3}], weight_decay=0.02)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[3e-5, 1e-3], total_steps=10000, pct_start=0.15)
def phase(name, fn, iters=8, warm=2):
    c0 = ncomp(); t = time.time()
    for _ in range(warm): fn(); xm.mark_step()
    xm.wait_device_ops(); c1 = ncomp(); tw = time.time() - t; t = time.time()
    for _ in range(iters): fn(); xm.mark_step()
    xm.wait_device_ops(); dt = time.time() - t
    stamp(f"{name:34s} {dt/iters*1000:7.0f} ms/step {dt/iters/NI*1000:6.1f} ms/img | warmup {tw:.0f}s compiles {c1-c0} | timed compiles {ncomp()-c1}")
def f_fwd_eval():
    model.eval()
    with torch.no_grad(), torch.autocast("xla", dtype=torch.bfloat16): model(x)
def f_fwd_train_nograd():
    model.train()
    with torch.no_grad(), torch.autocast("xla", dtype=torch.bfloat16): model(x)
def f_fwdbwd():
    model.train(); model.zero_grad(set_to_none=True)
    with torch.autocast("xla", dtype=torch.bfloat16): loss = lossf(model(x).float(), y)
    loss.backward()
def f_step_onecycle():
    f_fwdbwd(); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step(); sched.step()
def f_step_const():
    f_fwdbwd(); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step()
def f_fwdbwd_noclip_step():
    f_fwdbwd(); opt.step()
def f_fwdbwd_fp32():
    model.train(); model.zero_grad(set_to_none=True); loss = lossf(model(x).float(), y); loss.backward()
phase("fwd eval bf16", f_fwd_eval)
phase("fwd train-mode no_grad bf16", f_fwd_train_nograd)
phase("fwd+bwd bf16", f_fwdbwd)
phase("fwd+bwd + AdamW (no clip)", f_fwdbwd_noclip_step)
phase("fwd+bwd + clip + AdamW const LR", f_step_const)
phase("fwd+bwd + clip + AdamW OneCycle", f_step_onecycle)
phase("fwd+bwd fp32 (no autocast)", f_fwdbwd, iters=4) if False else phase("fwd+bwd fp32 (no autocast)", f_fwdbwd_fp32, iters=4)
stamp("mem " + str(xm.get_memory_info(dev)))
print(met.short_metrics_report(), flush=True)
