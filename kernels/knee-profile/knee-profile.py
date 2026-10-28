# Throwaway: where does a training step's time go on Kaggle? (file read vs decompress vs GPU)
import glob, os, time, io, shutil, numpy as np, torch, timm
d = (glob.glob('/kaggle/input/knee-prep/cache') + glob.glob('/kaggle/input/notebooks/*/knee-prep/cache'))[0]
fs = sorted(os.listdir(d))[1000:1040]
t = time.time(); raw = [open(f'{d}/{f}', 'rb').read() for f in fs]; tr = time.time() - t
t = time.time(); arrs = [np.load(io.BytesIO(b))['x'] for b in raw]; td = time.time() - t
print(f'read {tr/len(fs):.3f}s/study ({sum(map(len,raw))/tr/1e6:.1f} MB/s), decompress {td/len(fs):.3f}s/study', flush=True)
t = time.time(); shutil.copytree(d, '/kaggle/tmp_cache', dirs_exist_ok=True) if False else None
du = os.popen(f'df -h /kaggle/working /tmp {d} | cat').read(); print(du, flush=True)
print('cpus', os.cpu_count(), 'ram GB', os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9, flush=True)
for name in ['efficientnet_b0.ra_in1k', 'coatnet_0_rw_224.sw_in1k', 'resnet50']:
    m = timm.create_model(name, pretrained=False, num_classes=12).cuda()
    opt = torch.optim.AdamW(m.parameters()); sc = torch.amp.GradScaler()
    x = torch.randn(160, 3, 224, 224, device='cuda')
    for i in range(8):
        if i == 3: torch.cuda.synchronize(); t = time.time()
        with torch.autocast('cuda', dtype=torch.float16):
            loss = m(x).float().mean()
        opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update()
    torch.cuda.synchronize(); dt = (time.time() - t) / 5
    print(f'{name}: {dt:.3f}s per 160-image step on 1 T4 = {160/dt:.0f} img/s, mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB', flush=True)
    del m, opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
