"""How noisy is the public/private macro-AUC? Binormal simulation (illustrative; prevalences from our hard labels).
Two submissions with per-finding latent score = y*mu + e; errors correlated rho_e between them (public stacks rho~0.86-0.89)."""
import numpy as np
from scipy.stats import norm, rankdata
rng = np.random.default_rng(0)
prev = np.array([.216, .155, .444, .19, .371, .268, .456, .596, .129, .248, .218, .071])
def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def macro(Y, S): return np.mean([auc(Y[:, j], S[:, j]) for j in range(12)])
def run(n, true_auc_a, true_auc_b, rho_e=0.87, reps=1500):
    mu_a = np.sqrt(2) * norm.ppf(true_auc_a); mu_b = np.sqrt(2) * norm.ppf(true_auc_b)
    out = []
    for _ in range(reps):
        Y = (rng.random((n, 12)) < prev).astype(int)
        z = rng.standard_normal((n, 12)); e1 = rng.standard_normal((n, 12))
        ea = z; eb = rho_e * z + np.sqrt(1 - rho_e ** 2) * e1
        out.append((macro(Y, Y * mu_a + ea), macro(Y, Y * mu_b + eb)))
    o = np.array(out); return o
for n, lab in ((390, 'public ~30% of 1300'), (910, 'private ~70%'), (1300, 'all')):
    o = run(n, 0.960, 0.960)
    print(f'{lab:22s} n={n}: SD of one team macro-AUC (true 0.960) = {o[:,0].std():.4f}; SD of A-B difference (same true AUC, err corr .87) = {(o[:,0]-o[:,1]).std():.4f}')
o = run(390, 0.9605, 0.9595)
print('public: P(model with +0.001 true AUC outranks the other) = %.2f' % (o[:, 0] > o[:, 1]).mean())
o = run(390, 0.9620, 0.9595)
print('public: P(+0.0025 true AUC outranks) = %.2f' % (o[:, 0] > o[:, 1]).mean())
