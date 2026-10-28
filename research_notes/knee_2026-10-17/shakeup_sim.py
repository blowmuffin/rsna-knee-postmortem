"""Illustrative public->private shake-up from the current public score distribution (empirical Bayes, independent noise).
Assumes: public p_i = t_i + N(0, sp), private q_i = t_i + N(0, sq); t_i | p_i ~ N(mu + k (p_i - mu), k sp^2), mu = mean public score of the top 2000.
Unique (non-clone) submissions only: exact clones share one private score and tie-break by submission time, which this ignores."""
import glob, numpy as np, pandas as pd
rng = np.random.default_rng(1)
f = sorted(glob.glob('lb_1601/*.csv'))[-1]
df = pd.read_csv(f)
p = df.Score.values + rng.uniform(-0.0005, 0.0005, len(df))   # undo 3-dp rounding
N = len(p); mu = np.sort(p)[::-1][:2000].mean()
lines = {'gold(22)': 22, 'silver(307)': 307, 'bronze(614)': 614}
sel = {'0.958 (rank ~540)': (df.Score == 0.958).values, '0.959': (df.Score == 0.959).values, '0.957': (df.Score == 0.957).values, '0.961': (df.Score == 0.961).values, '0.963': (df.Score == 0.963).values}
print('N', N, 'mu(top2000)=%.4f' % mu)
for sp, sq, k in ((0.0025, 0.0016, 0.5), (0.0025, 0.0016, 0.8), (0.0030, 0.0020, 0.8)):
    res = {n: [0, 0, 0] for n in sel}; cnt = {n: 0 for n in sel}; reps = 40
    for _ in range(reps):
        t = mu + k * (p - mu) + np.sqrt(k) * sp * rng.standard_normal(N)
        q = t + sq * rng.standard_normal(N)
        rk = (-q).argsort().argsort() + 1
        for n, m in sel.items():
            for j, (ln, r) in enumerate(lines.items()):
                res[n][j] += (rk[m] <= r).sum()
            cnt[n] += m.sum()
    print(f'sp={sp} sq={sq} k={k}:', ' | '.join(f'{n}: ' + '/'.join(f'{100*res[n][j]/cnt[n]:.1f}%' for j in range(3)) for n in sel), ' (P gold/silver/bronze)')
