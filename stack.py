"""Second stage: per-label logistic regression on every run's OOF logits (all 12 labels -> cross-label evidence),
cross-validated on the same report-grouped folds. Compares against the plain rank average.
usage: python stack.py runs/knee-train-eff31-mil runs/knee-train-rad31-mil ... [--meta meta.csv]"""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import knee

knee.find_input = lambda n: {'rsna-knee-abnormality-detection': 'data',
                             'rsna-knee-llm-report-labels': 'labels/rsna-knee-llm-report-labels'}[n]
L = knee.LABELS
args = [a for a in sys.argv[1:] if not a.startswith('--')]
meta = sys.argv[sys.argv.index('--meta') + 1] if '--meta' in sys.argv else None
args = [a for a in args if a != meta]
_, ye, gold = knee.load_targets()
oofs = [pd.read_csv(f'{d}/oof.csv', index_col=0)[L] for d in args]
ids = sorted(set.intersection(*(set(o.index) for o in oofs)))
P = [o.loc[ids].clip(1e-4, 1 - 1e-4) for o in oofs]
X = np.hstack([np.log(p.values / (1 - p.values)) for p in P])  # n x (runs*12) logits
if meta:
    m = pd.read_csv(meta, index_col=0).reindex(ids)
    X = np.hstack([X, m.fillna(m.median()).values])
fold = knee.report_folds().reindex(ids).values
Y = ye.loc[ids, L].values


def macro(p, ref, rows):
    return np.mean([roc_auc_score(ref[rows, j] > .5, p[rows, j]) for j in range(len(L))
                    if 0 < (ref[rows, j] > .5).sum() < len(rows)])


base = sum(p.rank(pct=True).values for p in P) / len(P)
out = {}
for C in (0.003, 0.01, 0.03, 0.1):
    st = np.zeros_like(Y)
    for f in range(5):
        tr, va = fold != f, fold == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        for j in range(len(L)):  # soft targets: duplicate rows with weights y and 1-y
            Xt = (X[tr] - mu) / sd
            lr = LogisticRegression(C=C, max_iter=2000)
            lr.fit(np.vstack([Xt, Xt]), np.r_[np.ones(tr.sum()), np.zeros(tr.sum())],
                   sample_weight=np.r_[Y[tr, j], 1 - Y[tr, j]])
            st[va, j] = lr.predict_proba((X[va] - mu) / sd)[:, 1]
    out[C] = st
all_rows = np.arange(len(ids))
g = np.array([i in gold.index for i in ids])
G = np.full_like(Y, np.nan); G[g] = gold.reindex([i for i in ids if i in gold.index])[L].values
grows = np.where(g)[0]
print(f'{len(ids)} studies, {X.shape[1]} features')
print(f'rank-avg      labels {macro(base, Y, all_rows):.4f}  gold {macro(base, G, grows):.4f}')
for C, st in out.items():
    print(f'stack C={C:<6} labels {macro(st, Y, all_rows):.4f}  gold {macro(st, G, grows):.4f}')
