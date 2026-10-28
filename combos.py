"""Rank-avg macro AUC (vs v4b labels, gold-58) of every subset of the given runs, on the studies all of them cover.
usage: python combos.py runs/eff320ptl-5f runs/cnx288r-5f runs/eff31-ptl-4f   (each dir holds an oof.csv)"""
import sys, itertools, pandas as pd
from sklearn.metrics import roc_auc_score
import knee
knee.find_input = lambda n: {'rsna-knee-abnormality-detection': 'data', 'rsna-knee-llm-report-labels': 'labels/rsna-knee-llm-report-labels'}[n]
_, y, gold = knee.load_targets()
o = {d.rstrip('/').split('/')[-1]: pd.read_csv(f'{d}/oof.csv', index_col=0)[knee.LABELS] for d in sys.argv[1:]}
common = sorted(set.intersection(*(set(v.index) for v in o.values())))
o = {k: v.loc[common].rank(pct=True) for k, v in o.items()}
def m(p, ref):
    ids = [i for i in p.index if i in ref.index]
    ok = [l for l in knee.LABELS if 0 < (ref.loc[ids, l] > 0.5).sum() < len(ids)]  # a one-class finding has no AUC (small gold subsets)
    return sum(roc_auc_score(ref.loc[ids, l] > 0.5, p.loc[ids, l]) for l in ok) / len(ok)
print(len(common), 'studies')
for r in range(1, len(o) + 1):
    for c in itertools.combinations(o, r):
        p = sum(o[k] for k in c) / r
        print(f'{m(p, y):.4f} {m(p, gold):.4f}  ' + ' + '.join(c))
