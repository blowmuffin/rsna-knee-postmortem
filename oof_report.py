"""Per-label OOF AUC of downloaded training runs vs report labels and vs the 58 gold studies, plus the
rank-average ensemble and pairwise rank correlation (blend headroom).
usage: python oof_report.py runs/knee-train runs/knee-train-rad ...   (each dir holds that kernel's oof.csv)"""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import knee

knee.find_input = lambda n: {'rsna-knee-abnormality-detection': 'data',
                             'rsna-knee-llm-report-labels': 'labels/rsna-knee-llm-report-labels'}[n]
_, y_eval, gold = knee.load_targets()
oofs = {d.rstrip('/\\').split('/')[-1].split('\\')[-1]: pd.read_csv(f'{d}/oof.csv', index_col=0) for d in sys.argv[1:]}
common = sorted(set.intersection(*(set(o.index) for o in oofs.values())))
oofs = {k: o.loc[common, knee.LABELS] for k, o in oofs.items()}
if len(oofs) > 1:
    oofs['rank-avg'] = sum(o.rank(pct=True) for o in oofs.values()) / len(oofs)


def aucs(p, ref):
    ids = [i for i in p.index if i in ref.index]
    return pd.Series({l: roc_auc_score(ref.loc[ids, l] > 0.5, p.loc[ids, l]) for l in knee.LABELS
                      if 0 < (ref.loc[ids, l] > 0.5).sum() < len(ids)})


rows = {}
for k, p in oofs.items():
    rows[(k, 'labels')] = aucs(p, y_eval)
    rows[(k, 'gold')] = aucs(p, gold)
t = pd.DataFrame(rows).T
t.insert(0, 'MACRO', t.mean(1))
pd.set_option('display.width', 250)
print(f'{len(common)} studies with OOF from every run\n')
print(t.round(3).to_string())
names = [k for k in oofs if k != 'rank-avg']
if len(names) > 1:
    print('\nmean per-label Spearman between runs (lower = more blend headroom):')
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = np.mean([oofs[a][l].corr(oofs[b][l], method='spearman') for l in knee.LABELS])
            print(f'  {a} vs {b}: {r:.3f}')
