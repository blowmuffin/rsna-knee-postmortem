"""Consensus training labels: per-cell mean of 4 independent LLM report reads, the uncertain middle masked (NaN = no loss).
   python consensus.py -> labels/knee-rubric-labels/consensus_labels.csv, then
   python -m kaggle datasets version -p labels/knee-rubric-labels -m "consensus labels"

Reads: steven v4b (+SILENT_FIX), our Claude rubric, pilkwang v2, gpt56sol. Gold-58: 0.901 macro AUC vs 0.893 (v4b) and
0.898 (rubric); all within gold noise. Mask: 0.3 < mean < 0.7 (10.5% of non-gold cells). On gold the consensus is on
the wrong side for 60% of such cells vs 12% of kept ones, and per-label positive rates barely move."""
import pandas as pd
import knee

knee.find_input = lambda n: {'rsna-knee-abnormality-detection': 'data',
                             'rsna-knee-llm-report-labels': 'labels/rsna-knee-llm-report-labels'}[n]
LO, HI = 0.3, 0.7

knee.LABEL_SRC, knee.PSEUDO_A = 'v4b', 0.0
_, v4b, gold = knee.load_targets()  # v4b + SILENT_FIX, gold rows hold the gold labels
read = lambda f: pd.read_csv(f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[knee.LABELS].astype(float)
reads = [v4b, read('labels/knee-rubric-labels/rubric_labels.csv'), read('labels/rsna-knee-llm-labels/report_labels_v2.csv'),
         read('labels/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv')]
m = pd.concat(reads).groupby(level=0).mean().reindex(v4b.index)  # NaN-skipping mean (pilkwang lacks one study)
assert m.notna().all().all() and len(m) == len(v4b)
y = m.mask((m > LO) & (m < HI))
y.loc[gold.index] = gold.values  # load_targets overwrites gold rows anyway; keep them unmasked here too
print(f'masked {y.isna().values.mean():.3f} of cells;', y.isna().mean().round(2).to_dict())
y.to_csv('labels/knee-rubric-labels/consensus_labels.csv')
