"""Leaderboard analysis over the saved snapshots. Prints ASCII only (no team names)."""
import glob, re, sys
import pandas as pd

OUR = 'ayushkasare'
files = sorted(glob.glob('lb_history/*.csv'))
snaps = {}
for f in files:
    ts = re.search(r'(\d{4}-\d\d-\d\dT\d\d_\d\d_\d\d)', f).group(1)
    df = pd.read_csv(f)
    snaps[ts] = df


def ours(df):
    m = df[df.TeamMemberUserNames.fillna('').str.split(',').apply(lambda l: OUR in [x.strip() for x in l])]
    return m.iloc[0] if len(m) else None


print('== per-snapshot summary')
for ts, df in snaps.items():
    o = ours(df)
    n = len(df)
    print(ts, 'teams', n, 'top score', df.Score.max(),
          'our rank', None if o is None else int(o.Rank), 'our score', None if o is None else o.Score,
          'subs', None if o is None else int(o.SubmissionCount))

last_ts = sorted(snaps)[-1]
df = snaps[last_ts]
print('\n== latest', last_ts)
n = len(df)
print('teams', n)
# rank-by-score blocks
blk = df.groupby('Score').Rank.agg(['min', 'max', 'count']).sort_index(ascending=False)
print('\nblocks (score: rank range, count) top 25 distinct scores')
print(blk.head(25).to_string())
print('\nbig blocks (count >= 100)')
print(blk[blk['count'] >= 100].to_string())

# medal lines: Kaggle rule for 1000+ teams: gold top 10 + 0.2%; silver top 5%; bronze top 10%
import math
gold = 10 + math.ceil(0.002 * n)  # kaggle: 10 + 0.2% of teams above 500 (rounded)
gold_alt = 10 + 0.002 * n
silver = math.ceil(0.05 * n)
bronze = math.ceil(0.10 * n)
print('\nmedal lines (rank cutoffs): gold %d (10+0.2%%=%.1f), silver %d, bronze %d' % (gold, gold_alt, silver, bronze))
for name, r in (('gold', gold), ('silver', silver), ('bronze', bronze)):
    row = df[df.Rank == r]
    sc = row.Score.iloc[0] if len(row) else None
    print(name, 'rank', r, 'score at that rank', sc)
    if sc is not None:
        b = blk.loc[sc]
        print('   block of that score: ranks %d-%d (%d teams)' % (b['min'], b['max'], b['count']))
# ranks of selected
print('\nscore at selected ranks')
for r in (1, 3, 5, 10, 22, 30, 50, 100, 200, 306, 500, 612, 1000):
    row = df[df.Rank == r]
    if len(row): print(r, row.Score.iloc[0])
# distribution
print('\ncounts at/above thresholds')
for t in (0.965, 0.964, 0.963, 0.962, 0.961, 0.96, 0.959, 0.958, 0.957, 0.956, 0.955, 0.95, 0.94, 0.93, 0.9):
    print(t, int((df.Score >= t).sum()))

# top 40 table (no names)
print('\ntop 40 (rank, score, lastsub, nsubs, nmembers)')
for _, r in df.head(40).iterrows():
    print(int(r.Rank), r.Score, r.LastSubmissionDate[:16], int(r.SubmissionCount), len(str(r.TeamMemberUserNames).split(',')))

# changes between snapshots: score counts across snapshots at thresholds
print('\n== history: teams at >= threshold per snapshot')
ths = (0.963, 0.962, 0.961, 0.960, 0.959, 0.958, 0.957)
print('snapshot', *ths)
for ts, d in snaps.items():
    print(ts, len(d), *[int((d.Score >= t).sum()) for t in ths])

print('\n== history: score of rank r per snapshot')
rs = (1, 5, 10, 22, 30, 50, 100, 306, 612)
print('snapshot', *rs)
for ts, d in snaps.items():
    print(ts, *[d[d.Rank == r].Score.iloc[0] if len(d[d.Rank == r]) else None for r in rs])

# top-30 membership change: use TeamId
print('\n== top 30 TeamId movement between first and last snapshot')
first_ts = sorted(snaps)[0]
for a, b in ((first_ts, last_ts), (sorted(snaps)[-2], last_ts)):
    A, B = snaps[a].set_index('TeamId'), snaps[b].set_index('TeamId')
    topA, topB = set(A.index[A.Rank <= 30]), set(B.index[B.Rank <= 30])
    print(a, '->', b, 'in top30 both', len(topA & topB), 'new', len(topB - topA), 'left', len(topA - topB))
    for tid in sorted(topB - topA, key=lambda t: B.loc[t, 'Rank']):
        ra = int(A.loc[tid, 'Rank']) if tid in A.index else None
        sa = A.loc[tid, 'Score'] if tid in A.index else None
        print('   new: rank now %d score %s (was rank %s score %s) subs %d' % (B.loc[tid, 'Rank'], B.loc[tid, 'Score'], ra, sa, B.loc[tid, 'SubmissionCount']))

# our row
o = ours(df)
print('\nOUR ROW:', {k: (v if k != 'TeamName' else '<hidden>') for k, v in o.items()})
same = df[df.Score == o.Score]
print('our block %.3f: ranks %d-%d (%d teams); we are %dth inside' % (o.Score, same.Rank.min(), same.Rank.max(), len(same), int(o.Rank) - same.Rank.min() + 1))
# teams in block that submitted before/after us
same = same.copy(); same['t'] = pd.to_datetime(same.LastSubmissionDate)
print('block teams with last submission earlier than ours:', int((same.t < pd.to_datetime(o.LastSubmissionDate)).sum()))
print('how many of the block rows have SubmissionCount==1:', int((same.SubmissionCount == 1).sum()))
# within-block order vs last submission date: is rank ordered by time (ties)?
same = same.sort_values('Rank')
rho = same[['Rank']].assign(tt=same.t.astype('int64')).corr(method='spearman').iloc[0, 1]
print('spearman(rank, last-submission time) within our block: %.3f' % rho)
