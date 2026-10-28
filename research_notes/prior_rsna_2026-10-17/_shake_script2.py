import csv, statistics as st, collections
rows=list(csv.DictReader(open('merged.csv',encoding='utf-8')))
for r in rows:
    for k in ('pub_rank','priv_rank'): r[k]=int(r[k])
    for k in ('pub_score','priv_score'): r[k]=float(r[k])
# group by public score exact
by=collections.defaultdict(list)
for r in rows: by[r['pub_score']].append(r)
big=[(s,v) for s,v in by.items() if len(v)>=8]
big.sort(key=lambda x:-x[0])
print('public block score, n, public rank range, private rank min/median/max, private score min/median/max')
for s,v in big:
    pr=[x['priv_rank'] for x in v]; ps=[x['priv_score'] for x in v]; pb=[x['pub_rank'] for x in v]
    print(s,len(v),(min(pb),max(pb)),(min(pr),int(st.median(pr)),max(pr)),(round(min(ps),4),round(st.median(ps),4),round(max(ps),4)))
# the 0.691 block detail
blk=[r for r in rows if abs(r['pub_score']-0.691)<5e-4 and r['pub_score']>=0.6905]
print('block 0.691-ish (public score 0.6905-0.6915)',len(blk))
pr=sorted(x['priv_rank'] for x in blk); print('private ranks pctl 5/25/50/75/95',[pr[int(len(pr)*p)] for p in (.05,.25,.5,.75,.95)], 'best',pr[:5],'worst',pr[-3:])
# exact 0.69100 block
# Teams that moved up the most overall
rows.sort(key=lambda r:r['pub_rank']-r['priv_rank'],reverse=True)
print('biggest gains',[(r['team'],r['pub_rank'],r['priv_rank'],r['pub_score'],r['priv_score']) for r in rows[:8]])
rows.sort(key=lambda r:r['pub_rank']-r['priv_rank'])
print('biggest drops',[(r['team'],r['pub_rank'],r['priv_rank'],r['pub_score'],r['priv_score']) for r in rows[:8]])
# among public rank 1..60 rank change
rows.sort(key=lambda r:r['pub_rank'])
print('public ranks 1-60 -> private')
print([(r['pub_rank'],r['priv_rank']) for r in rows[:60]])
# private-score drop vs public score level
for lo,hi in ((0.85,1),(0.8,0.85),(0.75,0.8),(0.7,0.75),(0.6,0.7),(0.5,0.6)):
    v=[r for r in rows if lo<=r['pub_score']<hi]
    if v: print((lo,hi),len(v),'mean priv-pub',round(st.mean(r['priv_score']-r['pub_score'] for r in v),4),'mean |rank move|',round(st.mean(abs(r['pub_rank']-r['priv_rank']) for r in v),1))
