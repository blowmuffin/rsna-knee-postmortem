import csv, glob, statistics as st, collections
pub=list(csv.DictReader(open(glob.glob('x/*.csv')[0],encoding='utf-8-sig')))
prv=list(csv.DictReader(open('lb_ov0.csv',encoding='utf-8')))
pub=[p for p in pub if p['TeamName']!='perfect_submission.parquet']
prv=[p for p in prv if p['team_name']!='perfect_submission.parquet']
print(len(pub),len(prv))
pr={p['TeamId']:(i+1,float(p['Score']),p['TeamName']) for i,p in enumerate(pub)}
vr={p['team_id']:(i+1,float(p['score']),p['team_name']) for i,p in enumerate(prv)}
common=set(pr)&set(vr); print('common',len(common))
rows=[(t,pr[t][0],vr[t][0],pr[t][1],vr[t][1],pr[t][2]) for t in common]
import math
# rank moves
mv=[(a-b) for _,a,b,*_ in rows]  # positive = improved
print('median abs move',st.median(abs(m) for m in mv),'mean abs',st.mean(abs(m) for m in mv))
# spearman
def rank(x): 
    s=sorted(range(len(x)),key=lambda i:x[i]); r=[0]*len(x)
    for k,i in enumerate(s): r[i]=k
    return r
a=[r[1] for r in rows]; b=[r[2] for r in rows]
ra,rb=rank(a),rank(b); n=len(a)
d2=sum((x-y)**2 for x,y in zip(ra,rb)); print('spearman',1-6*d2/(n*(n*n-1)))
# top-N private: where were they on public
print('PRIVATE top 20 -> public rank, public score, private score')
for t in sorted(common,key=lambda t:vr[t][0])[:25]:
    print(vr[t][0],pr[t][0],pr[t][1],vr[t][1],vr[t][2])
print('PUBLIC top 15 -> private rank')
for t in sorted(common,key=lambda t:pr[t][0])[:20]:
    print(pr[t][0],vr[t][0],pr[t][1],vr[t][1],pr[t][2])
# score gap
gaps=[r[3]-r[4] for r in rows]; print('mean pub-priv gap',st.mean(gaps),'median',st.median(gaps))
# top-N overlap
for N in (10,50,100):
    pt=set(sorted(common,key=lambda t:pr[t][0])[:N]); vt=set(sorted(common,key=lambda t:vr[t][0])[:N])
    print('top',N,'overlap',len(pt&vt))
# medal lines
nteams=len(prv)
for rk in (1,10,12,13,57,58,114,115):
    t=[t for t in common if vr[t][0]==rk]
    if t: print('private rank',rk,vr[t[0]][1],vr[t[0]][2])
# public score block sizes in the 0.80-0.87 range
cnt=collections.Counter(round(p[1],3) for p in pr.values())
print('public tie blocks (largest):',cnt.most_common(8))
cntv=collections.Counter(round(p[1],3) for p in vr.values())
print('private tie blocks (largest):',cntv.most_common(5))
# who dropped most among public top 100
big=[(pr[t][0],vr[t][0],pr[t][2]) for t in common if pr[t][0]<=100]
big.sort(key=lambda x:x[1]-x[0],reverse=True)
print('public top100 biggest drops',big[:8]); 
big.sort(key=lambda x:x[1]-x[0])
print('public top100 biggest gains',big[:8])
# distribution: public top 50 -> private rank percentiles
for N in (10,50,100):
    ranks=sorted(vr[t][0] for t in common if pr[t][0]<=N); print('public top',N,'-> private rank median',st.median(ranks),'max',max(ranks))
# public rank band of private top10/50/100
for N in (10,50,100):
    ranks=sorted(pr[t][0] for t in common if vr[t][0]<=N); print('private top',N,'public rank median',st.median(ranks),'max',max(ranks), ranks[:N][:15])
# save merged
with open('merged.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['team_id','team','pub_rank','priv_rank','pub_score','priv_score'])
    for t in sorted(common,key=lambda t:vr[t][0]): w.writerow([t,vr[t][2],pr[t][0],vr[t][0],pr[t][1],vr[t][1]])
