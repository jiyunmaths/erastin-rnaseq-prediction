"""Response-blind lineage stratification with intact identity groups."""
import numpy as np


def lineage_groups(lineage,groups):
    result={}
    for label in np.unique(lineage):
        ids=np.unique(groups[lineage==label])
        result[str(label)]=[(str(g),np.flatnonzero(groups==g)) for g in ids]
        for _,rows in result[str(label)]:assert np.all(lineage[rows]==label)
    return result


def sample_group_subset(sizes,target,rng):
    """Uniform draw among group subsets with the nearest feasible sample count.

    Integer dynamic-program counts avoid a preference for single-line groups.
    The subset and its complement must both contain at least one group.
    """
    sizes=[int(v) for v in sizes];total=sum(sizes);m=len(sizes);assert m>=2
    dp=[[0]*(total+1) for _ in range(m+1)];dp[0][0]=1
    for i,size in enumerate(sizes,1):
        for t in range(total+1):dp[i][t]=dp[i-1][t]+(dp[i-1][t-size] if t>=size else 0)
    feasible=np.array([t for t in range(1,total) if dp[m][t]>0]);distance=np.abs(feasible-target)
    t=int(rng.choice(feasible[distance==distance.min()]));selected=[]
    for i in range(m,0,-1):
        size=sizes[i-1];take=dp[i-1][t-size] if t>=size else 0
        if take and rng.random()<take/dp[i][t]:selected.append(i-1);t-=size
    assert t==0 and 0<len(selected)<m
    return np.array(sorted(selected),dtype=int)


def outer_split(lineage,groups,seed,test_fraction=.2):
    rng=np.random.default_rng(int(seed));test=np.zeros(len(lineage),bool)
    for label,entries in lineage_groups(lineage,groups).items():
        sizes=[len(ix) for _,ix in entries];n=sum(sizes);target=max(1,min(n-1,int(np.floor(test_fraction*n+.5))))
        chosen=sample_group_subset(sizes,target,rng)
        for i in chosen:test[entries[i][1]]=True
    tr=np.flatnonzero(~test);te=np.flatnonzero(test)
    assert set(groups[tr]).isdisjoint(groups[te])
    assert set(lineage[tr])==set(lineage[te])==set(lineage)
    return test


def inner_folds(lineage,groups,outer_test,seed,k=5):
    """Five group-stratified folds, retaining singleton training lineages.

    -1: outer test, -2: always inner training (singleton lineage group),
    0..k-1: ordinary inner validation assignment.
    """
    rng=np.random.default_rng(int(seed));result=np.full(len(lineage),-1,dtype=np.int8)
    tr=np.flatnonzero(~outer_test);counts=np.zeros(k,int)
    for label,entries in lineage_groups(lineage,groups).items():
        entries=[(g,ix) for g,ix in entries if not outer_test[ix[0]]]
        if len(entries)==1:result[entries[0][1]]=-2;continue
        order=rng.permutation(len(entries)).tolist();order.sort(key=lambda j:-len(entries[j][1]));lc=np.zeros(k,int)
        for j in order:
            options=np.flatnonzero(lc==lc.min());options=options[counts[options]==counts[options].min()];fold=int(rng.choice(options))
            ix=entries[j][1];result[ix]=fold;lc[fold]+=len(ix);counts[fold]+=len(ix)
        assert np.sum(lc>0)>=2
    assert np.all(result[outer_test]==-1) and np.all(result[~outer_test]!=-1)
    for fold in range(k):
        a,b=inner_rows(result,fold)
        assert set(a)|set(b)==set(tr) and set(groups[a]).isdisjoint(groups[b])
        assert set(lineage[a])==set(lineage)
    return result


def inner_rows(assignment,fold):
    return np.flatnonzero((assignment==-2)|((assignment>=0)&(assignment!=fold))),np.flatnonzero(assignment==fold)
