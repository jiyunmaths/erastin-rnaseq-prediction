"""Training-only moderate weights, normalized to mean one.

Capped weights use bounded normalization: min(scale / lineage_count, 5).
The scale is solved jointly with clipping so the FINAL weights obey the cap.
"""
import numpy as np
SCHEMES=['inverse_sqrt','capped_inverse_5']
CAP=5.0

def weights_for(lineages,scheme):
    labels,inverse,counts=np.unique(lineages,return_inverse=True,return_counts=True)
    if not len(labels):raise ValueError('The fitting subset must not be empty.')
    n=len(lineages)
    if scheme=='unweighted':per=np.ones(len(labels))
    elif scheme=='inverse_frequency':per=n/(len(labels)*counts.astype(float))
    elif scheme=='inverse_sqrt':
        per=1/np.sqrt(counts.astype(float));per*=n/np.dot(counts,per)
    elif scheme=='capped_inverse_5':
        # For free groups, n_l*w_l is constant. Capped groups stay at CAP.
        free=np.ones(len(labels),bool);per=np.empty(len(labels))
        while True:
            scale=(n-CAP*counts[~free].sum())/free.sum()
            candidate=scale/counts[free]
            over=candidate>CAP
            if not np.any(over):per[free]=candidate;per[~free]=CAP;break
            indices=np.flatnonzero(free);free[indices[over]]=False
            assert free.any()
        assert np.max(per)<=CAP
    else:raise ValueError(scheme)
    w=per[inverse]
    assert np.isfinite(w).all() and (w>0).all()
    np.testing.assert_allclose(w.sum(),n,rtol=0,atol=1e-9)
    return w

def weight_audit(lineages,scheme):
    w=weights_for(lineages,scheme);labels,counts=np.unique(lineages,return_counts=True)
    return dict(scheme=scheme,n=len(w),represented_lineages=len(labels),weight_min=float(w.min()),weight_max=float(w.max()),
        weight_sum=float(w.sum()),effective_sample_size=float(w.sum()**2/np.dot(w,w)),
        capped_cells=int(np.sum(w==CAP)) if scheme=='capped_inverse_5' else 0,
        groups=[dict(lineage=str(label),n=int(n),weight_per_cell=float(w[lineages==label][0]),total_weight=float(w[lineages==label].sum())) for label,n in zip(labels,counts)])
