"""Continuous Gaussian Bayesian-network regression with exact conditioning.

The response is the root of a statistical factorization, not a causal claim.
The TAN extension learns a maximum spanning tree of gene dependencies given Y.
All moments and all edges are estimated on the supplied training rows only.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler


def maximum_tree(weights):
    p = len(weights)
    components = list(range(p))
    def root(i):
        while components[i] != i:
            components[i] = components[components[i]]
            i = components[i]
        return i
    # Fixed index ordering gives deterministic ties without perturbing scores.
    choices = sorted((-float(weights[i,j]), i, j) for i in range(p) for j in range(i+1,p))
    edges = []
    for _, i, j in choices:
        a, b = root(i), root(j)
        if a != b:
            components[a] = b; edges.append((i,j))
            if len(edges) == p-1: break
    assert len(edges) == p-1
    return np.array(edges, dtype=int).reshape(-1,2)


def from_covariance(covariance, family, root_gene=0):
    s = np.asarray(covariance, dtype=float)
    p = len(s)-1
    residual = s[1:,1:] - np.outer(s[1:,0], s[0,1:])/s[0,0]
    corr = residual / np.sqrt(np.outer(np.diag(residual),np.diag(residual)))
    score = -.5*np.log1p(-np.minimum(corr*corr,1-1e-12))
    np.fill_diagonal(score,0.)
    edges = maximum_tree(score) if family == 'gaussian_tan' else np.empty((0,2),dtype=int)
    parent = np.full(p,-1,dtype=int)
    if family == 'gaussian_tan':
        adjacency = [[] for _ in range(p)]
        for i,j in edges: adjacency[i].append(j); adjacency[j].append(i)
        queue = [root_gene]; visited = {root_gene}
        for i in queue:
            for j in sorted(adjacency[i]):
                if j not in visited: visited.add(j); parent[j]=i; queue.append(j)
        assert len(visited)==p
    elif family != 'gaussian_naive': raise ValueError(family)
    b = np.zeros_like(s); noise = np.empty(p+1); noise[0]=s[0,0]
    for j in range(p):
        child=j+1; parents=[0] if parent[j]<0 else [0,int(parent[j])+1]
        beta=np.linalg.solve(s[np.ix_(parents,parents)],s[parents,child])
        b[child,parents]=beta
        noise[child]=s[child,child]-s[child,parents]@beta
    assert np.all(noise>1e-12)
    a=np.eye(p+1)-b
    precision=a.T@(a/noise[:,None])
    coefficient=-precision[0,1:]/precision[0,0]
    conditional_variance=1/precision[0,0]
    return {'family':family,'covariance_input':s,'structural_coefficients':b,'noise_variance':noise,
            'precision':precision,'conditional_coefficient':coefficient,'conditional_variance_z':conditional_variance,
            'gene_edges':edges,'gene_parent':parent,'conditional_mutual_information':score,'root_gene':int(root_gene)}


def fit_network(x,y,family,shrinkage):
    scaler=StandardScaler().fit(np.column_stack([y,x]))
    z=scaler.transform(np.column_stack([y,x]))
    assert np.min(z.var(axis=0))>1e-10
    covariance=z.T@z/len(z)
    covariance=(1-shrinkage)*covariance+shrinkage*np.eye(len(covariance))
    model=from_covariance(covariance,family)
    model.update(mean=scaler.mean_,scale=scaler.scale_,shrinkage=float(shrinkage),training_n=len(x))
    return model


def predict(model,x):
    z=(x-model['mean'][1:])/model['scale'][1:]
    p=model['mean'][0]+model['scale'][0]*(z@model['conditional_coefficient'])
    variance=model['conditional_variance_z']*model['scale'][0]**2
    return p, float(variance)


def verify_math():
    # Known continuous TAN with nontrivial response and gene dependencies.
    b=np.zeros((5,5));b[1:,0]=[.4,-.25,.6,-.15]
    b[2,1]=.7;b[3,2]=-.55;b[4,3]=.35
    noise=np.array([1.1,.8,.5,.7,.9]);a=np.eye(5)-b
    inv=np.linalg.inv(a);s=inv@np.diag(noise)@inv.T
    model=from_covariance(s,'gaussian_tan')
    assert {tuple(sorted(e)) for e in model['gene_edges']}=={(0,1),(1,2),(2,3)}
    reconstructed=np.linalg.inv(model['precision'])
    covariance_error=float(np.max(np.abs(reconstructed-s)))
    expected=np.linalg.solve(s[1:,1:],s[1:,0])
    coefficient_error=float(np.max(np.abs(model['conditional_coefficient']-expected)))
    variance_error=abs(model['conditional_variance_z']-(s[0,0]-s[0,1:]@expected))
    alternate=from_covariance(s,'gaussian_tan',root_gene=3)
    orientation_error=float(np.max(np.abs(alternate['precision']-model['precision'])))
    star=from_covariance(s,'gaussian_naive')
    slopes=s[1:,0]/s[0,0];var=np.diag(s)[1:]-slopes**2*s[0,0]
    posterior_var=1/(1/s[0,0]+np.sum(slopes**2/var))
    naive_error=float(np.max(np.abs(star['conditional_coefficient']-posterior_var*slopes/var)))
    assert max(covariance_error,coefficient_error,variance_error,orientation_error,naive_error)<1e-10
    # Independent MST implementation checks the total maximum tree score.
    from scipy.sparse.csgraph import minimum_spanning_tree
    scores=model['conditional_mutual_information']
    reference=minimum_spanning_tree(-scores).toarray()
    tree_error=abs(sum(scores[i,j] for i,j in model['gene_edges'])+reference.sum())
    assert tree_error<1e-10
    return {'passed':True,'known_covariance_reconstruction_error':covariance_error,
            'conditional_coefficient_error':coefficient_error,'conditional_variance_error':float(variance_error),
            'tree_root_orientation_invariance_error':orientation_error,'naive_closed_form_error':naive_error,
            'maximum_spanning_tree_weight_error':float(tree_error)}
