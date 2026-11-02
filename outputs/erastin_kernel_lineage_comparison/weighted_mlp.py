"""Weighted squared-error adapter for the pinned sklearn 1.5.2 L-BFGS MLP.

Architecture, initialization, optimizer and L2 scaling are inherited unchanged.
Only the data loss and output deltas are multiplied by mean-one sample weights.
"""
import numpy as np
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network._base import DERIVATIVES
from sklearn.utils.extmath import safe_sparse_dot


class WeightedMLPRegressor(MLPRegressor):
    def fit(self,X,y,sample_weight=None):
        assert sklearn.__version__=='1.5.2'
        assert self.solver=='lbfgs' and self.activation=='tanh'
        w=np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight,dtype=float)
        if w.shape!=(len(X),) or not np.isfinite(w).all() or np.any(w<=0):
            raise ValueError('Weights must be finite, positive, and match the training rows.')
        if not np.isclose(w.sum(),len(w),atol=1e-9,rtol=0):
            raise ValueError('Weights must have mean one to preserve the prior L2 scale.')
        self.training_sample_weight_=w.copy()
        return super().fit(X,y)

    def _backprop(self,X,y,activations,deltas,coef_grads,intercept_grads):
        n_samples=X.shape[0]
        activations=self._forward_pass(activations)
        assert y.shape==(n_samples,1) and self.out_activation_=='identity'
        w=self.training_sample_weight_
        residual=activations[-1]-y
        loss=.5*np.mean(residual**2*w[:,None])
        values=0
        for coef in self.coefs_:
            flat=coef.ravel();values+=np.dot(flat,flat)
        loss+=(.5*self.alpha)*values/n_samples
        last=self.n_layers_-2
        deltas[last]=residual*w[:,None]
        self._compute_loss_grad(last,n_samples,activations,deltas,coef_grads,intercept_grads)
        derivative=DERIVATIVES[self.activation]
        for i in range(self.n_layers_-2,0,-1):
            deltas[i-1]=safe_sparse_dot(deltas[i],self.coefs_[i].T)
            derivative(activations[i],deltas[i-1])
            self._compute_loss_grad(i-1,n_samples,activations,deltas,coef_grads,intercept_grads)
        return loss,coef_grads,intercept_grads


def inverse_lineage_weights(lineages):
    _,inverse,counts=np.unique(lineages,return_inverse=True,return_counts=True)
    w=len(lineages)/(len(counts)*counts[inverse].astype(float))
    assert np.isclose(w.sum(),len(w),atol=1e-9,rtol=0)
    np.testing.assert_allclose(np.bincount(inverse,weights=w),len(w)/len(counts),rtol=0,atol=1e-10)
    return w
