"""Unchanged ridge path from the previous panel comparison."""
import numpy as np
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler,OneHotEncoder

def prepare_features(features, tr, te):
    categories = None
    if features.dtype.kind in 'OUS':
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float64)
        a = encoder.fit_transform(features[tr].reshape(-1, 1))
        b = encoder.transform(features[te].reshape(-1, 1))
        categories = encoder.categories_[0].tolist()
    else:
        a, b = np.asarray(features[tr], dtype=np.float64), np.asarray(features[te], dtype=np.float64)
    scaler = StandardScaler().fit(a)
    return scaler.transform(a), scaler.transform(b), scaler, categories

def ridge_path(features, y, tr, te, alphas, retain=False):
    a, b, scaler, categories = prepare_features(features, tr, te)
    mean_y = float(y[tr].mean())
    u, s, vt = svd(a, full_matrices=False, check_finite=False, lapack_driver='gesdd')
    # One SVD for all alphas, algebraically the same standardized ridge fit.
    factors = (s[:, None] / (s[:, None]**2 + np.asarray(alphas)[None, :])) * (u.T @ (y[tr]-mean_y))[:, None]
    prediction = (b @ vt.T) @ factors + mean_y
    model = None
    if retain:
        coef = vt.T @ factors[:, 0]
        model = {'mean': scaler.mean_, 'scale': scaler.scale_, 'coef': coef,
                 'intercept_after_scaling': mean_y, 'alpha': float(alphas[0]), 'categories': categories}
        # Check exported model reproduces prediction with no hidden state.
        assert np.allclose(b @ coef + mean_y, prediction[:, 0], atol=1e-12, rtol=1e-10)
    return prediction, model
