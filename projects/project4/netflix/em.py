"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    f = np.zeros((n, K))

    for u in range(n):
        cu = np.where(X[u] != 0)[0]
        c_len = len(cu)
        for j in range(K):
            sse = np.sum((X[u, cu] - mixture.mu[j, cu]) ** 2)
            log_p = np.log(mixture.p[j] + 1e-16)
            log_g = -0.5 * c_len * np.log(2 * np.pi * mixture.var[j]) - 0.5 * sse / mixture.var[j]
            f[u, j] = log_p + log_g

    ll_u = logsumexp(f, axis=1)
    LL = float(np.sum(ll_u))
    post = np.exp(f - ll_u[:, None])
    return post, LL




def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape
    delta = (X != 0).astype(float)
    c_len = delta.sum(axis=1)

    p = post.sum(axis=0) / n
    mu = np.zeros((K, d))
    var = np.zeros(K)

    for j in range(K):
        denom_mu = post[:, j] @ delta
        num_mu = post[:, j] @ X
        mu[j, :] = np.where(denom_mu >= 1.0, num_mu / denom_mu, mixture.mu[j])

        diff_sq = ((X - mu[j])**2) * delta
        sse_u = diff_sq.sum(axis=1)
        sse_total = post[:, j] @ sse_u
        denom_var = post[:, j] @ c_len

        var_j = sse_total / denom_var
        if var_j < min_variance:
            var_j = min_variance
        var[j] = var_j

    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_LL = None
    LL = None
    while prev_LL is None or (LL - prev_LL > 1e-6 * abs(LL)):
        prev_LL = LL
        post, LL = estep(X, mixture)
        mixture = mstep(X, post, mixture)
    return mixture, post, LL


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    post, _ = estep(X, mixture)
    X_pred = X.copy()
    missing = (X == 0)
    X_pred[missing] = (post @ mixture.mu)[missing]
    return X_pred

