"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))

    for j in range(K):
        diff = X - mixture.mu[j]
        sse = (diff**2).sum(axis=1)
        coeff = (2 * np.pi * mixture.var[j]) ** (-d / 2.0)
        post[:, j] = mixture.p[j] * coeff * np.exp(-0.5 * sse / mixture.var[j])

    likelihoods = post.sum(axis=1)
    LL = float(np.sum(np.log(likelihoods)))
    post = post / likelihoods[:, None]
    return post, LL


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape
    n_hat = post.sum(axis=0)
    p = n_hat / n
    mu = np.zeros((K, d))
    var = np.zeros(K)
    for j in range(K):
        mu[j, :] = post[:, j] @ X / n_hat[j]
        sse = ((X - mu[j])**2).sum(axis=1) @ post[:, j]
        var[j] = sse / (d * n_hat[j])
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
        mixture = mstep(X, post)
    return mixture, post, LL

