import numpy as np
import em
import common

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 4
n, d = X.shape
seed = 0

mixture, post = common.init(X, K, seed)
mixture, post, LL = em.run(X, mixture, post)
print(f"Log-Likelihood: {LL:.4f}")

X_pred = em.fill_matrix(X, mixture)
rmse_val = common.rmse(X_gold, X_pred)
print(f"RMSE: {rmse_val:.4f}")

