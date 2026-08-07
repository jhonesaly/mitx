import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

print("=== 2. K-means na toy_data ===")
for K in [1, 2, 3, 4]:
    best_cost = float('inf')
    for seed in range(5):
        mixture, post = common.init(X, K, seed)
        mixture, post, cost = kmeans.run(X, mixture, post)
        if cost < best_cost:
            best_cost = cost
    print(f"Cost|K={K} = {best_cost:.4f}")

print("\n=== 4 & 5. Naive EM e BIC na toy_data ===")
best_bic_score = float('-inf')
best_K_bic = None

for K in [1, 2, 3, 4]:
    best_ll = float('-inf')
    best_mixture = None
    best_post = None
    for seed in range(5):
        mixture, post = common.init(X, K, seed)
        mixture, post, ll = naive_em.run(X, mixture, post)
        if ll > best_ll:
            best_ll = ll
            best_mixture = mixture
            best_post = post
    bic_score = common.bic(X, best_mixture, best_ll)
    if bic_score > best_bic_score:
        best_bic_score = bic_score
        best_K_bic = K
    print(f"Log-likelihood|K={K} = {best_ll:.4f} (BIC = {bic_score:.4f})")

print(f"\nMelhor K pelo BIC: {best_K_bic} com BIC = {best_bic_score:.4f}")

print("\n=== 8. Netflix Data ===")
X_netflix = np.loadtxt("netflix_incomplete.txt")
X_netflix_gold = np.loadtxt("netflix_complete.txt")

for K in [1, 12]:
    best_ll = float('-inf')
    best_mixture = None
    for seed in range(5):
        mixture, post = common.init(X_netflix, K, seed)
        mixture, post, ll = em.run(X_netflix, mixture, post)
        if ll > best_ll:
            best_ll = ll
            best_mixture = mixture
    print(f"Log-likelihood|K={K} = {best_ll:.4f}")

    if K == 12:
        X_pred = em.fill_matrix(X_netflix, best_mixture)
        rmse_val = common.rmse(X_netflix_gold, X_pred)
        print(f"RMSE (K=12) = {rmse_val:.4f}")

