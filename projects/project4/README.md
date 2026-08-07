# 1. Introduction

Your task is to build a mixture model for collaborative filtering. You are given a data matrix containing movie ratings made by users where the matrix is extracted from a much larger Netflix database. Any particular user has rated only a small fraction of the movies so the data matrix is only partially filled. The goal is to predict all the remaining entries of the matrix.

You will use mixtures of Gaussians to solve this problem. The model assumes that each user's rating profile is a sample from a mixture model. In other words, we have $K$ possible types of users and, in the context of each user, we must sample a user type and then the rating profile from the Gaussian distribution associated with the type. We will use the Expectation Maximization (EM) algorithm to estimate such a mixture from a partially observed rating matrix. The EM algorithm proceeds by iteratively assigning (softly) users to types (E-step) and subsequently re-estimating the Gaussians associated with each type (M-step). Once we have the mixture, we can use it to predict values for all the missing entries in the data matrix.

---

Setup:

As with the last project, please use Python's **NumPy** numerical library for handling arrays and array operations; use **matplotlib** for producing figures and plots.

1. *Note on software:* For all the projects, we will use python 3.11 augmented with the **NumPy** numerical toolbox, the **matplotlib** plotting toolbox. In this project, we will also use the `typing` library, which is already included in the standard library (no need to install anything).

2. Download `netflix.tar.gz` and untar it into a working directory. The archive contains the following python files:

- `kmeans.py`: where we have implemented a baseline using the K-means algorithm
- `naive_em.py`: where you will implement a first version of the EM algorithm (tabs 3-4)
- `em.py`: where you will build a mixture model for collaborative filtering (tabs 7-8)
- `common.py`: where you will implement the common functions for all models (tab 5)
- `main.py`: where you will write code to answer the questions for this project
- `test.py`: where you will write code to test your implementation of EM for a given test case

Additionally, you are provided with the following data files:

- `toy_data.txt`: a 2D dataset that you will work with in tabs 2-5
- `netflix_incomplete.txt`: the netflix dataset with missing entries to be completed
- `netflix_complete.txt`: the netflix dataset with missing entries completed
- `test_incomplete.txt`: a test dataset to test your code against our implementation
- `test_complete.txt`: a test dataset to test your code against our implementation
- `test_solutions.txt`: a test dataset to test your code against our implementation

Tip: Throughout the whole online grading system, you can assume the NumPy python library is already imported as `np`. In some problems you will also have access to other functions you've already implemented. Look out for the "Available Functions" Tip before the codebox, as you did in the previous projects.

This project will unfold both on MITx and on your local machine. However, we encourage you to first implement the functions locally and run them to validate basic functionality. Think of the online graders as a submission box to submit your code when it is ready. You should not have to use the online graders to debug your code.

---

# 2. K-means

Project due Aug 12, 2026 08:59 -03

## K-means

0.0/1.0 point (graded)

For this part of the project you will compare clustering obtained via K-means to the (soft) clustering induced by EM. In order to do so, our K-means algorithm will differ a bit from the one you learned. Here, the means are estimated exactly as before but the algorithm returns additional information. More specifically, we use the resulting clusters of points to estimate a Gaussian model for each cluster. Thus, our K-means algorithm actually returns a mixture model where the means of the component Gaussians are the $K$ centroids computed by the K-means algorithm. This is to make it such that we can now directly plot and compare solutions returned by the two algorithms as if they were both estimating mixtures.

Read a 2D toy dataset using `X = np.loadtxt('toy_data.txt')`. Your task is to run the K-means algorithm on this data using the implementation we have provided in `kmeans.py`. Initialize K-means using `common.init(X, K, seed)`, where $K$ is the number of clusters and `seed` is the random seed used to randomly initialize the parameters.

Note that `init(X, K)` returns a K-component mixture model with means, variances and mixing proportions. The K-means algorithm will only care about the means, however, and returns a mixture that is retrofitted based on the K-means solution.

Try $K \in \{1, 2, 3, 4\}$ on this data, plotting each solution using our `common.plot` function. Since the initialization is random, please use seeds $0, 1, 2, 3, 4$ and select the one that minimizes the total cost. Save the associated plots (best solution for each $K$). The code for this task can be written in `main.py`.

Report the lowest cost for each $K$:

- $\text{Cost}\big|_{K=1} =$ [          ]
- $\text{Cost}\big|_{K=2} =$ [          ]
- $\text{Cost}\big|_{K=3} =$ [          ]
- $\text{Cost}\big|_{K=4} =$ [          ]

---

# 3. Expectation Maximization (EM)

Recall the Gaussian mixture model presented in class:

$$P(x \mid \theta) = \sum_{j=1}^{K} \pi_j \mathcal{N}(x; \mu^{(j)}, \sigma_j^2 I)$$

where $\theta$ denotes all the parameters in the mixture (means $\mu^{(j)}$, mixing proportions $\pi_j$, and variances $\sigma_j^2$). The goal of the EM algorithm is to estimate these unknown parameters by maximizing the log-likelihood of the observed data $x^{(1)}, \ldots, x^{(n)}$. Starting with some initial guess of the unknown parameters, the algorithm iterates between E- and M-steps. The E-Step softly assigns each data point $x^{(i)}$ to mixture components. The M-step takes these soft-assignments as given and finds a new setting of the parameters by maximizing the log-likelihood of the weighted dataset (expected complete log-likelihood).

Implement the EM algorithm for the Gaussian mixture model described above. To this end, complete the functions `estep`, `mstep` and `run` in `naive_em.py`. In our notation:

- `x`: an $(n, d)$ Numpy array of $n$ data points, each with $d$ features
- `K`: number of mixture components
- `mu`: $(K, d)$ Numpy array where the $j^{\text{th}}$ row is the mean vector $\mu^{(j)}$
- `p`: $(K,)$ Numpy array of mixing proportions $\pi_j, j = 1, \ldots, K$
- `var`: $(K,)$ Numpy array of variances $\sigma_j^2, j = 1, \ldots, K$

The convergence criteria that you should use is that the improvement in the log-likelihood is less than or equal to $10^{-6}$ multiplied by the absolute value of the new log-likelihood. In slightly more algebraic notation:

$$\text{new log-likelihood} - \text{old log-likelihood} \le 10^{-6} \cdot \lvert \text{new log-likelihood} \rvert$$

Your code will output updated versions of a `GaussianMixture` (with means `mu`, variances `var` and mixing proportions `p`) as defined in `common.py` as well as an $(n, K)$ Numpy array `post`, where $\text{post}[i, j]$ is the posterior probability $p(j \mid x^{(i)})$, and `LL` which is the log-likelihood of the weighted dataset.

Here are a few points to check to make sure that your implementation is indeed correct:

1. Make sure that all your functions return objects with the right dimension.
2. EM should monotonically increase the log-likelihood of the data. Initialize and run the EM algorithm on the toy dataset as you did earlier with K-means. You should check that the LL values that the algorithm returns after each run are indeed always monotonically increasing (non-decreasing).
3. Using $K = 3$ and a seed of $0$, on the toy dataset, you should get a log likelihood of -1388.0818 after first iteration.
4. As a runtime guideline, in your testing on the toy dataset, calls of `run` using the values of $K$ that we are testing should run in on the order of seconds (i.e. if each call isn't fairly quick, that may be an indication that something is wrong).
5. Try plotting the solutions obtained with your EM implementation. Do they make sense?

---

## Implementing E-step

0.0/1.0 point (graded)

Write a function `estep` that performs the E-step of the EM algorithm.

Available Functions: You have access to the NumPy python library as `np`, to the `GaussianMixture` class and to typing annotation `typing.Tuple` as `Tuple`.

```python
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
    raise NotImplementedError
```

---

## Implementing M-step

0.0/1.0 point (graded)

Write a function `mstep` that performs the M-step of the EM algorithm.

Available Functions: You have access to the NumPy python library as `np`, to the `GaussianMixture` class and to typing annotation `typing.Tuple` as `Tuple`.

```python
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
    raise NotImplementedError
```

---

## Implementing run

0.0/1.0 point (graded)

Write a function `run` that runs the EM algorithm. The convergence criterion you should use is described above.

Available Functions: You have access to the NumPy python library as `np`, to the `GaussianMixture` class and to typing annotation `typing.Tuple` as `Tuple`. You also have access to the `estep` and `mstep` functions you have just implemented.

```python
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
    raise NotImplementedError
```

---

# 4. Comparing K-means and EM

Project due Aug 12, 2026 08:59 -03

Generate analogous plots to K-means using your EM implementation. Note that the EM algorithm can also get stuck in a locally optimal solution. For each value of $K$, please run the EM algorithm with seeds $0, 1, 2, 3, 4$ and select the solution that achieves the highest log-likelihood. Compare the K-means and mixture solutions for $K \in \{1, 2, 3, 4\}$. Ask yourself when, how, and why they differ.

---

## Reporting log likelihood values

0.0/1.0 point (graded)

Report the maximum likelihood for each $K$ using seeds $0, 1, 2, 3, 4$:

- $\text{Log-likelihood}\big|_{K=1} =$ [          ]
- $\text{Log-likelihood}\big|_{K=2} =$ [          ]
- $\text{Log-likelihood}\big|_{K=3} =$ [          ]
- $\text{Log-likelihood}\big|_{K=4} =$ [          ]

---

## Analysing plots

0.0/1.0 point (graded)

Which of the following sentences are true? (Check all that apply)

**Note:** This question is the multichoice version of the free-text question: “Compare the K-means and mixture solutions for $K \in \{1, 2, 3, 4\}$. Ask yourself when, how, and why they differ.”
In order to answer this, you should look at the plots side by side, either by adapting the code to plot them together or by simply saving the plots as you go. For each value of $K$, ask yourself whether the plots you see are similar or different. If they are different, why are they different?

*Hint:* What are we optimizing for in each case? What are we plotting? In the case of K-means, we have clusters, with EM, can these really be called clusters? What is EM optimizing for?

Now, write a descriptive paragraph of your observations as if it were part of a report for this project and you were going hand this back for us to grade. Try matching your paragraph with the options provided. If they don't match, then we wouldn't have given you full credit for this question.

**Note:** We have increased the attempt by 1.

- [ ] In the case $K=1$, the mixture parameters and point assignments are the same for both methods
- [ ] In the case $K=2$, both methods have similar parameters and point assignments
- [ ] In the case $K=3$, the k-means solution accounts for point density better than EM
- [ ] In the case $K=4$, the k-means solution equally spaces the clusters to minimize distortion cost

---

# 5. Bayesian Information Criterion

Project due Aug 12, 2026 08:59 -03

So far we have simply set the number of mixture components $K$ but this is also a parameter that we must estimate from data. How does the log-likelihood of the data vary as a function of $K$ assuming we avoid locally optimal solutions?

To compensate, we need a selection criterion that penalizes the number of parameters used in the model. The Bayesian information criterion (BIC) is a criterion for model selection. It captures the tradeoff between the log-likelihood of the data, and the number of parameters that the model uses. The BIC of a model $M$ is defined as:

$$\text{BIC}(M) = l - \frac{1}{2} p \log n$$

where $l$ is the log-likelihood of the data under the current model (highest log-likelihood we can achieve by adjusting the parameters in the model), $p$ is the number of free parameters, and $n$ is the number of data points. This score rewards a larger log-likelihood, but penalizes the number of parameters used to train the model. In a situation where we wish to select models, we want a model with the highest BIC.

---

## Implementing the Bayesian Information Criterion

0.0/1.0 point (graded)

Fill in the missing Bayesian Information Criterion (BIC) calculation (`bic` function) in `common.py`.

Available Functions: You have access to the NumPy python library as `np`, to the `GaussianMixture` class and to typing annotation `typing.Tuple` as `Tuple`.

```python
def bic(X: np.ndarray, mixture: GaussianMixture,
        log_likelihood: float) -> float:
    """Computes the Bayesian Information Criterion for a
    mixture of gaussians

    Args:
        X: (n, d) array holding the data
        mixture: a mixture of spherical gaussian
        log_likelihood: the log-likelihood of the data

    Returns:
        float: the BIC for this mixture
    """
    raise NotImplementedError
```

---

## Picking the best K

0.0/1.0 point (graded)

Find the best $K$ from $\{1, 2, 3, 4\}$ on the toy dataset. This will be the $K$ that produces the optimal BIC score. Report the best $K$ and the corresponding BIC score. Measure the BIC on EM models, only. Does the criterion select the correct number of clusters for the toy data?

- $\text{Best K} =$ [          ]
- $\text{Best BIC} =$ [          ]

---

# 6. Mixture models for matrix completion

We can now extend our Gaussian mixture model to predict actual movie ratings. Let $X$ again denote the $(n, d)$ data matrix. The rows of this matrix correspond to users and columns specify movies so that `X[u, i]` gives the rating value of user $u$ for movie $i$ (if available). Both $n$ and $d$ are typically quite large. The ratings range from one to five stars and are mapped to integers $\{1, 2, 3, 4, 5\}$. We will set `X[u, i] = 0` whenever the entry is missing.

In a realistic setting, most of the entries of $X$ are missing. For this reason, we define $C_u$ as the set of movies (column indexes) that user $u$ has rated and $H_u$ as its complement (the set of remaining unwatched/unrated movies we wish to predict ratings for). We use $|C_u|$ to denote the number of observed rating values from user $u$. From the point of view of our mixture model, each user $u$ is an example $x^{(u)} = X[u, :]$. But since most of the coordinates of $x^{(u)}$ are missing, we need to focus the model during training on just the observed portion. To this end, we use $x^{(u)}_{C_u} = \{x^{(u)}_i : i \in C_u\}$ as the vector of only observed ratings. If columns are indexed as $\{0, \ldots, d - 1\}$, then a user $u$ with a rating vector $x^{(u)} = (5, 4, 0, 0, 2)$, where zeros indicate missing values, has $C_u = \{0, 1, 4\}$, $H_u = \{2, 3\}$, and $x^{(u)}_{C_u} = (5, 4, 2)$.

In this part, we will extend our mixture model in two key ways:

- First, we are going to estimate a mixture model based on partially observed ratings. See notes below.
- Second, since we will be dealing with a large, high-dimensional data set, we will need to be more mindful of numerical underflow issues. To this end, you should perform most of your computations in the log domain. Remember, $\log(a \cdot b) = \log(a) + \log(b)$. This can be useful to remember when $a$ and $b$ are very small – in these cases, addition should result in fewer numerical underflow issues than multiplication.

  An additional numerical optimization trick that you will find useful is the LogSumExp trick. Assume that we wish to evaluate $y = \log(\exp(x_1) + \ldots + \exp(x_n))$. We define $x^* = \max\{x_1, \ldots, x_n\}$. Then, $y = x^* + \log(\exp(x_1 - x^*) + \ldots + \exp(x_n - x^*))$. This is just another trick to help ensure numerical stability.

---

## Marginalizing over unobserved coordinates

If $x^{(u)}$ were a complete rating vector, the mixture model from Part 1 would simply say that $P(x^{(u)} \mid \theta) = \sum_{j=1}^{K} \pi_j \mathcal{N}(x^{(u)}; \mu^{(j)}, \sigma_j^2 I)$. In the presence of missing values, we must use the marginal probability $P(x^{(u)}_{C_u} \mid \theta)$ that is over only the observed values. This marginal corresponds to integrating the mixture density $P(x^{(u)} \mid \theta)$ over all the unobserved coordinate values. In our case, this marginal can be computed as follows.

The mixture model for a complete rating vector is written as:

$$P(x^{(u)} \mid \theta) = \sum_{j=1}^{K} \pi_j \mathcal{N}(x^{(u)}; \mu^{(j)}, \sigma_j^2 I)$$

We can decompose the multivariate spherical Gaussian as a product of univariate Gaussians (since there is no covariance between coordinates):

$$P(x^{(u)} \mid \theta) = \sum_{j=1}^{K} \pi_j \prod_{i} \mathcal{N}(x^{(u)}_i; \mu^{(j)}_i, \sigma_j^2)$$

$$= \sum_{j=1}^{K} \pi_j \prod_{m \in C_u} \mathcal{N}(x^{(u)}_m; \mu^{(j)}_m, \sigma_j^2) \prod_{m' \in H_u} \mathcal{N}(x^{(u)}_{m'}; \mu^{(j)}_{m'}, \sigma_j^2)$$

For $m' \in H_u$, we can marginalize over all of the unobserved values to get:

$$\int \mathcal{N}(x^{(u)}_{m'}; \mu^{(j)}_{m'}, \sigma_j^2) \, dx^{(u)}_{m'} = 1$$

Thus, our mixture density can be written as:

$$P(x^{(u)}_{C_u} \mid \theta) = \sum_{j=1}^{K} \pi_j \mathcal{N}(x^{(u)}_{C_u}; \mu^{(j)}_{C_u}, \sigma_j^2 I_{|C_u| \times |C_u|})$$

where $I_{|C_u| \times |C_u|}$ is the identity matrix in $|C_u|$ dimensions.

---

# 7. Implementing EM for matrix completion

Project due Aug 12, 2026 08:59 -03

We need to update our EM algorithm a bit to deal with the fact that the observations are no longer complete vectors. We use Bayes' rule to find an updated expression for the posterior probability $p(j \mid u) = P(y = j \mid x^{(u)}_{C_u})$:

$$p(j \mid u) = \frac{p(u \mid j) \cdot p(j)}{p(u)} = \frac{p(u \mid j) \cdot p(j)}{\sum_{j=1}^K p(u \mid j) \cdot p(j)} = \frac{\pi_j \mathcal{N}(x^{(u)}_{C_u}; \mu^{(j)}_{C_u}, \sigma_j^2 I_{|C_u| \times |C_u|})}{\sum_{j=1}^K \pi_j \mathcal{N}(x^{(u)}_{C_u}; \mu^{(j)}_{C_u}, \sigma_j^2 I_{|C_u| \times |C_u|})}$$

This is the soft assignment of cluster $j$ to data point $u$.

To minimize numerical instability, you will be re-implementing the E-step in the log-domain, so you should calculate the values for the log of the posterior probability, $\ell(j \mid u) = \log(p(j \mid u))$ (though the actual output of your E-step should include the non-log posterior).

Let $f(u, i) = \log(\pi_i) + \log \left( \mathcal{N} \left( x^{(u)}_{C_u}; \mu^{(i)}_{C_u}, \sigma_i^2 I_{|C_u| \times |C_u|} \right) \right)$. Then, in terms of $f$, the log posterior is:

$$\ell(j \mid u) = \log(p(j \mid u)) = \log \left( \frac{\pi_j \mathcal{N}(x^{(u)}_{C_u}; \mu^{(j)}_{C_u}, \sigma_j^2 I_{|C_u| \times |C_u|})}{\sum_{j=1}^K \pi_j \mathcal{N}(x^{(u)}_{C_u}; \mu^{(j)}_{C_u}, \sigma_j^2 I_{|C_u| \times |C_u|})} \right)$$

$$= \log \left( \pi_j \mathcal{N}(x^{(u)}_{C_u}; \mu^{(j)}_{C_u}, \sigma_j^2 I_{|C_u| \times |C_u|}) \right) - \log \left( \sum_{j=1}^K \pi_j \mathcal{N}(x^{(u)}_{C_u}; \mu^{(j)}_{C_u}, \sigma_j^2 I_{|C_u| \times |C_u|}) \right)$$

$$= \log(\pi_j) + \log \left( \mathcal{N}(x^{(u)}_{C_u}; \mu^{(j)}_{C_u}, \sigma_j^2 I_{|C_u| \times |C_u|}) \right) - \log \left( \sum_{j=1}^K \exp \left( \log \left( \pi_j \mathcal{N}(x^{(u)}_{C_u}; \mu^{(j)}_{C_u}, \sigma_j^2 I_{|C_u| \times |C_u|}) \right) \right) \right)$$

$$= f(u, j) - \log \left( \sum_{j=1}^K \exp(f(u, j)) \right)$$

Once we have evaluated $p(j \mid u)$ in the E-step, we can proceed to the M-step. We wish to find the parameters $\pi$, $\mu$, and $\sigma$ that maximize $\ell(X; \theta)$.

First, note that, by decomposing the multivariate spherical Gaussians into univariate spherical Gaussians as before, we can write, if $l \in C_u$:

$$\frac{\partial}{\partial \mu_l^{(k)}} \mathcal{N}(x^{(u)}_{C_u} \mid \mu^{(k)}_{C_u}, \sigma_k^2 I_{|C_u| \times |C_u|}) = \mathcal{N}(\dots) \frac{\frac{\partial}{\partial \mu_l^{(k)}} \left( \frac{1}{\sqrt{2\pi}\sigma_k} \exp \left( -\frac{1}{2\sigma_k^2} (x^{(u)}_l - \mu^{(k)}_l)^2 \right) \right)}{\left( \frac{1}{\sqrt{2\pi}\sigma_k} \exp \left( -\frac{1}{2\sigma_k^2} (x^{(u)}_l - \mu^{(k)}_l)^2 \right) \right)}$$

$$= \mathcal{N}(\dots) \frac{x^{(u)}_l - \mu^{(k)}_l}{\sigma_k^2}$$

where $\mathcal{N}(\dots) = \mathcal{N}(x^{(u)}_{C_u} \mid \mu^{(k)}_{C_u}, \sigma_k^2 I_{|C_u| \times |C_u|})$.

If $l \notin C_u$, that derivative is $0$. To cover both cases, we can write:

$$\frac{\partial}{\partial \mu_l^{(k)}} \mathcal{N}(x^{(u)}_{C_u} \mid \mu^{(k)}_{C_u}, \sigma_k^2 I_{|C_u| \times |C_u|}) = \mathcal{N}(x^{(u)}_{C_u} \mid \mu^{(k)}_{C_u}, \sigma_k^2 I_{|C_u| \times |C_u|}) \delta(l, C_u) \frac{x^{(u)}_l - \mu^{(k)}_l}{\sigma_k^2}$$

where $\delta(i, C_u)$ is an indicator function: $1$ if $i \in C_u$ and zero otherwise.

Following the EM algorithm's approach of maximizing a proxy likelihood function $\hat{\ell}(X; \theta)$ during the M step, consider the following function:

$$\hat{\ell}(X; \theta) = \sum_{u=1}^n \sum_{j=1}^K p(j \mid u) \log \left( \frac{p(x^{(u)} \text{ generated by cluster } j; \theta)}{p(j \mid u)} \right)$$

$$= \sum_{u=1}^n \sum_{j=1}^K p(j \mid u) \log \left( \frac{\pi_j \mathcal{N}(x^{(u)}_{C_u} \mid \mu^{(j)}_{C_u}, \sigma_j^2 I_{|C_u| \times |C_u|})}{p(j \mid u)} \right)$$

where $p(x^{(u)} \text{ generated by cluster } j; \theta)$ is the likelihood of $x^{(u)}$ generated by cluster $j$ and the parameter set is $\theta$. The values $p(j \mid u)$ are the ones as we computed in the E step and they are constants for the M step.

We now take the derivative of $\hat{\ell}(X; \theta)$ with respect to $\mu_l^{(k)}$ to find the optimal value of $\mu_l^{(k)}$ that maximizes $\hat{\ell}(X; \theta)$:

$$\frac{\partial \hat{\ell}(X; \theta)}{\partial \mu_l^{(k)}} = -\frac{\partial}{\partial \mu_l^{(k)}} \left[ \sum_{u=1}^n \sum_{j=1}^K p(j \mid u) \cdot \frac{1}{2} \cdot \frac{\|x^{(u)}_{C_u} - \mu^{(j)}_{C_u}\|^2}{\sigma_j^2} \right]$$

$$= \sum_{u=1}^n p(k \mid u) \delta(l, C_u) \frac{x^{(u)}_l - \mu^{(k)}_l}{\sigma_k^2}$$

where $\delta(i, C_u) = 1$ if $i \in C_u$ and $\delta(i, C_u) = 0$ if $i \notin C_u$.

Setting the partial derivative equal to zero, we obtain that:

$$\widehat{\mu_l^{(k)}} = \frac{\sum_{u=1}^n p(k \mid u) \delta(l, C_u) x^{(u)}_l}{\sum_{u=1}^n p(k \mid u) \delta(l, C_u)}$$

We leave it as an exercise to the reader to obtain the estimates of $\sigma_k^2$ and $\pi_k$ for $k = 1, \dots, K$. Verify that:

$$\widehat{\sigma_k^2} = \frac{1}{\sum_{u=1}^n |C_u| p(k \mid u)} \sum_{u=1}^n p(k \mid u) \|x^{(u)}_{C_u} - \widehat{\mu^{(k)}_{C_u}}\|^2$$

$$\widehat{\pi_k} = \frac{1}{n} \sum_{u=1}^n p(k \mid u)$$

Implementation guidelines:

- You may find LogSumExp useful. But remember that your M-step should return the new $P = \hat{\pi}$, not the log of $\hat{\pi}$.
- The following will not affect the update equation above, but will affect your implementation: since we are dealing with incomplete data, we might have a case where most of the points in cluster $j$ are missing the $i$-th coordinate. If we are not careful, the value of this coordinate in the mean will be determined by a small number of points, which leads to erratic results. Instead, we should only update the mean when $\sum_{u=1}^n p(j \mid u) \delta(i, C_u) \ge 1$. Since $p(j \mid u)$ is a soft probability assignment, this corresponds to the case when at least one full point supports the mean.
- To also avoid the variances of clusters going to zero due to a small number of points being assigned to them, in the M-step you will need to implement a minimum variance for your clusters. We recommend a value of 0.25, though you are free to experiment with it if you wish. Note that this issue, as well as the thresholded mean update in the point above, are better dealt with through regularization; however, to keep things simple, we do not do regularization here.
- To debug your EM implementation, you may use the data files `test_incomplete.txt` and `test_complete.txt`. Compare your results to ours from `test_solutions.txt`.

---

## Implementing E-step (2)

0.0/1.0 point (graded)

In `em.py`, fill in the `estep` function so that it works with partially observed vectors where missing values are indicated with zeros, and perform the computations in the log domain to help with numerical stability.

Available Functions: You have access to the NumPy python library as `np`, to the `GaussianMixture` class and to typing annotation `typing.Tuple` as `Tuple`. You also have access to `scipy.special.logsumexp` as `logsumexp`.

Hint: For this function, you will want to use `log(mixture.p[j] + 1e-16)` instead of `log(mixture.p[j])` to avoid numerical underflow.

```python
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
    raise NotImplementedError
```

---

## Implementing M-step (2)

0.0/1.0 point (graded)

In `em.py`, fill in the `mstep` function so that it works with partially observed vectors where missing values are indicated with zeros, and perform the computations in the log domain to help with numerical stability.

Available Functions: You have access to the NumPy python library as `np`, to the `GaussianMixture` class and to typing annotation `typing.Tuple` as `Tuple`.

```python
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
    raise NotImplementedError
```

---

## Implementing run

0.0/1.0 point (graded)

In `em.py`, fill in the `run` function so that it runs the EM algorithm. As before, the convergence criteria that you should use is that the improvement in the log-likelihood is less than or equal to $10^{-6}$ multiplied by the absolute value of the new log-likelihood. Note: do not alter data ‘X’ in place. Deep copy it and enter missing values.

Available Functions: You have access to the NumPy python library as `np`, to the `GaussianMixture` class and to typing annotation `typing.Tuple` as `Tuple`. You also have access to the `estep` and `mstep` functions you have just implemented.

```python
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
    raise NotImplementedError
```

---

# 8. Using the mixture model for collaborative filtering

Project due Aug 12, 2026 08:59 -03

## Reporting log likelihood values on Netflix data

0.0/1.0 point (graded)

Now, run the EM algorithm on the incomplete data matrix from Netflix ratings `netflix_incomplete.txt`. As before, please use seeds from $\{0, 1, 2, 3, 4\}$ and report the best log likelihood you achieve with $K = 1$ and $K = 12$.

This may take on the order of a couple minutes for $K = 12$.

Report the maximum likelihood for each $K$ using seeds $0, 1, 2, 3, 4$:

- $\text{Log-likelihood}\big|_{K=1} =$ [          ]
- $\text{Log-likelihood}\big|_{K=12} =$ [          ]

---

## Completing missing entries

0.0/1.0 point (graded)

Now that we have a mixture model, how do we use it to complete a partially observed rating matrix? Derive an expression for completing a particular row, say $x_C$ where the observed values are $i \in C$.

In `em.py` implement the function `fill_matrix` that takes as input an incomplete data matrix `X` as well as a mixture model, and outputs a completed version of the matrix `X_pred`.

Available Functions: You have access to the NumPy python library as `np`, to the `GaussianMixture` class and to typing annotation `typing.Tuple` as `Tuple`. You also have access to `scipy.special.logsumexp` as `logsumexp`.

```python
def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns:
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
```

---

## Comparing with gold targets

0.0/1.0 point (graded)

Test the accuracy of your predictions against actual target values by loading the complete matrix `X_gold = np.loadtxt('netflix_complete.txt')` and measuring the root mean squared error between the two matrices using `common.rmse(X_gold, X_pred)`. Use your best mixture for $K = 12$ from the first question of this tab to generate the results.

- $\text{RMSE} =$ [          ]
