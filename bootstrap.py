
"""
Strong linear model in regression
    Y = X beta + eps, where eps~ N(0, sigma^2 I)
    Under the null where beta_1 = ... = beta_p = 0,
    the R-squared coefficient has a known distribution
    (if you have an intercept beta_0), 
        R^2 ~ Beta(p/2, (n-p-1)/2)
"""

import numpy as np

def bootstrap_sample(X, y, compute_stat, n_bootstrap=1000):
    """
    Generate bootstrap distribution of a statistic

    Parameters
    ----------
    X : array-like, shape (n, p+1)
        Design matrix
    y : array-like, shape (n,)
    compute_stat : callable
        Function that computes a statistic (float) from data (X, y)
    n_bootstrap : int, default 1000
        Number of bootstrap samples to generate

    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics, length n_bootstrap

    ....
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    if X.shape[0] != len(y):
        raise ValueError("Number of rows in X must match length of y")
    
    n = len(y)
    stats = np.empty(n_bootstrap, dtype=float)
    
    for i in range(n_bootstrap):
        # Sample indices with replacement
        idx = np.random.randint(0, n, size=n)
        X_boot = X[idx]
        y_boot = y[idx]
        
        # Compute statistic on bootstrap sample
        stats[i] = compute_stat(X_boot, y_boot)
    
    return stats

def bootstrap_ci(bootstrap_stats, alpha=0.05):
    """
    Calculate confidence interval from the bootstrap samples

    Parameters
    ----------
    bootstrap_stats : array-like
        Array of bootstrap statistics
    alpha : float, default 0.05
        Significance level (e.g. 0.05 gives 95% CI)

    Returns
    -------
    tuple 
        (lower_bound, upper_bound) of the CI
    
    ....
    """
    lower_bound = np.percentile(bootstrap_stats, 100 * (alpha / 2))
    upper_bound = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    return lower_bound, upper_bound

def R_squared(X, y):
    """
    Calculate R-squared from multiple linear regression.

    Parameters
    ----------
    X : array-like, shape (n, p+1)
        Design matrix
    y : array-like, shape (n,)

    Returns
    -------
    float
        R-squared value (between 0 and 1) from OLS
    
    Raises
    ------
    ValueError
        If X.shape[0] != len(y)
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != len(y):
        raise ValueError("Number of rows in X must match length of y")
    
    if np.linalg.matrix_rank(X) == X.shape[1]:
        beta = np.linalg.solve(X.T @ X, X.T @ y)  # More numerically stable
    else:
        beta = np.linalg.pinv(X) @ y  # Fallback for rank-deficient
    
    # Predicted values
    y_hat = X @ beta
    
    # Total sum of squares
    ss_total = np.sum((y - np.mean(y))**2)
    
    if ss_total == 0:
        return 0.0

    # Residual sum of squares
    ss_res = np.sum((y - y_hat)**2)
    
    # R-squared
    r2 = 1 - ss_res / ss_total
    
    return float(r2)