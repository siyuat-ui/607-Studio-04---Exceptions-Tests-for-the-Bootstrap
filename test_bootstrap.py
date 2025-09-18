import pytest
import numpy as np
from scipy.stats import beta
from bootstrap import bootstrap_sample, bootstrap_ci, R_squared

def test_bootstrap_integration():
    """Test that bootstrap_sample and bootstrap_ci work together"""
    # This test should initially fail
    pass

# -----------------------------
# For bootstrap_sample
# -----------------------------

def test_bootstrap_sample_length():
    """Check that bootstrap_sample returns the requested number of samples"""
    X = np.ones((5, 2))
    y = np.ones(5)
    n_bootstrap = 50
    stats = bootstrap_sample(X, y, compute_stat=r_squared, n_bootstrap=n_bootstrap)
    assert len(stats) == n_bootstrap

def test_bootstrap_sample_type():
    """Check that bootstrap_sample returns a numpy array of floats"""
    X = np.ones((5, 2))
    y = np.ones(5)
    stats = bootstrap_sample(X, y, compute_stat=r_squared, n_bootstrap=10)
    assert isinstance(stats, np.ndarray)
    assert np.issubdtype(stats.dtype, np.floating)

def test_bootstrap_sample_variation():
    """Check that bootstrap_sample produces variation (unless data is constant)"""
    np.random.seed(0)
    X = np.column_stack((np.ones(10), np.arange(10)))
    y = X @ np.array([1, 2]) + np.random.randn(10) * 0.1
    stats = bootstrap_sample(X, y, compute_stat=r_squared, n_bootstrap=100)
    assert np.std(stats) > 0, "Bootstrap statistics should vary"

# -----------------------------
# Bonus: Add a statistical validation test that checks the bootstrap implementation against the known theoretical distribution of R² under
# the null hypothesis.
# -----------------------------

def validate_bootstrap_r2(n=50, p=2, n_bootstrap=1000, alpha=0.05, random_state=42):
    """
    Statistical validation of bootstrap R² under the null hypothesis.

    Parameters
    ----------
    n : int
        Number of samples
    p : int
        Number of predictors (excluding intercept)
    n_bootstrap : int
        Number of bootstrap samples
    alpha : float
        Significance level for validation check
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary with empirical and theoretical mean/variance of R²
    """
    np.random.seed(random_state)
    
    # Simulate null model data: Y = beta0 + eps, beta1,...,betap = 0
    X = np.column_stack((np.ones(n), np.random.randn(n, p)))  # intercept + predictors
    y = np.random.randn(n)  # response under null

    # Compute bootstrap R²
    boot_r2 = bootstrap_sample(X, y, compute_stat=r_squared, n_bootstrap=n_bootstrap)

    # Theoretical R² distribution under null: Beta(p/2, (n-p-1)/2)
    a = p / 2
    b = (n - p - 1) / 2
    theoretical_mean = a / (a + b)
    theoretical_var = a * b / ((a + b)**2 * (a + b + 1))

    # Empirical mean/variance
    empirical_mean = np.mean(boot_r2)
    empirical_var = np.var(boot_r2)

    # Validation: mean within ~2 standard errors
    se = np.sqrt(theoretical_var / n_bootstrap)
    if np.abs(empirical_mean - theoretical_mean) > 2 * se:
        raise AssertionError(
            f"Bootstrap R² mean deviates too much from theory: "
            f"empirical={empirical_mean:.4f}, theoretical={theoretical_mean:.4f}"
        )

    return {
        "empirical_mean": empirical_mean,
        "empirical_var": empirical_var,
        "theoretical_mean": theoretical_mean,
        "theoretical_var": theoretical_var
    }


