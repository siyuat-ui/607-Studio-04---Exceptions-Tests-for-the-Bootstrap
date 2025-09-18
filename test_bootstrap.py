import pytest
import numpy as np
from scipy.stats import kstest, beta
from bootstrap import bootstrap_sample, bootstrap_ci, R_squared

def test_bootstrap_integration():
    """Integration test: bootstrap_sample + bootstrap_ci end-to-end"""

    np.random.seed(42)

    # 1. Simulate linear data
    n, p = 20, 2
    X = np.column_stack((np.ones(n), np.random.randn(n, p)))  # intercept + predictors
    beta_true = np.array([1.0, 0.5, -0.5])
    y = X @ beta_true + np.random.randn(n) * 0.1

    # 2. Compute bootstrap distribution of R²
    n_bootstrap = 200
    boot_r2 = bootstrap_sample(X, y, compute_stat=R_squared, n_bootstrap=n_bootstrap)

    # 3. Compute 95% CI
    ci_lower, ci_upper = bootstrap_ci(boot_r2, alpha=0.05)

    # 4. Compute R² on the full dataset
    r2_full = R_squared(X, y)

    # 5. Assertions
    assert 0 <= ci_lower <= ci_upper <= 1, "CI bounds should be valid"
    assert ci_lower <= r2_full <= ci_upper, "Full-data R² should be within the bootstrap CI"
    assert len(boot_r2) == n_bootstrap, "Bootstrap array length mismatch"

# -----------------------------
# For bootstrap_sample
# -----------------------------

def test_bootstrap_sample_length():
    """Happy path: returns correct number of bootstrap samples"""
    X = np.ones((5, 2))
    y = np.ones(5)
    n_bootstrap = 50
    stats = bootstrap_sample(X, y, compute_stat=R_squared, n_bootstrap=n_bootstrap)
    assert len(stats) == n_bootstrap

def test_bootstrap_sample_type():
    """Happy path: returns a numpy array of floats"""
    X = np.ones((5, 2))
    y = np.ones(5)
    stats = bootstrap_sample(X, y, compute_stat=R_squared, n_bootstrap=10)
    assert isinstance(stats, np.ndarray)
    assert np.issubdtype(stats.dtype, np.floating)

def test_bootstrap_sample_variation():
    """Happy path: bootstrap produces variation"""
    np.random.seed(0)
    X = np.column_stack((np.ones(10), np.arange(10)))
    y = X @ np.array([1, 2]) + np.random.randn(10) * 0.1
    stats = bootstrap_sample(X, y, compute_stat=R_squared, n_bootstrap=100)
    assert np.std(stats) > 0

# ---------------------------
# Edge cases
# ---------------------------

def test_bootstrap_sample_nbootstrap_one():
    """Edge case: n_bootstrap=1 returns a single-element array"""
    X = np.ones((5, 2))
    y = np.ones(5)
    stats = bootstrap_sample(X, y, compute_stat=R_squared, n_bootstrap=1)
    assert len(stats) == 1

def test_bootstrap_sample_single_row():
    """Edge case: single-row X and y"""
    X = np.array([[1, 2]])
    y = np.array([5])
    stats = bootstrap_sample(X, y, compute_stat=R_squared, n_bootstrap=5)
    assert len(stats) == 5
    assert np.all(np.isfinite(stats))

def test_bootstrap_sample_single_column():
    """Edge case: single predictor column (plus intercept)"""
    X = np.column_stack((np.ones(5), np.arange(5)))
    y = np.arange(5)
    stats = bootstrap_sample(X, y, compute_stat=R_squared, n_bootstrap=10)
    assert len(stats) == 10

# ---------------------------
# Invalid inputs
# ---------------------------

def test_bootstrap_sample_mismatched_X_y():
    """Invalid input: X and y have different number of rows"""
    X = np.ones((5, 2))
    y = np.ones(4)
    with pytest.raises(ValueError):
        bootstrap_sample(X, y, compute_stat=R_squared, n_bootstrap=10)

# ---------------------------
# Tests for R_squared
# ---------------------------

def test_R_squared_basic():
    """Happy path: correct R² for simple linear regression"""
    X = np.column_stack((np.ones(5), np.arange(5)))  # intercept + 1 predictor
    y = np.array([1, 3, 5, 7, 9])  # perfectly linear
    r2 = R_squared(X, y)
    assert np.isclose(r2, 1.0)  # perfect fit

def test_R_squared_bounds():
    """R² should always be between 0 and 1"""
    np.random.seed(0)
    X = np.column_stack((np.ones(10), np.random.randn(10, 2)))
    y = np.random.randn(10)
    r2 = R_squared(X, y)
    assert 0 <= r2 <= 1

def test_R_squared_constant_y():
    """Edge case: constant y should return R² = 0"""
    X = np.column_stack((np.ones(5), np.arange(5)))
    y = np.ones(5)
    r2 = R_squared(X, y)
    assert np.isclose(r2, 0.0)

def test_R_squared_single_row():
    """Edge case: single-row X and y"""
    X = np.array([[1, 2]])
    y = np.array([5])
    r2 = R_squared(X, y)
    # With a single observation, total sum of squares is zero, can define R²=0
    assert r2 == 0.0 or np.isnan(r2)  # accept NaN or 0

def test_R_squared_mismatched_X_y():
    """Invalid input: X and y have different number of rows"""
    X = np.ones((5, 2))
    y = np.ones(4)
    with pytest.raises(ValueError):
        R_squared(X, y)

# ---------------------------
# Tests for bootstrap_ci
# ---------------------------

def test_bootstrap_ci_basic():
    """Happy path: CI correctly computed for a typical bootstrap array"""
    data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    lower, upper = bootstrap_ci(data, alpha=0.05)
    expected_lower = np.percentile(data, 2.5)
    expected_upper = np.percentile(data, 97.5)
    assert np.isclose(lower, expected_lower)
    assert np.isclose(upper, expected_upper)
    assert lower <= upper

def test_bootstrap_ci_small_array():
    """Edge case: small bootstrap array"""
    data = np.array([0.5, 0.6])
    lower, upper = bootstrap_ci(data, alpha=0.2)
    expected_lower = np.percentile(data, 10)
    expected_upper = np.percentile(data, 90)
    assert np.isclose(lower, expected_lower)
    assert np.isclose(upper, expected_upper)

def test_bootstrap_ci_alpha_extremes():
    """Edge case: alpha near 0 or 1"""
    data = np.linspace(0, 1, 10)
    
    # Very small alpha => almost full range
    lower, upper = bootstrap_ci(data, alpha=0.001)
    assert lower <= upper
    assert np.isclose(lower, np.percentile(data, 0.05))
    assert np.isclose(upper, np.percentile(data, 99.95))
    
    # Large alpha => narrow interval
    lower, upper = bootstrap_ci(data, alpha=0.9)
    assert lower <= upper
    assert np.isclose(lower, np.percentile(data, 45))
    assert np.isclose(upper, np.percentile(data, 55))

def test_bootstrap_ci_input_types():
    """CI works with both lists and numpy arrays"""
    data_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    data_array = np.array(data_list)
    lower1, upper1 = bootstrap_ci(data_list, alpha=0.05)
    lower2, upper2 = bootstrap_ci(data_array, alpha=0.05)
    assert np.isclose(lower1, lower2)
    assert np.isclose(upper1, upper2)

def test_bootstrap_ci_bounds():
    """Check that lower <= upper"""
    data = np.random.rand(100)
    lower, upper = bootstrap_ci(data, alpha=0.1)
    assert lower <= upper

# -----------------------------
# Bonus: Add a statistical validation test that checks the bootstrap implementation against the known theoretical distribution of R² under
# the null hypothesis.
# -----------------------------

from scipy.stats import beta
import numpy as np

def validate_bootstrap_r2(n=50, p=2, n_bootstrap=5000, alpha=0.05, random_state=42):
    """
    Statistical validation of bootstrap R² under the null hypothesis using Beta quantiles.

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
    
    # 1. Simulate null model: Y = beta0 + eps
    X = np.column_stack((np.ones(n), np.random.randn(n, p)))  # intercept + predictors
    y = np.random.randn(n)  # response under null
    
    # 2. Compute bootstrap R²
    boot_r2 = bootstrap_sample(X, y, compute_stat=R_squared, n_bootstrap=n_bootstrap)
    
    # 3. Theoretical Beta distribution for R² under null
    a = p / 2
    b = (n - p - 1) / 2
    beta_dist = beta(a, b)
    
    # 4. Compare empirical R² to theoretical quantiles
    lower_q, upper_q = beta_dist.ppf(alpha/2), beta_dist.ppf(1 - alpha/2)
    n_outside = np.sum((boot_r2 < lower_q) | (boot_r2 > upper_q))
    proportion_outside = n_outside / n_bootstrap
    
    # 5. Raise error if too many R² values are outside the theoretical CI
    if proportion_outside > alpha * 1.8:  # allow slight tolerance
        raise AssertionError(
            f"Too many bootstrap R² values ({proportion_outside:.2%}) "
            f"fall outside the theoretical {100*(1-alpha):.1f}% interval "
            f"[{lower_q:.3f}, {upper_q:.3f}]"
        )
    
    # 6. Return summary
    return {
        "empirical_mean": np.mean(boot_r2),
        "empirical_var": np.var(boot_r2),
        "theoretical_mean": beta_dist.mean(),
        "theoretical_var": beta_dist.var(),
        "proportion_outside": proportion_outside,
        "theoretical_CI": (lower_q, upper_q)
    }

def test_validate_bootstrap_r2():
    validate_bootstrap_r2()
