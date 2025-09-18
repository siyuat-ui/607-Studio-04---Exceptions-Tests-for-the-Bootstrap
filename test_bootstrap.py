import pytest
import numpy as np
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


