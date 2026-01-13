"""
Algorithm Tests

Unit tests for machine learning algorithm implementations.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.supervised.linear_regression.scratch import (
    LinearRegressionScratch,
    RidgeRegressionScratch,
    compute_metrics
)


class TestLinearRegression:
    """Tests for Linear Regression implementation."""
    
    def test_normal_equation_perfect_fit(self):
        """Test that normal equation gives exact solution for noise-free data."""
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3
        
        model = LinearRegressionScratch(method='normal_equation')
        model.fit(X, y)
        
        np.testing.assert_array_almost_equal(model.weights_, [1, 2], decimal=5)
        np.testing.assert_almost_equal(model.bias_, 3, decimal=5)
        
        r2 = model.score(X, y)
        assert r2 > 0.9999, f"R² should be ~1.0, got {r2}"
    
    def test_gradient_descent_convergence(self):
        """Test that gradient descent converges to similar solution."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        true_w = np.array([2, -1, 0.5])
        y = X @ true_w + 1 + np.random.randn(100) * 0.1
        
        model = LinearRegressionScratch(method='gradient_descent')
        model.fit(X, y, learning_rate=0.1, n_iterations=1000)
        
        # Should be close to true weights
        np.testing.assert_array_almost_equal(model.weights_, true_w, decimal=1)
        
        # Loss should decrease
        assert model.loss_history_[-1] < model.loss_history_[0]
    
    def test_prediction_shape(self):
        """Test that prediction output has correct shape."""
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)
        X_test = np.random.randn(20, 5)
        
        model = LinearRegressionScratch()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)
    
    def test_ridge_regularization(self):
        """Test that ridge regularization shrinks weights."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = X @ np.random.randn(10) + np.random.randn(50)
        
        # Unregularized
        model_plain = LinearRegressionScratch(regularization=0)
        model_plain.fit(X, y)
        
        # With strong regularization
        model_ridge = LinearRegressionScratch(regularization=10.0)
        model_ridge.fit(X, y)
        
        # Ridge weights should have smaller norm
        plain_norm = np.linalg.norm(model_plain.weights_)
        ridge_norm = np.linalg.norm(model_ridge.weights_)
        
        assert ridge_norm < plain_norm, "Ridge should shrink weights"
    
    def test_1d_input(self):
        """Test with 1D feature input."""
        X = np.array([1, 2, 3, 4, 5])
        y = 2 * X + 1
        
        model = LinearRegressionScratch()
        model.fit(X, y)
        
        assert len(model.weights_) == 1
        np.testing.assert_almost_equal(model.weights_[0], 2, decimal=5)


class TestMetrics:
    """Tests for metric functions."""
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        metrics = compute_metrics(y_true, y_pred)
        
        assert metrics['mse'] == 0
        assert metrics['rmse'] == 0
        assert metrics['mae'] == 0
        assert metrics['r2'] == 1.0
    
    def test_r2_worse_than_mean(self):
        """Test R² can be negative if worse than predicting mean."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([10, 20, 30])  # Very wrong
        
        metrics = compute_metrics(y_true, y_pred)
        assert metrics['r2'] < 0


def run_tests():
    """Run all tests."""
    print("Running Linear Regression Tests")
    print("=" * 50)
    
    lr_tests = TestLinearRegression()
    metrics_tests = TestMetrics()
    
    tests = [
        ("Normal Equation Perfect Fit", lr_tests.test_normal_equation_perfect_fit),
        ("Gradient Descent Convergence", lr_tests.test_gradient_descent_convergence),
        ("Prediction Shape", lr_tests.test_prediction_shape),
        ("Ridge Regularization", lr_tests.test_ridge_regularization),
        ("1D Input", lr_tests.test_1d_input),
        ("Perfect Predictions Metrics", metrics_tests.test_perfect_predictions),
        ("R² Negative Case", metrics_tests.test_r2_worse_than_mean),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            print(f"  ✓ {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
