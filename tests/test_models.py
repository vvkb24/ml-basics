"""
Model Tests

Tests for model behavior and integration.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestModelIntegration:
    """Integration tests for models."""
    
    def test_linear_regression_sklearn_comparison(self):
        """Compare our implementation with sklearn."""
        try:
            from sklearn.linear_model import LinearRegression
            from algorithms.supervised.linear_regression.scratch import LinearRegressionScratch
            
            np.random.seed(42)
            X = np.random.randn(100, 5)
            y = X @ np.array([1, 2, 3, 4, 5]) + 2 + np.random.randn(100) * 0.5
            
            # Our implementation
            model_scratch = LinearRegressionScratch(method='normal_equation')
            model_scratch.fit(X, y)
            
            # sklearn
            model_sklearn = LinearRegression()
            model_sklearn.fit(X, y)
            
            # Compare
            np.testing.assert_array_almost_equal(
                model_scratch.weights_,
                model_sklearn.coef_,
                decimal=5
            )
            np.testing.assert_almost_equal(
                model_scratch.bias_,
                model_sklearn.intercept_,
                decimal=5
            )
            
        except ImportError:
            print("  ⚠ sklearn not installed, skipping comparison")
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        from utils.data_loader import generate_synthetic_regression
        
        X1, y1, w1 = generate_synthetic_regression(random_state=42)
        X2, y2, w2 = generate_synthetic_regression(random_state=42)
        
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


def run_tests():
    """Run all tests."""
    print("Running Model Integration Tests")
    print("=" * 50)
    
    test_obj = TestModelIntegration()
    
    passed = 0
    failed = 0
    
    for method_name in dir(test_obj):
        if method_name.startswith('test_'):
            try:
                getattr(test_obj, method_name)()
                print(f"  ✓ {method_name}")
                passed += 1
            except AssertionError as e:
                print(f"  ✗ {method_name}: {e}")
                failed += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {type(e).__name__}: {e}")
                failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
