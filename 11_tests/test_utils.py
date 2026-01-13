"""
Utility Tests

Unit tests for utility modules.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import (
    mean_squared_error, r2_score, accuracy, precision, recall, f1_score
)
from utils.data_loader import (
    train_test_split, normalize, k_fold_split, one_hot_encode
)
from utils.math_helpers import (
    sigmoid, softmax, relu, gradient_check
)


class TestMetrics:
    """Tests for metrics module."""
    
    def test_mse(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        assert mean_squared_error(y_true, y_pred) == 0
    
    def test_r2_perfect(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])
        assert r2_score(y_true, y_pred) == 1.0
    
    def test_accuracy(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        assert accuracy(y_true, y_pred) == 0.75


class TestDataLoader:
    """Tests for data loader module."""
    
    def test_train_test_split(self):
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        assert len(X_train) == 80
        assert len(X_test) == 20
    
    def test_normalize_standard(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        X_norm, params = normalize(X, method='standard')
        
        np.testing.assert_array_almost_equal(X_norm.mean(axis=0), [0, 0], decimal=10)
    
    def test_k_fold(self):
        folds = k_fold_split(100, n_folds=5, shuffle=False)
        
        assert len(folds) == 5
        for train_idx, val_idx in folds:
            assert len(val_idx) == 20
            assert len(train_idx) == 80


class TestMathHelpers:
    """Tests for math helpers."""
    
    def test_sigmoid_bounds(self):
        x = np.array([-100, -1, 0, 1, 100])
        s = sigmoid(x)
        
        assert np.all(s >= 0) and np.all(s <= 1)
        np.testing.assert_almost_equal(sigmoid(np.array([0]))[0], 0.5)
    
    def test_softmax_sums_to_one(self):
        x = np.array([1, 2, 3, 4])
        s = softmax(x)
        
        np.testing.assert_almost_equal(s.sum(), 1.0)
    
    def test_relu(self):
        x = np.array([-2, -1, 0, 1, 2])
        expected = np.array([0, 0, 0, 1, 2])
        np.testing.assert_array_equal(relu(x), expected)
    
    def test_gradient_check(self):
        f = lambda x: np.sum(x ** 2)
        grad_f = lambda x: 2 * x
        x = np.array([1.0, 2.0, 3.0])
        
        error = gradient_check(f, grad_f, x)
        assert error < 1e-5, f"Gradient check failed with error {error}"


def run_tests():
    """Run all tests."""
    print("Running Utility Tests")
    print("=" * 50)
    
    test_classes = [
        ("Metrics", TestMetrics()),
        ("DataLoader", TestDataLoader()),
        ("MathHelpers", TestMathHelpers()),
    ]
    
    passed = 0
    failed = 0
    
    for class_name, test_obj in test_classes:
        print(f"\n{class_name}:")
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
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
