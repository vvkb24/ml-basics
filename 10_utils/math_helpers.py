"""
Math Helper Functions

Common mathematical operations used in ML algorithms.
"""

import numpy as np
from typing import Union


# =============================================================================
# Activation Functions
# =============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.
    
    σ(x) = 1 / (1 + e^(-x))
    """
    # Numerically stable version
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid: σ'(x) = σ(x) * (1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, x)"""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU."""
    return (x > 0).astype(float)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU: max(αx, x)"""
    return np.where(x > 0, x, alpha * x)


def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent."""
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of tanh: 1 - tanh²(x)"""
    return 1 - np.tanh(x) ** 2


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax function with numerical stability.
    
    softmax(x)_i = exp(x_i) / Σexp(x_j)
    """
    # Subtract max for numerical stability
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# =============================================================================
# Distance Functions
# =============================================================================

def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Euclidean (L2) distance between two vectors."""
    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Manhattan (L1) distance between two vectors."""
    return np.sum(np.abs(x1 - x2))


def cosine_similarity(x1: np.ndarray, x2: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(x1, x2)
    norm1 = np.linalg.norm(x1)
    norm2 = np.linalg.norm(x2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def pairwise_distances(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Compute pairwise distances between all samples.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    metric : str
        'euclidean' or 'manhattan'.
        
    Returns
    -------
    distances : np.ndarray
        Distance matrix (n_samples, n_samples).
    """
    n = len(X)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            if metric == "euclidean":
                d = euclidean_distance(X[i], X[j])
            elif metric == "manhattan":
                d = manhattan_distance(X[i], X[j])
            else:
                raise ValueError(f"Unknown metric: {metric}")
            distances[i, j] = d
            distances[j, i] = d
    
    return distances


# =============================================================================
# Numerical Gradient
# =============================================================================

def numerical_gradient(
    f,
    x: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Compute gradient numerically using finite differences.
    
    Parameters
    ----------
    f : callable
        Function that takes x and returns scalar.
    x : np.ndarray
        Point at which to compute gradient.
    eps : float
        Step size for finite difference.
        
    Returns
    -------
    grad : np.ndarray
        Numerical gradient.
    """
    grad = np.zeros_like(x)
    
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    
    return grad


def gradient_check(
    f,
    grad_f,
    x: np.ndarray,
    eps: float = 1e-5
) -> float:
    """
    Check analytical gradient against numerical gradient.
    
    Returns relative error.
    """
    numerical = numerical_gradient(f, x, eps)
    analytical = grad_f(x)
    
    diff = np.linalg.norm(numerical - analytical)
    norm = np.linalg.norm(numerical) + np.linalg.norm(analytical)
    
    if norm == 0:
        return 0.0
    return diff / norm


# =============================================================================
# Matrix Operations
# =============================================================================

def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Add column of ones for bias term."""
    return np.c_[np.ones(len(X)), X]


def one_hot_encode(y: np.ndarray, n_classes: int = None) -> np.ndarray:
    """
    One-hot encode labels.
    
    Parameters
    ----------
    y : np.ndarray
        Integer labels.
    n_classes : int, optional
        Number of classes (inferred if not provided).
        
    Returns
    -------
    one_hot : np.ndarray
        One-hot encoded matrix (n_samples, n_classes).
    """
    if n_classes is None:
        n_classes = int(np.max(y)) + 1
    
    n_samples = len(y)
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), y.astype(int)] = 1
    
    return one_hot


if __name__ == "__main__":
    # Demo
    print("Activation Functions:")
    x = np.array([-2, -1, 0, 1, 2])
    print(f"  x:       {x}")
    print(f"  sigmoid: {sigmoid(x)}")
    print(f"  relu:    {relu(x)}")
    print(f"  tanh:    {tanh(x)}")
    
    print("\nSoftmax:")
    logits = np.array([2.0, 1.0, 0.1])
    print(f"  logits:  {logits}")
    print(f"  softmax: {softmax(logits)}")
    print(f"  sum:     {softmax(logits).sum()}")
    
    print("\nGradient Check:")
    f = lambda x: np.sum(x ** 2)
    grad_f = lambda x: 2 * x
    x_test = np.array([1.0, 2.0, 3.0])
    error = gradient_check(f, grad_f, x_test)
    print(f"  Relative error: {error:.2e}")
