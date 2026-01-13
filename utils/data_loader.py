"""
Data Loader Utilities

Functions for loading, splitting, and preprocessing data.
"""

import numpy as np
from typing import Tuple, Optional


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    test_size : float
        Fraction of data for testing.
    random_state : int, optional
        Random seed for reproducibility.
    shuffle : bool
        Whether to shuffle before splitting.
        
    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n = len(X)
    indices = np.arange(n)
    
    if shuffle:
        np.random.shuffle(indices)
    
    split_idx = int(n * (1 - test_size))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def k_fold_split(
    n_samples: int,
    n_folds: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None
) -> list:
    """
    Generate k-fold cross-validation indices.
    
    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_folds : int
        Number of folds.
    shuffle : bool
        Whether to shuffle indices.
    random_state : int, optional
        Random seed.
        
    Returns
    -------
    folds : list of (train_indices, val_indices) tuples
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    fold_size = n_samples // n_folds
    folds = []
    
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n_samples
        
        val_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        
        folds.append((train_indices, val_indices))
    
    return folds


def normalize(
    X: np.ndarray,
    method: str = "standard"
) -> Tuple[np.ndarray, dict]:
    """
    Normalize features.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    method : str
        'standard' (z-score) or 'minmax'.
        
    Returns
    -------
    X_normalized : np.ndarray
    params : dict
        Parameters for applying to new data.
    """
    if method == "standard":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1  # Avoid division by zero
        X_normalized = (X - mean) / std
        params = {"mean": mean, "std": std, "method": method}
    elif method == "minmax":
        min_val = X.min(axis=0)
        max_val = X.max(axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        X_normalized = (X - min_val) / range_val
        params = {"min": min_val, "max": max_val, "method": method}
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return X_normalized, params


def apply_normalization(X: np.ndarray, params: dict) -> np.ndarray:
    """Apply saved normalization parameters to new data."""
    if params["method"] == "standard":
        return (X - params["mean"]) / params["std"]
    elif params["method"] == "minmax":
        return (X - params["min"]) / (params["max"] - params["min"])
    else:
        raise ValueError(f"Unknown method: {params['method']}")


def add_polynomial_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Add polynomial features.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    degree : int
        Maximum polynomial degree.
        
    Returns
    -------
    X_poly : np.ndarray
        Expanded feature matrix.
    """
    n_samples, n_features = X.shape
    features = [X]
    
    # Add higher-degree terms
    for d in range(2, degree + 1):
        features.append(X ** d)
    
    return np.hstack(features)


def generate_synthetic_regression(
    n_samples: int = 100,
    n_features: int = 3,
    noise_std: float = 0.1,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data.
    
    Returns
    -------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target values.
    true_weights : np.ndarray
        Ground truth weights.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    noise = np.random.randn(n_samples) * noise_std
    y = X @ true_weights + noise
    
    return X, y, true_weights


def generate_synthetic_classification(
    n_samples: int = 100,
    n_features: int = 2,
    n_classes: int = 2,
    separation: float = 1.0,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic classification data.
    
    Returns
    -------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Class labels.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    samples_per_class = n_samples // n_classes
    
    X_list = []
    y_list = []
    
    for c in range(n_classes):
        # Random center for each class
        center = np.random.randn(n_features) * separation * c
        X_class = np.random.randn(samples_per_class, n_features) + center
        X_list.append(X_class)
        y_list.append(np.full(samples_per_class, c))
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    return X[shuffle_idx], y[shuffle_idx]


if __name__ == "__main__":
    # Demo
    X, y, true_w = generate_synthetic_regression(n_samples=100, random_state=42)
    print(f"Generated regression data: X{X.shape}, y{y.shape}")
    print(f"True weights: {true_w}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(f"Split: Train={len(X_train)}, Test={len(X_test)}")
    
    X_norm, params = normalize(X_train)
    print(f"Normalized mean: {X_norm.mean(axis=0)}")
