"""
Metrics Module

Common evaluation metrics for regression and classification tasks.
"""

import numpy as np
from typing import Optional


# =============================================================================
# Regression Metrics
# =============================================================================

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error.
    
    MSE = (1/n) * Σ(y - ŷ)²
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.
    
    MAE = (1/n) * Σ|y - ŷ|
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² (Coefficient of Determination).
    
    R² = 1 - SS_res / SS_tot
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1 - (ss_res / ss_tot)


# =============================================================================
# Classification Metrics
# =============================================================================

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification accuracy."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Returns matrix where C[i,j] is count of true=i, pred=j.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i, c_true in enumerate(classes):
        for j, c_pred in enumerate(classes):
            cm[i, j] = np.sum((y_true == c_true) & (y_pred == c_pred))
    
    return cm


def precision(y_true: np.ndarray, y_pred: np.ndarray, 
              positive_class: int = 1) -> float:
    """
    Precision = TP / (TP + FP)
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == positive_class) & (y_pred == positive_class))
    fp = np.sum((y_true != positive_class) & (y_pred == positive_class))
    
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall(y_true: np.ndarray, y_pred: np.ndarray,
           positive_class: int = 1) -> float:
    """
    Recall = TP / (TP + FN)
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == positive_class) & (y_pred == positive_class))
    fn = np.sum((y_true == positive_class) & (y_pred != positive_class))
    
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray,
             positive_class: int = 1) -> float:
    """
    F1 = 2 * (precision * recall) / (precision + recall)
    """
    prec = precision(y_true, y_pred, positive_class)
    rec = recall(y_true, y_pred, positive_class)
    
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray,
                         eps: float = 1e-15) -> float:
    """
    Binary cross-entropy loss.
    
    BCE = -(1/n) * Σ[y*log(p) + (1-y)*log(1-p)]
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray,
                  eps: float = 1e-15) -> float:
    """
    Categorical cross-entropy loss.
    
    y_true: one-hot encoded (n_samples, n_classes)
    y_pred: probabilities (n_samples, n_classes)
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Regression example
    y_true_reg = np.array([1, 2, 3, 4, 5])
    y_pred_reg = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    
    print("Regression Metrics:")
    print(f"  MSE:  {mean_squared_error(y_true_reg, y_pred_reg):.4f}")
    print(f"  RMSE: {root_mean_squared_error(y_true_reg, y_pred_reg):.4f}")
    print(f"  MAE:  {mean_absolute_error(y_true_reg, y_pred_reg):.4f}")
    print(f"  R²:   {r2_score(y_true_reg, y_pred_reg):.4f}")
    
    # Classification example
    y_true_clf = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    y_pred_clf = np.array([0, 1, 0, 0, 1, 1, 1, 1])
    
    print("\nClassification Metrics:")
    print(f"  Accuracy:  {accuracy(y_true_clf, y_pred_clf):.4f}")
    print(f"  Precision: {precision(y_true_clf, y_pred_clf):.4f}")
    print(f"  Recall:    {recall(y_true_clf, y_pred_clf):.4f}")
    print(f"  F1:        {f1_score(y_true_clf, y_pred_clf):.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true_clf, y_pred_clf))
