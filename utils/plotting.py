"""
Plotting Utilities

Helper functions for common ML visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple


def plot_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Regression Fit",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot regression predictions vs actual values.
    
    Parameters
    ----------
    X : np.ndarray
        Feature values (1D for plotting).
    y : np.ndarray
        Actual target values.
    y_pred : np.ndarray
        Predicted values.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if X.ndim == 1 or X.shape[1] == 1:
        X_plot = X.ravel()
        ax.scatter(X_plot, y, alpha=0.6, label='Actual')
        
        # Sort for line plot
        sort_idx = np.argsort(X_plot)
        ax.plot(X_plot[sort_idx], y_pred[sort_idx], 'r-', 
                linewidth=2, label='Predicted')
        
        ax.set_xlabel('X')
        ax.set_ylabel('y')
    else:
        # For multi-dimensional, plot predicted vs actual
        ax.scatter(y, y_pred, alpha=0.6)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
    
    ax.set_title(title)
    ax.legend()
    
    return fig


def plot_learning_curves(
    train_scores: List[float],
    val_scores: Optional[List[float]] = None,
    xlabel: str = "Iteration",
    ylabel: str = "Loss",
    title: str = "Learning Curves",
    figsize: Tuple[int, int] = (10, 5)
) -> plt.Figure:
    """
    Plot training and validation curves.
    
    Parameters
    ----------
    train_scores : list
        Training scores/losses.
    val_scores : list, optional
        Validation scores/losses.
    xlabel, ylabel, title : str
        Axis labels and title.
    figsize : tuple
        Figure size.
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(train_scores, label='Training', alpha=0.8)
    if val_scores is not None:
        ax.plot(val_scores, label='Validation', alpha=0.8)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.
    
    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix.
    classes : list, optional
        Class labels.
    title : str
        Plot title.
    cmap : str
        Colormap.
    figsize : tuple
        Figure size.
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm, cmap=cmap)
    plt.colorbar(im, ax=ax)
    
    n = len(cm)
    if classes is None:
        classes = [str(i) for i in range(n)]
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(n):
        for j in range(n):
            color = 'white' if cm[i, j] > thresh else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color)
    
    return fig


def plot_decision_boundary_2d(
    model,
    X: np.ndarray,
    y: np.ndarray,
    h: float = 0.02,
    title: str = "Decision Boundary",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot decision boundary for 2D classification.
    
    Parameters
    ----------
    model : object
        Fitted model with predict method.
    X : np.ndarray
        Feature matrix (n_samples, 2).
    y : np.ndarray
        Target labels.
    h : float
        Step size for meshgrid.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: Tuple[int, int] = (15, 4)
) -> plt.Figure:
    """
    Plot residual diagnostics.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Predicted values.
    figsize : tuple
        Figure size.
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Residual')
    axes[0].set_title('Residuals vs Predicted')
    
    # Histogram
    axes[1].hist(residuals, bins=30, edgecolor='black')
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    
    # Actual vs Predicted
    axes[2].scatter(y_true, y_pred, alpha=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[2].plot(lims, lims, 'r--')
    axes[2].set_xlabel('Actual')
    axes[2].set_ylabel('Predicted')
    axes[2].set_title('Actual vs Predicted')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Demo
    np.random.seed(42)
    
    # Regression example
    X = np.random.randn(100, 1)
    y = 2 * X.ravel() + 1 + np.random.randn(100) * 0.5
    y_pred = 2.1 * X.ravel() + 0.9
    
    fig = plot_regression_fit(X, y, y_pred, "Demo: Regression Fit")
    plt.show()
