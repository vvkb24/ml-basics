"""
Linear Regression - From Scratch Implementation

This module implements linear regression using only NumPy,
demonstrating both the closed-form solution and gradient descent.

Mathematical basis:
- Model: ŷ = Xθ
- Loss: L(θ) = (1/n)||y - Xθ||²
- Normal equation: θ* = (XᵀX)⁻¹Xᵀy
- Gradient: ∇L = (2/n)Xᵀ(Xθ - y)
"""

import numpy as np
from typing import Optional, Literal, Tuple


class LinearRegressionScratch:
    """
    Linear Regression implementation from scratch.
    
    Supports both closed-form (normal equation) and gradient descent solutions.
    Optionally includes L2 regularization (Ridge regression).
    
    Parameters
    ----------
    method : {'normal_equation', 'gradient_descent'}
        Optimization method to use.
    regularization : float, default=0.0
        L2 regularization strength (lambda). Set to 0 for no regularization.
        
    Attributes
    ----------
    weights_ : np.ndarray of shape (n_features,)
        Learned feature weights after fitting.
    bias_ : float
        Learned bias (intercept) term.
    loss_history_ : list
        Loss values during training (gradient descent only).
        
    Examples
    --------
    >>> import numpy as np
    >>> from scratch import LinearRegressionScratch
    >>> 
    >>> # Generate sample data
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 3)
    >>> y = 2*X[:, 0] - X[:, 1] + 0.5*X[:, 2] + 1 + np.random.randn(100)*0.1
    >>> 
    >>> # Fit with normal equation
    >>> model = LinearRegressionScratch(method='normal_equation')
    >>> model.fit(X, y)
    >>> print(f"Weights: {model.weights_}")
    >>> print(f"Bias: {model.bias_}")
    >>> print(f"R²: {model.score(X, y):.4f}")
    """
    
    def __init__(
        self,
        method: Literal['normal_equation', 'gradient_descent'] = 'normal_equation',
        regularization: float = 0.0
    ):
        self.method = method
        self.regularization = regularization
        self.weights_: Optional[np.ndarray] = None
        self.bias_: Optional[float] = None
        self.loss_history_: list = []
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        tolerance: float = 1e-6,
        verbose: bool = False
    ) -> 'LinearRegressionScratch':
        """
        Fit the linear regression model.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.
        learning_rate : float, default=0.01
            Learning rate for gradient descent (ignored for normal equation).
        n_iterations : int, default=1000
            Maximum iterations for gradient descent (ignored for normal equation).
        tolerance : float, default=1e-6
            Convergence threshold for gradient descent.
        verbose : bool, default=False
            Print progress during training.
            
        Returns
        -------
        self : LinearRegressionScratch
            Fitted model.
        """
        # Validate input
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        n_samples, n_features = X.shape
        
        if y.shape[0] != n_samples:
            raise ValueError(f"X and y have inconsistent samples: {n_samples} vs {y.shape[0]}")
        
        # Add bias column (column of ones)
        X_b = np.c_[np.ones(n_samples), X]
        
        if self.method == 'normal_equation':
            self._fit_normal_equation(X_b, y)
        elif self.method == 'gradient_descent':
            self._fit_gradient_descent(
                X_b, y, learning_rate, n_iterations, tolerance, verbose
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        return self
    
    def _fit_normal_equation(self, X_b: np.ndarray, y: np.ndarray) -> None:
        """
        Fit using the normal equation: θ = (XᵀX + λI)⁻¹Xᵀy
        
        Mathematical derivation:
        1. Loss: L(θ) = (1/n)||y - Xθ||² + λ||θ||²
        2. Gradient: ∇L = (2/n)(-Xᵀy + XᵀXθ) + 2λθ = 0
        3. Solve: (XᵀX + nλI)θ = Xᵀy
        4. Solution: θ = (XᵀX + nλI)⁻¹Xᵀy
        """
        n_samples = X_b.shape[0]
        n_params = X_b.shape[1]
        
        # XᵀX
        XtX = X_b.T @ X_b
        
        # Add regularization (don't regularize bias term)
        if self.regularization > 0:
            reg_matrix = self.regularization * n_samples * np.eye(n_params)
            reg_matrix[0, 0] = 0  # Don't regularize bias
            XtX = XtX + reg_matrix
        
        # Xᵀy
        Xty = X_b.T @ y
        
        # Solve: θ = (XᵀX)⁻¹Xᵀy
        # Use np.linalg.solve for numerical stability instead of explicit inverse
        theta = np.linalg.solve(XtX, Xty)
        
        # Extract bias and weights
        self.bias_ = theta[0]
        self.weights_ = theta[1:]
        
    def _fit_gradient_descent(
        self,
        X_b: np.ndarray,
        y: np.ndarray,
        learning_rate: float,
        n_iterations: int,
        tolerance: float,
        verbose: bool
    ) -> None:
        """
        Fit using gradient descent.
        
        Gradient derivation:
        L(θ) = (1/2n)||y - Xθ||² + (λ/2)||θ||²
        ∇L = (1/n)Xᵀ(Xθ - y) + λθ
        
        Update rule: θ ← θ - η∇L
        """
        n_samples, n_params = X_b.shape
        
        # Initialize weights (zeros or small random)
        theta = np.zeros(n_params)
        
        self.loss_history_ = []
        
        for iteration in range(n_iterations):
            # Compute predictions
            y_pred = X_b @ theta
            
            # Compute residuals
            residuals = y_pred - y
            
            # Compute gradient: (1/n)Xᵀ(Xθ - y)
            gradient = (1 / n_samples) * (X_b.T @ residuals)
            
            # Add L2 regularization gradient (don't regularize bias)
            if self.regularization > 0:
                reg_gradient = self.regularization * theta
                reg_gradient[0] = 0  # Don't regularize bias
                gradient = gradient + reg_gradient
            
            # Update parameters
            theta = theta - learning_rate * gradient
            
            # Compute and store loss
            loss = self._compute_loss(y, y_pred, theta)
            self.loss_history_.append(loss)
            
            # Check convergence
            if len(self.loss_history_) > 1:
                improvement = abs(self.loss_history_[-2] - self.loss_history_[-1])
                if improvement < tolerance:
                    if verbose:
                        print(f"Converged at iteration {iteration + 1}")
                    break
            
            # Print progress
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{n_iterations}, Loss: {loss:.6f}")
        
        # Extract bias and weights
        self.bias_ = theta[0]
        self.weights_ = theta[1:]
        
    def _compute_loss(self, y: np.ndarray, y_pred: np.ndarray, theta: np.ndarray) -> float:
        """Compute MSE loss with optional regularization."""
        n = len(y)
        mse = (1 / (2 * n)) * np.sum((y - y_pred) ** 2)
        
        if self.regularization > 0:
            # L2 regularization (exclude bias)
            reg = (self.regularization / 2) * np.sum(theta[1:] ** 2)
            return mse + reg
        return mse
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Features to predict.
            
        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values.
        """
        if self.weights_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
            
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        return X @ self.weights_ + self.bias_
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score.
        
        R² = 1 - SS_res / SS_tot
           = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Features.
        y : np.ndarray of shape (n_samples,)
            True target values.
            
        Returns
        -------
        r2 : float
            R² score (coefficient of determination).
        """
        y_pred = self.predict(X)
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
            
        return 1 - (ss_res / ss_tot)
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'weights': self.weights_,
            'bias': self.bias_,
            'method': self.method,
            'regularization': self.regularization
        }


class RidgeRegressionScratch(LinearRegressionScratch):
    """
    Ridge Regression (Linear Regression with L2 regularization).
    
    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    method : {'normal_equation', 'gradient_descent'}
        Optimization method.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        method: Literal['normal_equation', 'gradient_descent'] = 'normal_equation'
    ):
        super().__init__(method=method, regularization=alpha)
        self.alpha = alpha


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
        
    Returns
    -------
    metrics : dict
        Dictionary containing MSE, RMSE, MAE, and R².
    """
    residuals = y_true - y_pred
    
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Linear Regression from Scratch - Demo")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 200
    n_features = 3
    
    # True parameters
    true_weights = np.array([2.0, -1.0, 0.5])
    true_bias = 1.5
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate targets with noise
    noise = np.random.randn(n_samples) * 0.2
    y = X @ true_weights + true_bias + noise
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nData shape: X={X.shape}, y={y.shape}")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"True weights: {true_weights}")
    print(f"True bias: {true_bias}")
    
    # Method 1: Normal Equation
    print("\n" + "-" * 40)
    print("Method 1: Normal Equation")
    print("-" * 40)
    
    model_ne = LinearRegressionScratch(method='normal_equation')
    model_ne.fit(X_train, y_train)
    
    print(f"Learned weights: {model_ne.weights_}")
    print(f"Learned bias: {model_ne.bias_:.4f}")
    print(f"Train R²: {model_ne.score(X_train, y_train):.4f}")
    print(f"Test R²: {model_ne.score(X_test, y_test):.4f}")
    
    # Method 2: Gradient Descent
    print("\n" + "-" * 40)
    print("Method 2: Gradient Descent")
    print("-" * 40)
    
    model_gd = LinearRegressionScratch(method='gradient_descent')
    model_gd.fit(X_train, y_train, learning_rate=0.1, n_iterations=1000, verbose=False)
    
    print(f"Learned weights: {model_gd.weights_}")
    print(f"Learned bias: {model_gd.bias_:.4f}")
    print(f"Train R²: {model_gd.score(X_train, y_train):.4f}")
    print(f"Test R²: {model_gd.score(X_test, y_test):.4f}")
    print(f"Iterations: {len(model_gd.loss_history_)}")
    
    # Method 3: Ridge Regression
    print("\n" + "-" * 40)
    print("Method 3: Ridge Regression (α=0.1)")
    print("-" * 40)
    
    model_ridge = RidgeRegressionScratch(alpha=0.1)
    model_ridge.fit(X_train, y_train)
    
    print(f"Learned weights: {model_ridge.weights_}")
    print(f"Learned bias: {model_ridge.bias_:.4f}")
    print(f"Train R²: {model_ridge.score(X_train, y_train):.4f}")
    print(f"Test R²: {model_ridge.score(X_test, y_test):.4f}")
    
    # Detailed metrics
    print("\n" + "-" * 40)
    print("Detailed Metrics (Normal Equation)")
    print("-" * 40)
    
    y_pred = model_ne.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    
    for name, value in metrics.items():
        print(f"{name.upper()}: {value:.4f}")
