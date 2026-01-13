"""
Linear Regression - scikit-learn Implementation

This module demonstrates linear regression using scikit-learn,
including best practices for preprocessing, validation, and evaluation.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Dict, Any


def basic_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[LinearRegression, Dict[str, float]]:
    """
    Train a basic linear regression model.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    test_size : float
        Fraction of data for testing.
    random_state : int
        Random seed for reproducibility.
        
    Returns
    -------
    model : LinearRegression
        Trained model.
    metrics : dict
        Performance metrics.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Create and train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
    }
    
    return model, metrics


def regularized_regression(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'ridge',
    alpha: float = 1.0,
    l1_ratio: float = 0.5
) -> Tuple[Any, Dict[str, float]]:
    """
    Train regularized regression models.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    method : str
        'ridge', 'lasso', or 'elasticnet'.
    alpha : float
        Regularization strength.
    l1_ratio : float
        ElasticNet mixing parameter (0 = Ridge, 1 = Lasso).
        
    Returns
    -------
    model : sklearn estimator
        Trained model.
    cv_scores : dict
        Cross-validation results.
    """
    # Select model
    if method == 'ridge':
        model = Ridge(alpha=alpha)
    elif method == 'lasso':
        model = Lasso(alpha=alpha)
    elif method == 'elasticnet':
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Cross-validation scores
    cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_neg_mse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    
    # Fit on full data
    model.fit(X, y)
    
    cv_scores = {
        'cv_r2_mean': cv_r2.mean(),
        'cv_r2_std': cv_r2.std(),
        'cv_rmse_mean': np.sqrt(-cv_neg_mse.mean()),
        'cv_rmse_std': np.sqrt(-cv_neg_mse).std(),
    }
    
    return model, cv_scores


def create_pipeline(
    degree: int = 1,
    regularization: str = 'none',
    alpha: float = 1.0
) -> Pipeline:
    """
    Create a preprocessing and regression pipeline.
    
    Parameters
    ----------
    degree : int
        Polynomial degree for feature expansion.
    regularization : str
        'none', 'ridge', 'lasso'.
    alpha : float
        Regularization strength.
        
    Returns
    -------
    pipeline : Pipeline
        sklearn Pipeline ready for fitting.
    """
    steps = []
    
    # Polynomial features (if degree > 1)
    if degree > 1:
        steps.append(('poly', PolynomialFeatures(degree=degree, include_bias=False)))
    
    # Standardization
    steps.append(('scaler', StandardScaler()))
    
    # Regression model
    if regularization == 'none':
        steps.append(('regressor', LinearRegression()))
    elif regularization == 'ridge':
        steps.append(('regressor', Ridge(alpha=alpha)))
    elif regularization == 'lasso':
        steps.append(('regressor', Lasso(alpha=alpha)))
    else:
        raise ValueError(f"Unknown regularization: {regularization}")
    
    return Pipeline(steps)


def compare_regularization(
    X: np.ndarray,
    y: np.ndarray,
    alphas: list = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare different regularization strengths.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    alphas : list
        Regularization strengths to compare.
        
    Returns
    -------
    results : dict
        Results for each alpha value.
    """
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    results = {}
    
    for alpha in alphas:
        # Ridge
        ridge = Ridge(alpha=alpha)
        ridge_scores = cross_val_score(ridge, X, y, cv=5, scoring='r2')
        
        # Lasso
        lasso = Lasso(alpha=alpha)
        lasso_scores = cross_val_score(lasso, X, y, cv=5, scoring='r2')
        
        results[alpha] = {
            'ridge_r2': ridge_scores.mean(),
            'ridge_std': ridge_scores.std(),
            'lasso_r2': lasso_scores.mean(),
            'lasso_std': lasso_scores.std(),
        }
    
    return results


def feature_importance(model, feature_names: list = None) -> Dict[str, float]:
    """
    Get feature importance from linear model coefficients.
    
    Parameters
    ----------
    model : sklearn linear model
        Fitted linear model.
    feature_names : list
        Names of features.
        
    Returns
    -------
    importance : dict
        Feature names mapped to absolute coefficient values.
    """
    coef = model.coef_
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(coef))]
    
    # Sort by absolute value
    importance = dict(zip(feature_names, np.abs(coef)))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    return importance


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Linear Regression with scikit-learn - Demo")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    true_coef = np.array([3.0, -2.0, 1.0, 0.0, 0.5])
    y = X @ true_coef + 2.0 + np.random.randn(n_samples) * 0.5
    
    feature_names = [f'x{i}' for i in range(n_features)]
    
    # Basic linear regression
    print("\n1. Basic Linear Regression")
    print("-" * 40)
    
    model, metrics = basic_linear_regression(X, y)
    
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Train R²: {metrics['train_r2']:.4f}")
    print(f"Test R²: {metrics['test_r2']:.4f}")
    
    # Feature importance
    print("\nFeature Importance:")
    importance = feature_importance(model, feature_names)
    for name, imp in importance.items():
        print(f"  {name}: {imp:.4f}")
    
    # Regularized regression
    print("\n2. Ridge Regression")
    print("-" * 40)
    
    ridge_model, ridge_scores = regularized_regression(X, y, method='ridge', alpha=1.0)
    
    print(f"Coefficients: {ridge_model.coef_}")
    print(f"CV R² (mean ± std): {ridge_scores['cv_r2_mean']:.4f} ± {ridge_scores['cv_r2_std']:.4f}")
    
    # Lasso (note sparsity)
    print("\n3. Lasso Regression")
    print("-" * 40)
    
    lasso_model, lasso_scores = regularized_regression(X, y, method='lasso', alpha=0.1)
    
    print(f"Coefficients: {lasso_model.coef_}")
    print(f"Non-zero coefficients: {np.sum(lasso_model.coef_ != 0)}")
    print(f"CV R² (mean ± std): {lasso_scores['cv_r2_mean']:.4f} ± {lasso_scores['cv_r2_std']:.4f}")
    
    # Compare regularization strengths
    print("\n4. Regularization Comparison")
    print("-" * 40)
    
    comparison = compare_regularization(X, y)
    
    print(f"{'Alpha':<10} {'Ridge R²':<15} {'Lasso R²'}")
    print("-" * 40)
    for alpha, scores in comparison.items():
        print(f"{alpha:<10.3f} {scores['ridge_r2']:.4f} ± {scores['ridge_std']:.2f}   "
              f"{scores['lasso_r2']:.4f} ± {scores['lasso_std']:.2f}")
    
    # Pipeline example
    print("\n5. Pipeline with Polynomial Features")
    print("-" * 40)
    
    # Create 1D example for polynomial
    X_1d = np.random.randn(100, 1)
    y_1d = 3 + 2*X_1d.ravel() + X_1d.ravel()**2 + np.random.randn(100)*0.3
    
    for degree in [1, 2, 3]:
        pipeline = create_pipeline(degree=degree, regularization='ridge', alpha=0.1)
        scores = cross_val_score(pipeline, X_1d, y_1d, cv=5, scoring='r2')
        print(f"Degree {degree}: CV R² = {scores.mean():.4f} ± {scores.std():.4f}")
