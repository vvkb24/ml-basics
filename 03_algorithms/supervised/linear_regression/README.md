# Linear Regression

A complete, mathematically rigorous treatment of linear regression.

## Overview

Linear regression models the relationship between features and a continuous target as a linear function.

## Contents

| File | Description |
|------|-------------|
| [theory.md](./theory.md) | Mathematical foundations and derivations |
| [scratch.py](./scratch.py) | NumPy implementation from scratch |
| [sklearn_impl.py](./sklearn_impl.py) | scikit-learn implementation |
| [experiments.ipynb](./experiments.ipynb) | Visualizations and experiments |

## Quick Start

```python
from scratch import LinearRegressionScratch
import numpy as np

# Generate data
X = np.random.randn(100, 3)
y = X @ np.array([2, -1, 0.5]) + 1 + np.random.randn(100) * 0.1

# Train
model = LinearRegressionScratch(method='gradient_descent')
model.fit(X, y, learning_rate=0.01, n_iterations=1000)

# Predict
predictions = model.predict(X)
print(f"R² Score: {model.score(X, y):.4f}")
```

## Key Equations

**Model:**
$$\hat{y} = \mathbf{X}\boldsymbol{\theta}$$

**Loss (MSE):**
$$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|_2^2$$

**Normal Equation:**
$$\boldsymbol{\theta}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

**Gradient Descent Update:**
$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \cdot \frac{1}{n}\mathbf{X}^T(\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$$

## Learning Objectives

After studying this module, you should be able to:

1. Derive the closed-form solution using MLE
2. Implement gradient descent optimization
3. Understand the assumptions of linear regression
4. Apply regularization (Ridge, Lasso)
5. Evaluate model performance correctly
