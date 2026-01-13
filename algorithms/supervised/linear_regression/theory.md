# Linear Regression: Mathematical Theory

This document provides a complete mathematical treatment of linear regression.

---

## 1. Problem Definition

**Goal:** Given training data $\{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ where $\mathbf{x}_i \in \mathbb{R}^d$ and $y_i \in \mathbb{R}$, find a linear function that predicts $y$ from $\mathbf{x}$.

**Model:**
$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_d x_d = \boldsymbol{\theta}^T \mathbf{x}$$

**Matrix Form:**
$$\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\theta}$$

where:
- $\mathbf{X} \in \mathbb{R}^{n \times (d+1)}$: Design matrix with bias column
- $\boldsymbol{\theta} \in \mathbb{R}^{d+1}$: Parameter vector
- $\hat{\mathbf{y}} \in \mathbb{R}^n$: Predictions

---

## 2. Assumptions

Linear regression assumes:

1. **Linearity:** $E[y|\mathbf{x}] = \boldsymbol{\theta}^T\mathbf{x}$
2. **Independence:** Observations are independent
3. **Homoscedasticity:** Constant variance: $\text{Var}(\epsilon) = \sigma^2$
4. **No multicollinearity:** Features are not perfectly correlated
5. **Normality (for inference):** $\epsilon \sim \mathcal{N}(0, \sigma^2)$

---

## 3. Loss Function

### Mean Squared Error (MSE)

$$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \frac{1}{n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|_2^2$$

**Why MSE?**
- Penalizes large errors more heavily
- Mathematically convenient (differentiable)
- Maximum likelihood estimate under Gaussian noise

### Expanded Form

$$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{n}(\mathbf{y} - \mathbf{X}\boldsymbol{\theta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\theta})$$

$$= \frac{1}{n}(\mathbf{y}^T\mathbf{y} - 2\boldsymbol{\theta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\theta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\theta})$$

---

## 4. Maximum Likelihood Derivation

Assume Gaussian noise: $y = \boldsymbol{\theta}^T\mathbf{x} + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2)$

**Likelihood:**
$$P(y|\mathbf{x}, \boldsymbol{\theta}) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y - \boldsymbol{\theta}^T\mathbf{x})^2}{2\sigma^2}\right)$$

**Log-likelihood (i.i.d. samples):**
$$\log P(\mathbf{y}|\mathbf{X}, \boldsymbol{\theta}) = \sum_{i=1}^{n} \log P(y_i|\mathbf{x}_i, \boldsymbol{\theta})$$

$$= -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \boldsymbol{\theta}^T\mathbf{x}_i)^2$$

**Maximizing log-likelihood ≡ Minimizing MSE**

---

## 5. Normal Equation (Closed-Form Solution)

### Derivation

Set gradient to zero:

$$\nabla_{\boldsymbol{\theta}}\mathcal{L} = \frac{1}{n}(-2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\theta}) = 0$$

$$\mathbf{X}^T\mathbf{X}\boldsymbol{\theta} = \mathbf{X}^T\mathbf{y}$$

### Solution

$$\boxed{\boldsymbol{\theta}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}}$$

### Conditions

- $\mathbf{X}^T\mathbf{X}$ must be invertible
- Requires $n \geq d$ and linearly independent features
- Computational complexity: $O(nd^2 + d^3)$

### Verification: Second Derivative

$$\nabla^2_{\boldsymbol{\theta}}\mathcal{L} = \frac{2}{n}\mathbf{X}^T\mathbf{X}$$

This is positive semi-definite, confirming a minimum.

---

## 6. Gradient Descent

When $d$ is large, computing and inverting $\mathbf{X}^T\mathbf{X}$ is expensive.

### Gradient

$$\nabla_{\boldsymbol{\theta}}\mathcal{L} = \frac{2}{n}\mathbf{X}^T(\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$$

### Update Rule

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \cdot \frac{2}{n}\mathbf{X}^T(\mathbf{X}\boldsymbol{\theta}_t - \mathbf{y})$$

### Stochastic Gradient Descent (SGD)

Use single sample or mini-batch:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \cdot 2(\hat{y}_i - y_i)\mathbf{x}_i$$

### Complexity

- Per iteration: $O(nd)$
- Total: $O(ndk)$ for $k$ iterations

---

## 7. Regularization

### Ridge Regression (L2)

Add L2 penalty to prevent large weights:

$$\mathcal{L}_{\text{ridge}} = \frac{1}{n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|_2^2 + \lambda\|\boldsymbol{\theta}\|_2^2$$

**Closed-form solution:**
$$\boldsymbol{\theta}^*_{\text{ridge}} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

**Benefits:**
- Always invertible (even when $n < d$)
- Shrinks weights toward zero
- Reduces variance at cost of bias

### Lasso Regression (L1)

$$\mathcal{L}_{\text{lasso}} = \frac{1}{n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|_2^2 + \lambda\|\boldsymbol{\theta}\|_1$$

**Properties:**
- Produces sparse solutions (feature selection)
- No closed-form solution (use coordinate descent)

### Elastic Net

$$\mathcal{L}_{\text{elastic}} = \frac{1}{n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|_2^2 + \lambda_1\|\boldsymbol{\theta}\|_1 + \lambda_2\|\boldsymbol{\theta}\|_2^2$$

---

## 8. Statistical Properties

### Gauss-Markov Theorem

Under assumptions 1-4, OLS estimator is **BLUE**:
- **B**est
- **L**inear
- **U**nbiased
- **E**stimator

### Unbiasedness

$$E[\hat{\boldsymbol{\theta}}] = E[(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}]$$
$$= (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T E[\mathbf{y}]$$
$$= (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{X}\boldsymbol{\theta}$$
$$= \boldsymbol{\theta}$$

### Variance

$$\text{Var}(\hat{\boldsymbol{\theta}}) = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$$

### Confidence Intervals

For coefficient $\theta_j$:

$$\hat{\theta}_j \pm t_{\alpha/2, n-d-1} \cdot \text{SE}(\hat{\theta}_j)$$

where $\text{SE}(\hat{\theta}_j) = \hat{\sigma}\sqrt{[(\mathbf{X}^T\mathbf{X})^{-1}]_{jj}}$

---

## 9. Evaluation Metrics

### Mean Squared Error (MSE)
$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

### Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\text{MSE}}$$

### Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

### R² (Coefficient of Determination)

$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

**Interpretation:**
- $R^2 = 1$: Perfect fit
- $R^2 = 0$: Same as predicting mean
- $R^2 < 0$: Worse than predicting mean

### Adjusted R²

$$R^2_{\text{adj}} = 1 - \frac{(1-R^2)(n-1)}{n-d-1}$$

Penalizes adding features that don't improve fit.

---

## 10. Computational Complexity

| Method | Time Complexity | Space Complexity |
|--------|-----------------|------------------|
| Normal Equation | $O(nd^2 + d^3)$ | $O(d^2)$ |
| Gradient Descent | $O(ndk)$ | $O(d)$ |
| SGD | $O(dk)$ per sample | $O(d)$ |

**When to use which:**
- $d \leq 10,000$: Normal equation
- $d > 10,000$ or streaming: Gradient descent

---

## 11. Common Issues

### Multicollinearity

When features are highly correlated, $\mathbf{X}^T\mathbf{X}$ is nearly singular.

**Detection:**
- Variance Inflation Factor (VIF)
- Correlation matrix

**Solutions:**
- Remove correlated features
- Use Ridge regression
- PCA for dimensionality reduction

### Overfitting

When model complexity exceeds data capacity.

**Detection:**
- Large gap between train/test error
- Very large coefficients

**Solutions:**
- Regularization (Ridge, Lasso)
- More training data
- Feature selection

---

## Summary

| Concept | Key Formula |
|---------|-------------|
| Model | $\hat{y} = \mathbf{X}\boldsymbol{\theta}$ |
| MSE Loss | $\mathcal{L} = \frac{1}{n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|_2^2$ |
| Normal Equation | $\boldsymbol{\theta}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ |
| Gradient | $\nabla\mathcal{L} = \frac{2}{n}\mathbf{X}^T(\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$ |
| Ridge | $\boldsymbol{\theta}^* = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$ |

---

## References

1. Bishop, "Pattern Recognition and Machine Learning," Chapter 3
2. Hastie et al., "The Elements of Statistical Learning," Chapter 3
3. Stanford CS229 Lecture Notes
