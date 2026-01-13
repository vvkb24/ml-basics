# Linear Regression: Complete Mathematical Theory

A rigorous, research-level treatment of linear regression covering all theoretical perspectives required for deep understanding.

---

## 1. Problem Definition

### What Problem Is Being Solved?

Given paired observations $(x_i, y_i)$ where $x_i \in \mathbb{R}^d$ and $y_i \in \mathbb{R}$, find a function $f: \mathbb{R}^d \to \mathbb{R}$ that predicts $y$ from $x$.

**The Linear Assumption**: We restrict $f$ to be linear:
$$f(x) = w^T x + b = \sum_{j=1}^d w_j x_j + b$$

### Why Is This Problem Non-Trivial?

1. **Noise**: Observations contain measurement error: $y_i = f(x_i) + \epsilon_i$
2. **Finite Data**: We have $n$ samples but want to generalize to unseen data
3. **Dimensionality**: When $d > n$, infinitely many solutions exist
4. **Model Misspecification**: True relationship may be nonlinear
5. **Multicollinearity**: Features may be correlated, making attribution ambiguous

---

## 2. Mathematical Formulation

### Notation

| Symbol | Meaning |
|--------|---------|
| $X \in \mathbb{R}^{n \times d}$ | Design matrix (n samples, d features) |
| $y \in \mathbb{R}^n$ | Target vector |
| $w \in \mathbb{R}^d$ | Weight vector |
| $b \in \mathbb{R}$ | Bias (intercept) |
| $\hat{y} = Xw + b$ | Predictions |
| $\epsilon = y - \hat{y}$ | Residuals |

### Objective Function: Ordinary Least Squares (OLS)

$$\mathcal{L}(w) = \frac{1}{2n}\sum_{i=1}^n (y_i - w^T x_i)^2 = \frac{1}{2n}\|y - Xw\|_2^2$$

**Why Squared Error?**
1. Differentiable everywhere (unlike absolute error)
2. Penalizes large errors more heavily
3. Has closed-form solution
4. Maximum likelihood under Gaussian noise assumption

### The Normal Equation

Setting $\nabla_w \mathcal{L} = 0$:

$$\nabla_w \mathcal{L} = -\frac{1}{n}X^T(y - Xw) = 0$$

$$X^T Xw = X^T y$$

$$w^* = (X^T X)^{-1} X^T y$$

This is the **optimal weight vector** (if $X^TX$ is invertible).

---

## 3. Why This Formulation?

### Assumptions That Justify OLS

1. **Linearity**: $\mathbb{E}[y|x] = w^Tx$ (correct model specification)
2. **Independence**: Observations are independent
3. **Homoscedasticity**: $\text{Var}(\epsilon_i) = \sigma^2$ (constant variance)
4. **No multicollinearity**: $X^TX$ is invertible (full column rank)
5. **Gaussian errors** (for inference): $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$

### What Breaks If Assumptions Fail?

| Assumption | Violation | Consequence |
|------------|-----------|-------------|
| Linearity | Nonlinear relationship | Systematic bias, poor predictions |
| Independence | Time series, clustering | Underestimated standard errors |
| Homoscedasticity | Heteroscedastic errors | Inefficient estimates |
| No multicollinearity | Correlated features | Unstable coefficients, large variance |
| Gaussian errors | Heavy-tailed noise | Outliers dominate solution |

### Alternatives to OLS

| Method | When to Use |
|--------|-------------|
| Ridge Regression | Multicollinearity, high dimensions |
| Lasso | Feature selection needed |
| Huber Loss | Outliers present |
| Weighted LS | Heteroscedasticity |
| GLS | Correlated errors |

---

## 4. Derivation and Optimization

### Gradient Derivation

$$\mathcal{L}(w) = \frac{1}{2n}(y - Xw)^T(y - Xw)$$

Expanding:
$$= \frac{1}{2n}(y^Ty - 2y^TXw + w^TX^TXw)$$

Taking gradient:
$$\nabla_w \mathcal{L} = \frac{1}{n}(-X^Ty + X^TXw) = \frac{1}{n}X^T(Xw - y)$$

### Hessian (Second Derivative)

$$H = \nabla^2_w \mathcal{L} = \frac{1}{n}X^TX$$

Since $X^TX$ is positive semi-definite, $\mathcal{L}$ is **convex** → unique global minimum.

### Numerical Stability Concerns

**Problem**: Computing $(X^TX)^{-1}$ directly is numerically unstable.

**Solution**: Use QR decomposition or SVD.

**QR Approach**:
$$X = QR \implies w^* = R^{-1}Q^Ty$$

**SVD Approach**:
$$X = U\Sigma V^T \implies w^* = V\Sigma^{-1}U^Ty$$

**Condition Number**: $\kappa(X^TX) = \kappa(X)^2$. If $\kappa > 10^6$, expect numerical issues.

### Gradient Descent Alternative

When $n$ or $d$ is very large, use iterative methods:

$$w_{t+1} = w_t - \eta \nabla_w \mathcal{L} = w_t - \frac{\eta}{n}X^T(Xw_t - y)$$

**Convergence Rate**: $O(1/t)$ for convex, $O(\exp(-t))$ with strong convexity.

---

## 5. Geometric Interpretation

### The Column Space View

The prediction $\hat{y} = Xw$ is a linear combination of columns of $X$:
$$\hat{y} = w_1 x_{\cdot 1} + w_2 x_{\cdot 2} + \cdots + w_d x_{\cdot d}$$

**Key Insight**: $\hat{y}$ lives in the column space of $X$, which is a $d$-dimensional subspace of $\mathbb{R}^n$.

### Projection Interpretation

OLS finds the orthogonal projection of $y$ onto $\text{col}(X)$:

$$\hat{y} = X(X^TX)^{-1}X^Ty = Py$$

Where $P = X(X^TX)^{-1}X^T$ is the **projection matrix** (also called "hat matrix").

**Properties of P**:
- $P^2 = P$ (idempotent)
- $P^T = P$ (symmetric)
- $\text{rank}(P) = d$
- $Py$ is closest point to $y$ in $\text{col}(X)$

### Residual Orthogonality

The residual $\epsilon = y - \hat{y}$ is orthogonal to the column space:
$$X^T\epsilon = X^T(y - Xw^*) = 0$$

**Geometric Picture**:
```
y = target vector in R^n
↓
Project onto col(X)
↓
ŷ = closest point in subspace
↓  
ε = y - ŷ (perpendicular to subspace)
```

### Overfitting Geometrically

- **Underfitting**: $d$ too small → subspace can't capture $y$
- **Good fit**: $d$ appropriate → subspace approximates $y$ well
- **Overfitting**: $d = n$ → perfect fit but no generalization

When $d = n$ and $X$ is invertible:
$$\hat{y} = X(X^TX)^{-1}X^Ty = XX^{-1}(X^T)^{-1}X^Ty = y$$

Zero training error but terrible on new data!

---

## 6. Probabilistic Interpretation

### Generative Model

Assume data is generated by:
$$y = Xw^{true} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$

Then:
$$y | X, w \sim \mathcal{N}(Xw, \sigma^2 I)$$

### Maximum Likelihood Estimation

The likelihood:
$$p(y | X, w, \sigma^2) = \prod_{i=1}^n \mathcal{N}(y_i | w^Tx_i, \sigma^2)$$

Log-likelihood:
$$\log p = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_i(y_i - w^Tx_i)^2$$

Maximizing over $w$ gives:
$$w_{MLE} = \arg\min_w \|y - Xw\|^2 = (X^TX)^{-1}X^Ty$$

**Key Insight**: OLS = MLE under Gaussian noise assumption!

### Bayesian Linear Regression

**Prior**: $w \sim \mathcal{N}(0, \lambda^{-1}I)$

**Posterior**:
$$p(w | X, y) \propto p(y | X, w) p(w)$$
$$w | X, y \sim \mathcal{N}(\mu_{post}, \Sigma_{post})$$

Where:
$$\Sigma_{post} = (\lambda I + \sigma^{-2}X^TX)^{-1}$$
$$\mu_{post} = \sigma^{-2}\Sigma_{post}X^Ty$$

**Interpretation**:
- Posterior mean = Ridge regression solution
- Posterior gives uncertainty estimates on $w$
- Predictive distribution gives uncertainty on $\hat{y}$

### What Uncertainty Is Ignored in OLS?

1. **Model uncertainty**: Is linear model correct?
2. **Parameter uncertainty**: How confident are we in $w$?
3. **Heteroscedasticity**: Does variance change with $x$?
4. **Epistemic vs aleatoric**: What can be learned vs what is noise?

---

## 7. Failure Modes and Limitations

### Distribution Shift

**Training**: Learn $w$ from $p_{train}(x, y)$
**Test**: Apply to $p_{test}(x, y) \neq p_{train}$

If test distribution differs:
- Covariate shift: $p(x)$ changes
- Label shift: $p(y)$ changes
- Concept drift: $p(y|x)$ changes

**OLS has no mechanism to detect or adapt to shift.**

### Data Sparsity

When $n < d$:
- $X^TX$ is rank-deficient (not invertible)
- Infinitely many solutions exist
- Need regularization (Ridge, Lasso)

### Adversarial and Pathological Cases

1. **Outliers**: Single outlier can dramatically shift $w$
2. **Leverage points**: Outliers in $x$-space have outsized influence
3. **Perfect multicollinearity**: $X^TX$ singular, no unique solution

**Robustness**: OLS has breakdown point of $1/n$ — one bad point can destroy the fit.

### Nonlinearity

If true relationship is:
$$y = f(x) + \epsilon, \quad f \text{ nonlinear}$$

Then OLS approximates $f$ with best linear fit, but:
- Systematic bias in predictions
- Residuals show patterns (not i.i.d.)
- Model diagnostics reveal failure

---

## 8. Scaling and Computational Reality

### Time Complexity

| Method | Time | When to Use |
|--------|------|-------------|
| Normal equation | $O(nd^2 + d^3)$ | $d$ small |
| QR decomposition | $O(nd^2)$ | $d$ moderate |
| Gradient descent | $O(ndt)$ | $d$ or $n$ large |
| Stochastic GD | $O(dt)$ | $n$ very large |

### Memory Complexity

| Operation | Memory |
|-----------|--------|
| Store $X$ | $O(nd)$ |
| Compute $X^TX$ | $O(d^2)$ |
| SGD (mini-batch) | $O(bd)$ |

### What Bottlenecks First?

- **Small data** ($n < 10^4$): Computation is instant
- **Medium data** ($n < 10^6$): Normal equation works
- **Large data** ($n > 10^6$): Need SGD or online methods
- **High dimensions** ($d > 10^4$): Regularization essential

### Parallelization

OLS is embarrassingly parallel for:
- Matrix-vector products ($X^T y$)
- Mini-batch gradient computation

Distributed algorithms: Hogwild, Parameter Server, AllReduce

---

## 9. Real-World Deployment Considerations

### Latency vs Accuracy Trade-offs

| Scenario | Priority | Approach |
|----------|----------|----------|
| Real-time pricing | Latency | Pre-compute, simple features |
| Scientific analysis | Accuracy | Full model, confidence intervals |
| Recommendation | Both | Approximate inference |

### Feature Engineering Reality

Real production models often have:
- Hundreds of raw features
- Thousands after one-hot encoding
- Feature interactions, polynomials
- Missing value handling

**OLS rarely used directly** — but understanding it is foundational.

### Data Quality Issues

1. **Missing values**: Imputation or indicator variables
2. **Measurement error**: Errors-in-variables models
3. **Class imbalance**: Weighted regression
4. **Temporal dependence**: Time series methods

### Interpretability Requirements

Linear regression is often chosen for **interpretability**:
- Coefficient $w_j$ = effect of feature $j$ on $y$
- Easy to explain to stakeholders
- Regulatory requirements (finance, healthcare)

But interpretation requires:
- Proper scaling of features
- Understanding of confounding
- Causal assumptions (often violated)

---

## 10. Comparison With Alternatives

### When Linear Regression Wins

1. **True relationship is linear** (rare but possible)
2. **Interpretability required** (regulated industries)
3. **Data is limited** (complex models overfit)
4. **Baseline needed** (always start with linear)
5. **Speed matters** (real-time inference)

### When Alternatives Win

| Alternative | When It Wins |
|-------------|--------------|
| Ridge/Lasso | High dimensions, multicollinearity |
| Decision Trees | Nonlinear, discontinuous |
| Neural Networks | Massive data, complex patterns |
| Gaussian Processes | Uncertainty quantification needed |
| XGBoost | Tabular data, competitions |

### Historical Context

- **1805**: Legendre publishes least squares
- **1809**: Gauss proves optimality properties
- **1821**: Gauss-Markov theorem
- **1970s**: Ridge regression (Hoerl & Kennard)
- **1996**: Lasso (Tibshirani)
- **2000s**: Elastic net, group lasso

**Why Linear Regression Persists**:
- Complete theoretical understanding
- Closed-form solution
- Foundation for understanding complex methods
- Surprisingly effective baseline

---

## 11. Mental Model Checkpoint

### Explain Without Equations

Linear regression finds the best straight line (or flat surface in higher dimensions) through your data points. "Best" means the line that minimizes the total squared distance from points to the line. This line then predicts values for new inputs.

**Analogy**: Fitting a ruler through a scatter of dots to predict where future dots might fall.

### Explain Using Only Equations

$$w^* = \arg\min_w \|y - Xw\|_2^2 = (X^TX)^{-1}X^Ty$$

This is the orthogonal projection of $y$ onto $\text{col}(X)$, equivalent to MLE under $y \sim \mathcal{N}(Xw, \sigma^2I)$.

### Predict Behavior Before Running Code

1. **Adding a perfectly correlated feature**: Coefficients become unstable, numerical issues
2. **Doubling all y values**: Coefficients double
3. **Adding noise to x**: Coefficients shrink toward zero (attenuation bias)
4. **Removing outlier**: Large change in coefficients possible
5. **Increasing n with same d**: Variance of estimates decreases as $O(1/n)$

---

## References

### Foundational
- Hastie, Tibshirani, Friedman - *The Elements of Statistical Learning* (Ch. 3)
- Bishop - *Pattern Recognition and Machine Learning* (Ch. 3)

### Historical
- Legendre (1805) - Original least squares paper
- Gauss (1809) - *Theoria Motus* - Connection to astronomy

### Advanced
- Hoerl & Kennard (1970) - Ridge regression
- Tibshirani (1996) - Lasso
- Zou & Hastie (2005) - Elastic net
