# Logistic Regression: Complete Mathematical Theory

A rigorous treatment of logistic regression covering classification from first principles.

---

## 1. Problem Definition

### What Problem Is Being Solved?

Given paired observations $(x_i, y_i)$ where $x_i \in \mathbb{R}^d$ and $y_i \in \{0, 1\}$, find a function that predicts the **probability** that $y = 1$ given $x$.

Unlike linear regression, the output is a probability: $p(y=1|x) \in [0, 1]$.

### Why Is This Problem Non-Trivial?

1. **Discrete outputs**: Can't directly minimize squared error on $\{0, 1\}$
2. **Probability constraints**: Output must be in $[0, 1]$
3. **Decision boundary**: Need to find where $p = 0.5$
4. **Class imbalance**: Often one class dominates
5. **Calibration**: Predicted probabilities should match frequencies

---

## 2. Mathematical Formulation

### The Logistic Function (Sigmoid)

To map $\mathbb{R} \to [0, 1]$, use the sigmoid:

$$\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$$

**Properties**:
- $\sigma(-\infty) = 0$, $\sigma(0) = 0.5$, $\sigma(+\infty) = 1$
- Symmetric: $\sigma(-z) = 1 - \sigma(z)$
- Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

### The Model

$$p(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^Tx + b)}}$$

Equivalently in log-odds (logit):
$$\log\frac{p}{1-p} = w^T x + b$$

The **log-odds is linear** in features — hence "logistic regression."

### Objective Function: Cross-Entropy Loss

For binary classification:
$$\mathcal{L}(w) = -\frac{1}{n}\sum_{i=1}^n \left[y_i \log p_i + (1-y_i)\log(1-p_i)\right]$$

Where $p_i = \sigma(w^Tx_i + b)$.

**Why This Loss?**
1. Negative log-likelihood under Bernoulli assumption
2. Convex in parameters
3. Penalizes confident wrong predictions severely

---

## 3. Why This Formulation?

### Assumptions That Justify Logistic Regression

1. **Linearity in log-odds**: $\log\frac{p}{1-p} = w^Tx$
2. **Independence**: Observations are i.i.d.
3. **No perfect separation**: Both classes appear in all regions
4. **Correct features**: All relevant predictors included

### What Breaks If Assumptions Fail?

| Violation | Consequence |
|-----------|-------------|
| Nonlinear boundary | Misclassification, poor calibration |
| Perfect separation | Coefficients → ±∞, no convergence |
| Dependent observations | Invalid standard errors |
| Missing features | Biased coefficients |

### Alternatives

| Method | When to Use |
|--------|-------------|
| Probit | Different link function, Gaussian latent |
| SVM | Maximum margin, kernel methods |
| Random Forest | Nonlinear, no probability assumptions |
| Neural Networks | Complex patterns, large data |

---

## 4. Derivation and Optimization

### Gradient Derivation

Let $p_i = \sigma(w^Tx_i)$ and $z_i = w^Tx_i$.

$$\frac{\partial \mathcal{L}}{\partial w} = -\frac{1}{n}\sum_{i=1}^n \left[\frac{y_i}{p_i} \cdot p_i(1-p_i) - \frac{1-y_i}{1-p_i} \cdot p_i(1-p_i)\right] x_i$$

Simplifying:
$$= -\frac{1}{n}\sum_{i=1}^n (y_i - p_i) x_i = \frac{1}{n}X^T(p - y)$$

**Same form as linear regression!** Just with $p_i$ instead of prediction.

### Hessian

$$H = \frac{1}{n}X^T D X$$

Where $D = \text{diag}(p_i(1-p_i))$ is diagonal with positive entries.

Since $H$ is positive semi-definite, **loss is convex** → unique global minimum.

### Optimization Algorithms

**Newton-Raphson (IRLS)**:
$$w_{t+1} = w_t - H^{-1}\nabla\mathcal{L}$$

Converges quadratically but expensive for large $d$.

**Gradient Descent**:
$$w_{t+1} = w_t - \eta \cdot \frac{1}{n}X^T(p - y)$$

Slower but scales better.

### Numerical Stability

**Problem**: $\log p$ and $\log(1-p)$ can be $-\infty$ if $p \in \{0, 1\}$.

**Solution**: Use log-sum-exp trick:
$$\log\sigma(z) = -\log(1 + e^{-z}) = z - \log(1 + e^z) \text{ (for large } z \text{)}$$

---

## 5. Geometric Interpretation

### Decision Boundary

Classification rule: Predict $y = 1$ if $p \geq 0.5$, equivalently $w^Tx + b \geq 0$.

The decision boundary is the **hyperplane** $\{x : w^Tx + b = 0\}$.

- In 2D: A line
- In 3D: A plane
- In $d$-D: A $(d-1)$-dimensional hyperplane

### Distance from Boundary

For point $x$, signed distance to boundary:
$$\frac{w^Tx + b}{\|w\|}$$

Positive = class 1 side, negative = class 0 side.

### Probability Contours

Lines of constant probability are parallel to the decision boundary:
$$p = 0.5 \implies w^Tx + b = 0$$
$$p = 0.9 \implies w^Tx + b = \log(9) \approx 2.2$$
$$p = 0.99 \implies w^Tx + b = \log(99) \approx 4.6$$

**Probability changes most rapidly perpendicular to boundary.**

---

## 6. Probabilistic Interpretation

### Generative Model

Assume:
$$y | x \sim \text{Bernoulli}(\sigma(w^Tx))$$

### Maximum Likelihood

Likelihood:
$$p(y | X, w) = \prod_{i=1}^n p_i^{y_i}(1-p_i)^{1-y_i}$$

Maximizing log-likelihood = Minimizing cross-entropy loss.

### Connection to Exponential Family

Bernoulli is an exponential family distribution:
$$p(y|\eta) = \exp(\eta y - \log(1 + e^\eta))$$

Where $\eta = \log\frac{p}{1-p}$ is natural parameter.

Logistic regression models $\eta = w^Tx$ — **linear in natural parameter**.

### Bayesian Logistic Regression

With prior $w \sim \mathcal{N}(0, \lambda^{-1}I)$:
- Posterior is not conjugate (no closed form)
- Use approximations: Laplace, variational, MCMC
- Gives uncertainty on predictions

---

## 7. Failure Modes and Limitations

### Perfect Separation

If a hyperplane perfectly separates the classes:
- Likelihood → 1 as $\|w\| \to \infty$
- Coefficients diverge, no maximum exists
- Solution: Regularization

### Class Imbalance

With 99% class 0, predicting all 0 gives 99% accuracy but is useless.

**Solutions**:
- Weighted loss
- SMOTE / oversampling
- Threshold adjustment
- Use AUC-ROC instead of accuracy

### Calibration Issues

Model may be discriminative but poorly calibrated:
- Predicts $p = 0.8$ but true frequency is 0.6

**Solutions**:
- Platt scaling
- Isotonic regression
- Temperature scaling

---

## 8. Scaling and Computational Reality

### Time Complexity

| Method | Per Iteration | Overall |
|--------|---------------|---------|
| Newton | $O(nd^2 + d^3)$ | Few iterations |
| L-BFGS | $O(nd)$ | More iterations |
| SGD | $O(d)$ | Many iterations |

### Memory

- Store $X$: $O(nd)$
- Newton Hessian: $O(d^2)$
- L-BFGS: $O(md)$ for $m$ history vectors

### When Each Method Wins

- **Newton**: $n, d < 10^4$
- **L-BFGS**: $d < 10^5$, $n$ any size
- **SGD**: $n > 10^6$, online learning

---

## 9. Real-World Deployment Considerations

### Feature Engineering

Real-world logistic regression often includes:
- One-hot encoded categoricals
- Polynomial features
- Interaction terms
- Log transforms for skewed features

### Regularization is Essential

In practice, always use L2 (Ridge) or L1 (Lasso):
$$\mathcal{L}_{reg} = \mathcal{L} + \lambda\|w\|^2$$

Prevents overfitting, handles multicollinearity.

### Interpretability

Logistic regression coefficients are interpretable:
$$\exp(w_j) = \text{odds ratio for unit increase in } x_j$$

**Example**: If $w_{\text{age}} = 0.05$, then $\exp(0.05) = 1.05$ — each year of age increases odds by 5%.

---

## 10. Comparison With Alternatives

### Logistic Regression vs SVM

| Aspect | Logistic | SVM |
|--------|----------|-----|
| Output | Probability | Decision only |
| Objective | Log-likelihood | Hinge loss |
| Solution | Uses all points | Only support vectors |
| Kernels | Possible but rare | Common |

### When Logistic Regression Wins

1. Probability estimates needed
2. Interpretability required
3. Linear boundary is appropriate
4. Feature engineering handles nonlinearity

### When Alternatives Win

- **Random Forest**: Nonlinear, automatic feature selection
- **XGBoost**: Better accuracy, less interpretable
- **Neural Net**: Complex patterns, large data

---

## 11. Mental Model Checkpoint

### Without Equations

Logistic regression draws a straight line (or flat surface) that best separates two groups. Points are assigned to groups based on which side of the line they fall on. The further from the line, the more confident the prediction.

### With Equations

$$p(y=1|x) = \sigma(w^Tx) = \frac{1}{1 + e^{-w^Tx}}$$

Minimize cross-entropy: $-\frac{1}{n}\sum_i y_i \log p_i + (1-y_i)\log(1-p_i)$

### Predict Behavior

1. **Adding regularization**: Coefficients shrink, overfitting decreases
2. **Perfect separation**: Without regularization, coefficients explode
3. **Threshold 0.3 instead of 0.5**: More positives predicted
4. **Scaling features**: Coefficients scale inversely, predictions unchanged
