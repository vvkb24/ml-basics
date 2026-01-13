# Regularization: Complete Mathematical Theory

A rigorous treatment of regularization techniques for controlling model complexity and improving generalization.

---

## 1. Problem Definition

### What Problem Is Being Solved?

Prevent models from fitting noise in training data by imposing constraints or penalties on model complexity.

**Goal**: Find models that generalize well to unseen data.

### Why Is This Problem Non-Trivial?

1. **Bias-variance tradeoff**: Regularization increases bias to reduce variance
2. **Choosing strength**: Too much regularization → underfitting
3. **Different techniques**: Many approaches with different effects
4. **Hyperparameter tuning**: Regularization strength must be selected
5. **Implicit regularization**: Some methods regularize without explicit penalty

---

## 2. Mathematical Formulation

### General Framework

Regularized objective:
$$\mathcal{L}_{reg}(\theta) = \mathcal{L}(\theta) + \lambda \Omega(\theta)$$

Where:
- $\mathcal{L}(\theta)$: Data-fitting loss
- $\Omega(\theta)$: Regularization penalty
- $\lambda > 0$: Regularization strength

### L2 Regularization (Ridge / Weight Decay)

$$\Omega(\theta) = \frac{1}{2}\|\theta\|_2^2 = \frac{1}{2}\sum_j \theta_j^2$$

**Effect**: Shrinks weights toward zero, but not exactly zero.

### L1 Regularization (Lasso)

$$\Omega(\theta) = \|\theta\|_1 = \sum_j |\theta_j|$$

**Effect**: Encourages sparse solutions (some weights exactly zero).

### Elastic Net

Combination:
$$\Omega(\theta) = \alpha\|\theta\|_1 + \frac{1-\alpha}{2}\|\theta\|_2^2$$

Gets benefits of both L1 (sparsity) and L2 (stability).

---

## 3. Why This Formulation?

### Bayesian Interpretation

Regularization = Prior on parameters

**L2 = Gaussian prior**:
$$p(\theta) = \mathcal{N}(0, \lambda^{-1}I)$$

MAP estimate:
$$\theta_{MAP} = \arg\max_\theta p(D|\theta)p(\theta) = \arg\min_\theta \mathcal{L} + \lambda\|\theta\|_2^2$$

**L1 = Laplace prior**:
$$p(\theta) = \frac{\lambda}{2}\exp(-\lambda|\theta|)$$

### Geometric Interpretation

**L2**: Constraint $\|\theta\|_2 \leq c$ is a ball
**L1**: Constraint $\|\theta\|_1 \leq c$ is a diamond (corners touch axes)

Solution lies where loss contours touch constraint:
- L1: Often hits corner → sparse solution
- L2: Usually smooth point → non-sparse

### What Assumptions Are Required?

1. **Model is overparameterized**: Without enough params, regularization hurts
2. **Noise in data**: No point regularizing if data is perfect
3. **Prior belief**: Smaller weights are better (often reasonable)

---

## 4. Derivation and Optimization

### L2 Regularization (Ridge) Solution

For linear regression:
$$\min_w \|y - Xw\|^2 + \lambda\|w\|^2$$

Setting gradient to zero:
$$(X^TX + \lambda I)w = X^Ty$$
$$w^* = (X^TX + \lambda I)^{-1}X^Ty$$

**Key property**: Always invertible (even if $X^TX$ is singular)!

### L1 Optimization (Lasso)

No closed form. Use:
- **Coordinate descent**: Optimize one coordinate at a time
- **Proximal gradient descent**: Gradient step + soft thresholding
- **LARS**: Efficient path algorithm

**Soft thresholding**:
$$S_\lambda(z) = \text{sign}(z) \cdot \max(|z| - \lambda, 0)$$

### Why L1 Gives Sparsity

At optimum, for each coordinate:
$$\frac{\partial \mathcal{L}}{\partial \theta_j} + \lambda \cdot \text{sign}(\theta_j) = 0$$

If $|{\partial \mathcal{L}}/{\partial \theta_j}| < \lambda$ when $\theta_j = 0$, the weight stays at zero.

L2 doesn't have this property since $\partial\|\theta\|_2^2 / \partial\theta_j = 2\theta_j \to 0$ as $\theta_j \to 0$.

---

## 5. Geometric Interpretation

### Constrained vs Penalized Form

Equivalent formulations (for some $c \leftrightarrow \lambda$):
$$\min_\theta \mathcal{L}(\theta) + \lambda\Omega(\theta) \iff \min_\theta \mathcal{L}(\theta) \text{ s.t. } \Omega(\theta) \leq c$$

Larger $\lambda$ = smaller $c$ = more constrained.

### Effective Degrees of Freedom

For ridge regression:
$$\text{df}(\lambda) = \text{tr}(X(X^TX + \lambda I)^{-1}X^T) = \sum_j \frac{d_j^2}{d_j^2 + \lambda}$$

Where $d_j$ are singular values of $X$.

- $\lambda = 0$: df = number of parameters
- $\lambda \to \infty$: df → 0

### Shrinkage Direction

Ridge shrinks weights:
$$w_j^{ridge} = \frac{d_j^2}{d_j^2 + \lambda} w_j^{OLS}$$

Small singular values (noisy directions) get shrunk most.

---

## 6. Probabilistic Interpretation

### Prior Beliefs

| Regularization | Prior | Belief |
|----------------|-------|--------|
| L2 | Gaussian | Weights should be small |
| L1 | Laplace | Many weights should be zero |
| Elastic Net | Mixture | Some sparsity, some shrinkage |
| Dropout | Spike-and-slab | Features may be irrelevant |

### Uncertainty Quantification

Bayesian approach gives posterior:
$$p(\theta|D) \propto p(D|\theta)p(\theta)$$

Uncertainty in predictions comes from posterior variance.

### Marginalization vs Point Estimate

MAP estimate (regularized solution) is just the mode.

Full Bayesian: Integrate over posterior
$$p(y^*|x^*, D) = \int p(y^*|x^*, \theta)p(\theta|D)d\theta$$

More robust but computationally expensive.

---

## 7. Failure Modes and Limitations

### Over-regularization

Too much $\lambda$:
- Model too simple
- Cannot capture true pattern
- High bias

**Detection**: Poor training and test performance.

### Under-regularization

Too little $\lambda$:
- Overfitting
- Memorizes noise
- High variance

**Detection**: Good training, poor test performance.

### Feature Scaling

L1 and L2 are scale-dependent:
- Large-scale features penalized more
- Must standardize features first!

### Regularization of Different Layers

In deep learning:
- Different layers may need different $\lambda$
- Input layer features may be on different scales
- Often use same $\lambda$ everywhere (simplicity)

---

## 8. Scaling and Computational Reality

### Computational Cost

| Method | Cost |
|--------|------|
| Ridge (linear) | $O(d^3)$ or $O(nd^2)$ |
| Lasso | Iterative, $O(nd)$ per iteration |
| Neural net + L2 | Same as without (add gradient of penalty) |

### Choosing λ

**Cross-validation**:
1. Split data into K folds
2. For each $\lambda$, compute CV error
3. Choose $\lambda$ with lowest CV error

**One standard error rule**: Choose largest $\lambda$ within one SE of minimum.

### Regularization Path

Compute solutions for many $\lambda$ values:
- Start with large $\lambda$ (sparse/small solution)
- Decrease $\lambda$, warm-start from previous

Efficient algorithms (LARS for Lasso) compute entire path.

---

## 9. Real-World Deployment Considerations

### Neural Network Regularization

Beyond L2:
- **Dropout**: Randomly drop activations
- **Batch Normalization**: Implicit regularization
- **Data Augmentation**: Increases effective dataset size
- **Early Stopping**: Stop before overfitting

### Combining Regularizers

Often use multiple:
- Weight decay + Dropout + Data augmentation
- Each attacks overfitting differently

### Interpretable Sparsity

Lasso for feature selection:
- Non-zero weights = selected features
- Can use for interpretability
- But stability is low (small data change → different features)

**Stability selection**: Run Lasso many times with subsampling, keep frequently selected features.

### Transfer Learning

Pre-trained weights are implicit regularization:
- Start from good solution space
- Fine-tuning with small learning rate = staying close
- Equivalent to strong prior on weights

---

## 10. Comparison With Alternatives

### Regularization vs Model Selection

| Approach | How It Controls Complexity |
|----------|---------------------------|
| L1/L2 regularization | Continuous penalty |
| Cross-validation | Discrete model choices |
| Early stopping | Limit training time |
| Ensemble averaging | Average many models |

### L1 vs L2 Summary

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|------------|------------|
| Sparsity | Yes | No |
| Solution | Not unique (sometimes) | Unique |
| Grouped features | Selects one | Shares weight |
| Optimization | Harder | Easier |
| Feature selection | Built-in | No |

### When to Use What

- **L2**: Default choice, stable, fast
- **L1**: Feature selection needed, interpretability
- **Elastic Net**: Many correlated features
- **Dropout**: Deep learning
- **Early stopping**: When you see overfitting

---

## 11. Mental Model Checkpoint

### Without Equations

Regularization is like putting a budget on complexity:
- "You can fit the data, but don't use big weights"
- "Use as few features as possible (L1)"
- "Keep weights small on average (L2)"

The model balances fitting data against staying simple.

**Analogy**: Explaining something simply vs with jargon — prefer simple unless complexity is needed.

### With Equations

$$\min_\theta \underbrace{\mathcal{L}(\theta)}_{\text{Fit data}} + \lambda\underbrace{\Omega(\theta)}_{\text{Stay simple}}$$

L2: $\Omega = \|\theta\|_2^2$, L1: $\Omega = \|\theta\|_1$

### Predict Behavior

1. **λ = 0**: Same as unregularized (overfitting risk)
2. **λ → ∞**: All weights → 0 (underfitting)
3. **L1 increasing λ**: More weights become exactly zero
4. **L2 increasing λ**: All weights shrink, none exactly zero
5. **Double the data**: Can reduce λ (less regularization needed)

---

## References

### Foundational
- Hoerl & Kennard (1970) - Ridge regression
- Tibshirani (1996) - Lasso
- Zou & Hastie (2005) - Elastic net

### Modern
- Srivastava et al. (2014) - Dropout
- Goodfellow et al. - *Deep Learning* Chapter 7

### Theory
- Hastie, Tibshirani, Friedman - *ESL* Chapter 3
- Bühlmann & van de Geer (2011) - Statistics for high-dimensional data
