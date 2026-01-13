# Bias-Variance Tradeoff

The bias-variance tradeoff is fundamental to understanding model generalization.

---

## The Problem

Given data $\mathcal{D} = \{(x_i, y_i)\}$, we want to learn $f$ such that $y \approx f(x)$.

The **expected prediction error** for a new point:

$$\mathbb{E}[(y - \hat{f}(x))^2]$$

---

## Decomposition

$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

### Bias

$$\text{Bias}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)] - f(x)$$

**High bias means:**
- Model is too simple
- Systematic error in predictions
- **Underfitting**

### Variance

$$\text{Var}[\hat{f}(x)] = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]$$

**High variance means:**
- Model is too sensitive to training data
- Predictions vary significantly with different datasets
- **Overfitting**

### Irreducible Error

$$\sigma^2 = \text{Var}[\epsilon]$$

Noise inherent in the data. Cannot be reduced.

---

## The Tradeoff

| Model Complexity | Bias | Variance |
|------------------|------|----------|
| Low (simple) | High | Low |
| High (complex) | Low | High |

**Goal:** Find the sweet spot that minimizes total error.

---

## Visual Intuition

```
Error
  │
  │    ╲        Total Error
  │     ╲      ╱
  │      ╲    ╱
  │       ╲  ╱
  │    ────╳────  ← Optimal
  │   ╱    ╲
  │  ╱ Bias  ╲ Variance
  │ ╱          ╲
  └────────────────► Model Complexity
```

---

## Examples

### Linear Regression

- **High bias:** Assumes linear relationship
- **Low variance:** Stable across samples
- Use when: True relationship is approximately linear

### Deep Neural Network

- **Low bias:** Can approximate complex functions
- **High variance:** Sensitive to training data
- Use when: Large data, complex patterns

### Decision Tree (Deep)

- **Low bias:** Can fit any data perfectly
- **High variance:** Small data changes → different tree
- Fix: Ensemble methods (Random Forest)

---

## Strategies

### To Reduce Bias
- Use more complex model
- Add features
- Reduce regularization
- Train longer (neural nets)

### To Reduce Variance
- Get more training data
- Use simpler model
- Add regularization
- Ensemble methods
- Dropout (neural nets)
- Cross-validation for model selection

---

## Mathematical Derivation

For regression with squared loss:

Let $y = f(x) + \epsilon$ where $\mathbb{E}[\epsilon] = 0$, $\text{Var}(\epsilon) = \sigma^2$.

$$\begin{align}
\mathbb{E}[(y - \hat{f})^2] &= \mathbb{E}[(f + \epsilon - \hat{f})^2] \\
&= \mathbb{E}[(f - \hat{f})^2] + 2\mathbb{E}[(f - \hat{f})\epsilon] + \mathbb{E}[\epsilon^2] \\
&= \mathbb{E}[(f - \hat{f})^2] + \sigma^2 \\
&= \mathbb{E}[(f - \mathbb{E}[\hat{f}] + \mathbb{E}[\hat{f}] - \hat{f})^2] + \sigma^2 \\
&= (f - \mathbb{E}[\hat{f}])^2 + \mathbb{E}[(\hat{f} - \mathbb{E}[\hat{f}])^2] + \sigma^2 \\
&= \text{Bias}^2 + \text{Variance} + \sigma^2
\end{align}$$

---

## Key Takeaways

1. **No free lunch:** Reducing bias typically increases variance (and vice versa)
2. **More data helps:** Reduces variance without increasing bias
3. **Regularization:** Trades some bias for reduced variance
4. **Validation is crucial:** Use held-out data to detect overfitting

---

## Further Reading

- "The Elements of Statistical Learning" Chapter 7
- Stanford CS229: Learning Theory notes
