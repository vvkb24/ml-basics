# Generalization in Machine Learning

Generalization is the ability of a model to perform well on unseen data.

---

## Training vs. Generalization Error

**Training Error:**
$$R_{\text{train}}(\hat{f}) = \frac{1}{n}\sum_{i=1}^{n} L(y_i, \hat{f}(x_i))$$

**Generalization Error (True Risk):**
$$R(\hat{f}) = \mathbb{E}_{(x,y) \sim P}[L(y, \hat{f}(x))]$$

**Goal:** Minimize generalization error, not training error.

---

## The Generalization Gap

$$\text{Gap} = R(\hat{f}) - R_{\text{train}}(\hat{f})$$

Large gap → Overfitting

---

## Overfitting vs. Underfitting

| | Training Error | Test Error | Description |
|---|---|---|---|
| **Underfitting** | High | High | Model too simple |
| **Good fit** | Low | Low | Model appropriate |
| **Overfitting** | Very Low | High | Model too complex |

---

## Learning Curves

Plotting error vs. training set size:

**Learning Curve Behavior:**

| Training Set Size | Training Error | Test Error | Gap |
|-------------------|----------------|------------|-----|
| Very small | Low (overfits) | High | Large |
| Medium | Increasing | Decreasing | Moderate |
| Large | Asymptotic | Asymptotic | Small |

**Key patterns:**
- Training error **starts low** (easy to fit few points) and **increases** toward asymptote
- Test error **starts high** and **decreases** toward asymptote
- As n → ∞, both converge (gap closes)

**Interpretation:**
- Converging curves → Good generalization
- Large gap → More data might help
- Both high → Need more complex model

---

## Capacity and VC Dimension

### Model Capacity

The ability of a model to fit a wide variety of functions.

### VC Dimension

The **Vapnik-Chervonenkis dimension** is the largest number of points that can be shattered (classified in all possible ways).

**Examples:**
- Linear classifier in 2D: VC = 3
- Linear classifier in d dimensions: VC = d + 1

### Generalization Bound

With probability at least $1 - \delta$:

$$R(\hat{f}) \leq R_{\text{train}}(\hat{f}) + O\left(\sqrt{\frac{\text{VC}}{n}\log\frac{n}{\text{VC}} + \frac{1}{n}\log\frac{1}{\delta}}\right)$$

**Insight:** Generalization gap decreases with $n$ and increases with model complexity.

---

## PAC Learning

**Probably Approximately Correct (PAC):**

A concept class is PAC learnable if there exists an algorithm that, for any:
- $\epsilon > 0$ (accuracy)
- $\delta > 0$ (confidence)

produces hypothesis $\hat{f}$ such that:
$$P(R(\hat{f}) - R(f^*) \leq \epsilon) \geq 1 - \delta$$

with polynomial sample complexity.

---

## Double Descent

Modern observation in deep learning:

**The Three Regimes:**

| Regime | Model Size | Behavior |
|--------|------------|----------|
| **Classical** | Small → Medium | U-shaped curve (bias-variance tradeoff) |
| **Interpolation threshold** | ≈ Training samples | Peak test error |
| **Overparameterized** | >> Training samples | Error decreases again |

**Why it happens:**
- At interpolation threshold: Model barely fits data, very sensitive to noise
- Beyond threshold: Many solutions exist, optimization finds "simpler" ones

After the interpolation threshold, larger models may generalize better.

---

## Practical Strategies

### Data-Level
- Collect more diverse data
- Data augmentation
- Ensure representative splits

### Model-Level
- Choose appropriate complexity
- Use regularization
- Ensemble methods

### Training-Level
- Early stopping
- Cross-validation
- Hyperparameter tuning

---

## Cross-Validation

**K-Fold Cross-Validation:**

1. Split data into K folds
2. For each fold:
   - Train on K-1 folds
   - Evaluate on held-out fold
3. Average the K scores

**Provides:** Robust estimate of generalization error

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"Mean: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## Key Takeaways

1. **Training error ≠ Test error**
2. **More data improves generalization** (usually)
3. **Regularization helps** by constraining model capacity
4. **Validate on held-out data** for honest evaluation
5. **Double descent challenges classical wisdom** for overparameterized models
