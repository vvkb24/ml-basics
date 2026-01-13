# Regularization

Regularization techniques prevent overfitting by constraining model complexity.

---

## The Problem

Without regularization, models may:
- Fit noise in training data
- Have large, unstable weights
- Generalize poorly

---

## L2 Regularization (Ridge / Weight Decay)

### Formulation

Add squared norm penalty:

$$\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda \|\mathbf{w}\|_2^2 = \mathcal{L} + \lambda \sum_{j} w_j^2$$

### Effect on Weights

Gradient update:
$$w \leftarrow w - \eta\left(\frac{\partial \mathcal{L}}{\partial w} + 2\lambda w\right)$$

Weights shrink toward zero (but rarely reach zero).

### Linear Regression (Ridge)

$$\hat{\mathbf{w}} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

**Benefit:** Always invertible (stabilizes solution).

---

## L1 Regularization (Lasso)

### Formulation

Add absolute value penalty:

$$\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda \|\mathbf{w}\|_1 = \mathcal{L} + \lambda \sum_{j} |w_j|$$

### Effect on Weights

- Produces **sparse** solutions (many weights exactly zero)
- Automatic **feature selection**

### Comparison

| | L1 (Lasso) | L2 (Ridge) |
|---|---|---|
| Sparsity | Yes | No |
| Correlated features | Picks one | Shrinks all |
| Computational | More complex | Closed form |
| Use case | Feature selection | Multicollinearity |

---

## Elastic Net

Combine L1 and L2:

$$\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|_2^2$$

Gets benefits of both approaches.

---

## Dropout (Neural Networks)

### Algorithm

During training, randomly set activations to zero:

$$\tilde{h}_i = \begin{cases} 0 & \text{with probability } p \\ h_i / (1-p) & \text{otherwise} \end{cases}$$

### Interpretation
- Ensemble of exponentially many sub-networks
- Prevents co-adaptation of neurons
- At test time: use all neurons (already scaled)

### Common Values
- Typical dropout rate: 0.2 - 0.5
- Higher for larger layers

---

## Early Stopping

Stop training when validation error starts increasing:

```
  Error
  │  ╲ Training
  │   ╲ ─────────────
  │    ──────────────────
  │      ╱ Validation
  │     ╱
  │────╱─── ← Stop here
  │   ╱
  └────────────────────────► Epochs
```

**Equivalent to:** Implicit regularization on weight magnitude.

---

## Batch Normalization

Normalize activations within mini-batch:

$$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$

Then apply learnable scale and shift:
$$y_i = \gamma \hat{x}_i + \beta$$

### Benefits
- Allows higher learning rates
- Reduces internal covariate shift
- Has regularization effect
- Accelerates training

---

## Data Augmentation

Create additional training samples via transformations:

**Images:**
- Rotation, flipping
- Cropping, scaling
- Color jittering
- Cutout, mixup

**Text:**
- Synonym replacement
- Back-translation
- Random insertion/deletion

**Effect:** Increases effective training set size.

---

## Weight Constraints

### Max-Norm Constraint

Constrain weight vector norm:
$$\|\mathbf{w}\| \leq c$$

If exceeded after update, rescale:
$$\mathbf{w} \leftarrow \mathbf{w} \cdot \frac{c}{\|\mathbf{w}\|}$$

---

## Bayesian Interpretation

L2 regularization is equivalent to **Gaussian prior** on weights:

$$P(\mathbf{w}) = \mathcal{N}(0, \frac{1}{2\lambda}\mathbf{I})$$

L1 regularization is equivalent to **Laplace prior**:

$$P(\mathbf{w}) \propto \exp(-\lambda\|\mathbf{w}\|_1)$$

MAP estimation with prior → Regularized loss.

---

## Choosing λ

### Grid Search / Random Search

Try multiple values, evaluate on validation set.

### Cross-Validation

```python
from sklearn.linear_model import RidgeCV

model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
model.fit(X, y)
print(f"Best alpha: {model.alpha_}")
```

### Typical Range

Start with: $\lambda \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1\}$

---

## Summary

| Technique | When to Use |
|-----------|-------------|
| L2 (Ridge) | Default for linear models, neural nets |
| L1 (Lasso) | Feature selection needed |
| Dropout | Deep neural networks |
| Early stopping | All iterative methods |
| Batch norm | Deep networks |
| Data augmentation | Limited data, especially images |

---

## Key Takeaways

1. Regularization trades bias for reduced variance
2. L1 induces sparsity, L2 shrinks uniformly
3. Multiple techniques can be combined
4. Always tune regularization strength on validation data
