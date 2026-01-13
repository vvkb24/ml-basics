# Support Vector Machines: Complete Mathematical Theory

A rigorous treatment of SVMs covering maximum margin classification and kernel methods.

---

## 1. Problem Definition

### What Problem Is Being Solved?

Find a hyperplane that separates classes with **maximum margin** — the largest possible gap between the decision boundary and nearest training points.

### Why Is This Problem Non-Trivial?

1. **Margin maximization**: Why is maximum margin good?
2. **Non-separable data**: What if no perfect separation exists?
3. **Nonlinear boundaries**: Kernels enable complex patterns
4. **Computational cost**: Quadratic programming required
5. **Many parameters**: Kernel choice, C, gamma

---

## 2. Mathematical Formulation

### Linear SVM: Separable Case

For $(x_i, y_i)$ with $y_i \in \{-1, +1\}$:

$$\min_{w, b} \frac{1}{2}\|w\|^2$$

Subject to: $y_i(w^T x_i + b) \geq 1, \quad \forall i$

**Geometric interpretation**: 
- Hyperplane: $w^T x + b = 0$
- Margin: $2/\|w\|$
- Maximizing margin = minimizing $\|w\|$

### Soft Margin SVM

For non-separable data, allow violations:

$$\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \xi_i$$

Subject to: 
- $y_i(w^T x_i + b) \geq 1 - \xi_i$
- $\xi_i \geq 0$

Where $\xi_i$ = slack variable (margin violation), $C$ = tradeoff parameter.

### Dual Formulation

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j$$

Subject to: $0 \leq \alpha_i \leq C$, $\sum_i \alpha_i y_i = 0$

**Key insight**: Data appears only as inner products $x_i^T x_j$!

---

## 3. Why This Formulation?

### Why Maximum Margin?

**Statistical learning theory**: Margin relates to generalization.

VC dimension bound shows:
$$R(\text{classifier}) \leq R_{emp} + O\left(\sqrt{\frac{d/\gamma^2}{n}}\right)$$

Larger margin $\gamma$ → lower complexity → better generalization.

### The Kernel Trick

Replace $x_i^T x_j$ with kernel $K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$

**Mapping to high-dimensional space** without computing $\phi$ explicitly!

Common kernels:
- **Linear**: $K(x, x') = x^T x'$
- **Polynomial**: $K(x, x') = (x^T x' + c)^d$
- **RBF**: $K(x, x') = \exp(-\gamma\|x - x'\|^2)$

### Support Vectors

At optimum, $\alpha_i > 0$ only for points on or inside margin.

These are **support vectors** — the only points that matter for prediction!

Prediction: $f(x) = \sum_{i: \alpha_i > 0} \alpha_i y_i K(x_i, x) + b$

---

## 4. Derivation and Optimization

### Lagrangian and KKT Conditions

Primal Lagrangian:
$$L = \frac{1}{2}\|w\|^2 + C\sum_i \xi_i - \sum_i \alpha_i[y_i(w^Tx_i + b) - 1 + \xi_i] - \sum_i \mu_i \xi_i$$

KKT conditions:
$$\nabla_w L = 0 \implies w = \sum_i \alpha_i y_i x_i$$
$$\nabla_b L = 0 \implies \sum_i \alpha_i y_i = 0$$
$$\alpha_i(y_i(w^Tx_i + b) - 1 + \xi_i) = 0$$

### Quadratic Programming

Dual is a convex QP:
$$\max_\alpha \alpha^T \mathbf{1} - \frac{1}{2}\alpha^T Q \alpha$$

Where $Q_{ij} = y_i y_j K(x_i, x_j)$.

**Algorithms**:
- SMO (Sequential Minimal Optimization)
- Interior point methods
- Coordinate descent

### Computing $b$

From KKT: For any free support vector ($0 < \alpha_i < C$):
$$b = y_i - w^T x_i = y_i - \sum_j \alpha_j y_j K(x_j, x_i)$$

Average over all free SVs for stability.

---

## 5. Geometric Interpretation

### The Maximum Margin Hyperplane

SVM finds the "fattest" separator:
- Distance to nearest points is maximized
- Robust to perturbations
- Unique solution (in primal)

### Support Vectors as Critical Points

Only points on the margin boundary matter:
- Remove non-SV → same solution
- Move SV → solution changes

Typically 10-50% of data are SVs.

### Kernel-Induced Feature Space

RBF kernel: Infinite-dimensional feature space!

Points separable in high dimensions even if not in original space.

**Intuition**: Project data to space where linear separation works.

---

## 6. Probabilistic Interpretation

### SVM as Regularized Loss

SVM minimizes:
$$\frac{1}{n}\sum_i \max(0, 1 - y_i f(x_i)) + \frac{\lambda}{2}\|w\|^2$$

This is **hinge loss** + L2 regularization.

### Probability Estimates (Platt Scaling)

SVM outputs are not probabilities. Fit sigmoid:
$$P(y=1|x) = \frac{1}{1 + \exp(Af(x) + B)}$$

Fit $A, B$ on validation data.

### Bayesian SVM

Prior on weights:
- Gaussian prior → L2 regularization
- Can quantify uncertainty

But tractable inference is difficult.

---

## 7. Failure Modes and Limitations

### Scaling

SVM training: $O(n^2)$ to $O(n^3)$ complexity.

For $n > 100,000$, becomes impractical.

**Solutions**: Stochastic approximations, subset selection.

### Kernel Selection

Wrong kernel → poor performance:
- RBF too narrow → overfits
- RBF too wide → underfits
- Linear may miss nonlinearity

**Must tune kernel parameters** via cross-validation.

### Class Imbalance

SVM tries to maximize margin for all classes:
- Minority class may be ignored
- Need class weights or resampling

### Multi-class

SVM is inherently binary. Multi-class extensions:
- **One-vs-All**: $K$ binary SVMs
- **One-vs-One**: $K(K-1)/2$ binary SVMs
- Combine predictions by voting

---

## 8. Scaling and Computational Reality

### Complexity

| Operation | Time |
|-----------|------|
| Training | $O(n^2 d)$ to $O(n^3)$ |
| Prediction | $O(n_{SV} \cdot d)$ per point |
| Kernel matrix | $O(n^2 d)$ to compute |

### Memory

Must store kernel matrix: $O(n^2)$ — prohibitive for large $n$.

### Approximations

- **Nyström approximation**: Low-rank kernel approximation
- **Random Fourier features**: Approximate RBF kernel
- **SGD on hinge loss**: Linear SVM at scale (LIBLINEAR)

---

## 9. Real-World Deployment Considerations

### Preprocessing

**Essential**:
1. Normalize/standardize features (RBF is distance-based)
2. Handle missing values
3. Encode categoricals

### Hyperparameter Tuning

Grid search over:
- $C$: Typically $10^{-3}$ to $10^3$
- $\gamma$ (RBF): Typically $10^{-4}$ to $10^1$
- Kernel type

**Tip**: Start with RBF, try linear if fast results needed.

### When to Use SVM

**Good for**:
- Small to medium datasets ($n < 10^5$)
- High-dimensional data (text, genomics)
- When maximum margin is meaningful

**Avoid for**:
- Very large datasets
- Need probability outputs
- Simple linear patterns (logistic regression simpler)

---

## 10. Comparison With Alternatives

### SVM vs Logistic Regression

| Aspect | SVM | Logistic Regression |
|--------|-----|---------------------|
| Loss | Hinge | Log-loss |
| Output | Decision | Probability |
| Focus | Margin maximization | Likelihood |
| Sparse | In dual (SVs) | No |

### SVM vs Neural Networks

| Aspect | SVM | Neural Networks |
|--------|-----|-----------------|
| Data efficiency | Better on small data | Need more data |
| Kernel choice | Manual | Learned |
| Optimization | Convex (global optimum) | Non-convex |
| Scaling | Poor | Good with GPUs |

### When Alternatives Win

- **Neural networks**: Large data, complex patterns
- **Random forests**: Interpretability, mixed features  
- **Logistic regression**: Probabilities needed, simple relationships
- **Gradient boosting**: Tabular data, competitions

---

## 11. Mental Model Checkpoint

### Without Equations

SVM draws the line (or curve) that:
1. Separates classes correctly
2. Stays as far as possible from all training points

The "support vectors" are the closest points — they define the boundary. Moving other points doesn't change the line.

**Analogy**: Drawing a street (margin) between two neighborhoods — make it as wide as possible while respecting all buildings (support vectors are buildings at the edge).

### With Equations

$$\min \frac{1}{2}\|w\|^2 \text{ s.t. } y_i(w^Tx_i + b) \geq 1$$

Kernel: $f(x) = \sum_i \alpha_i y_i K(x_i, x) + b$

### Predict Behavior

1. **Increase C**: Tighter fit to data, less margin violation, may overfit
2. **Decrease C**: Allow more violations, larger margin, may underfit
3. **RBF γ too large**: Very local, overfits
4. **RBF γ too small**: Too smooth, underfits
5. **Remove non-SV**: No change to model

---

## References

### Foundational
- Vapnik (1995) - *The Nature of Statistical Learning Theory*
- Cortes & Vapnik (1995) - Support-vector networks

### Practical
- Hastie, Tibshirani, Friedman - *ESL* Chapter 12
- LIBSVM, LIBLINEAR documentation

### Extensions
- Platt (1999) - Probabilistic outputs for SVMs
- Schölkopf & Smola (2002) - *Learning with Kernels*
