# Principal Component Analysis: Complete Mathematical Theory

A rigorous treatment of PCA covering linear dimensionality reduction from all theoretical perspectives.

---

## 1. Problem Definition

### What Problem Is Being Solved?

Given data $X \in \mathbb{R}^{n \times d}$ with $n$ samples in $d$ dimensions, find a linear mapping to $k < d$ dimensions that **preserves maximum variance**.

### Why Is This Problem Non-Trivial?

1. **Information loss**: Any projection loses information
2. **Which directions?**: Infinitely many $k$-dimensional subspaces exist
3. **Optimality criterion**: What makes one projection "better"?
4. **Computational cost**: Finding optimal subspace efficiently
5. **Interpretation**: What do the new dimensions mean?

---

## 2. Mathematical Formulation

### Notation

| Symbol | Meaning |
|--------|---------|
| $X \in \mathbb{R}^{n \times d}$ | Data matrix (centered) |
| $C = \frac{1}{n}X^TX$ | Covariance matrix |
| $v_1, ..., v_k$ | Principal components |
| $\lambda_1 \geq ... \geq \lambda_d$ | Eigenvalues |
| $Z = XV$ | Projected data |

### Objective: Maximum Variance

Find unit vector $v$ that maximizes variance of projected data:

$$\max_{\|v\|=1} \text{Var}(Xv) = \max_{\|v\|=1} v^T C v$$

**Solution**: First eigenvector $v_1$ of $C$ with eigenvalue $\lambda_1$.

### Multiple Components

For $k$ components, find orthonormal $V = [v_1, ..., v_k]$ maximizing:

$$\max_{V^TV = I} \text{tr}(V^T C V) = \sum_{i=1}^k \lambda_i$$

**Solution**: Top $k$ eigenvectors of $C$.

---

## 3. Why This Formulation?

### Equivalent Perspectives

**Maximum Variance**: The directions that capture most data spread.

**Minimum Reconstruction Error**: The projection that allows best reconstruction:
$$\min_V \sum_{i=1}^n \|x_i - VV^T x_i\|^2$$

This is **exactly equivalent** to maximum variance!

**Proof**: Reconstruction error = Total variance - Captured variance
$$\sum_i \|x_i\|^2 - \sum_{j=1}^k \lambda_j$$

### What Assumptions Are Required?

1. **Linear relationships**: Only linear patterns captured
2. **Variance = importance**: High variance directions are informative
3. **Mean-centered data**: PCA is about variance around mean
4. **Scale matters**: Features must be on comparable scales

### What Breaks If Assumptions Fail?

| Violation | Consequence |
|-----------|-------------|
| Nonlinear structure | Manifold not captured (use kernel PCA) |
| Variance ≠ importance | May keep noise, discard signal |
| Different scales | Large-scale features dominate |
| Outliers | Directions fit outliers |

---

## 4. Derivation and Optimization

### Lagrangian Derivation

Maximize $v^TCv$ subject to $v^Tv = 1$:

$$\mathcal{L} = v^TCv - \lambda(v^Tv - 1)$$

Setting $\nabla_v \mathcal{L} = 0$:
$$2Cv - 2\lambda v = 0$$
$$Cv = \lambda v$$

**$v$ must be an eigenvector of $C$!**

Substituting back:
$$v^TCv = v^T\lambda v = \lambda$$

To maximize, choose **largest eigenvalue**.

### Algorithms

**Eigendecomposition**:
$$C = V\Lambda V^T$$
Time: $O(d^3)$, needs full $C$

**SVD of Data Matrix**:
$$X = U\Sigma V^T$$
Then $C = \frac{1}{n}V\Sigma^2V^T$, so eigenvectors of $C$ = right singular vectors of $X$.
Time: $O(\min(nd^2, n^2d))$

**Power Iteration** (for top-$k$):
$$v^{(t+1)} = \frac{Cv^{(t)}}{\|Cv^{(t)}\|}$$
Converges to largest eigenvector. Good for very large $d$.

**Randomized SVD**:
For very large data, use random projections + power iterations.
Time: $O(ndk)$ for $k$ components.

### Numerical Stability

- Use SVD rather than eigendecomposition of $C$
- Avoids forming $X^TX$ explicitly (condition number squared)
- Modern implementations: `np.linalg.svd(X, full_matrices=False)`

---

## 5. Geometric Interpretation

### Variance Ellipsoid

For Gaussian data, PCA finds the **axes of the covariance ellipsoid**:
- Major axis = first PC (direction of maximum spread)
- Minor axis = last PC (direction of minimum spread)

### Orthogonal Projection

PCA projects onto the $k$-dimensional subspace spanned by $v_1, ..., v_k$:

$$z_i = V^T x_i$$

This is an **orthogonal projection** — each point moves perpendicular to the subspace.

### Information Loss

Discarding dimension $j$ loses variance $\lambda_j$.

**Explained variance ratio**:
$$\text{EVR}_k = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i}$$

Rule of thumb: Keep enough components for EVR > 0.95.

### Dimensionality as Intrinsic Property

If data lies on a $k$-dimensional linear subspace:
- $\lambda_1, ..., \lambda_k > 0$
- $\lambda_{k+1}, ..., \lambda_d \approx 0$

PCA discovers the **intrinsic dimensionality**.

---

## 6. Probabilistic Interpretation

### Probabilistic PCA (PPCA)

Assume generative model:
$$z \sim \mathcal{N}(0, I_k)$$
$$x = Wz + \mu + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$

**MLE solution**: $W$ spans same subspace as top-$k$ PCs!

### Factor Analysis Connection

If noise covariance is diagonal but not spherical:
$$\epsilon \sim \mathcal{N}(0, \Psi)$$

This is Factor Analysis — different from PCA but related.

### Latent Variable Interpretation

Principal components $z = V^Tx$ are **latent variables**:
- Uncorrelated (orthogonal directions)
- Capture decreasing amounts of variance
- Can be interpreted as hidden factors

### Bayesian PCA

Place prior on $W$:
$$p(W) = \prod_j \mathcal{N}(w_j | 0, \alpha_j^{-1}I)$$

With automatic relevance determination (ARD):
- Learns number of components automatically
- Prunes unnecessary dimensions

---

## 7. Failure Modes and Limitations

### Nonlinear Structure

If data lies on a **curved manifold**:
- PCA captures tangent space at mean
- Misses curvature
- Requires nonlinear methods (kernel PCA, autoencoders)

**Example**: Swiss roll — PCA projects to 2D but doesn't "unroll"

### Scale Dependence

If features have different scales:
- Large-scale features dominate PCs
- Must standardize (divide by std dev) or not depending on problem

**When to standardize**: 
- Different units (meters vs dollars)
- Want each feature equally important

**When not to standardize**:
- Same units, variance is meaningful
- Preserving original geometry matters

### Outliers

Single outlier can dramatically shift principal components.

**Solutions**:
- Robust PCA (uses robust covariance estimator)
- Remove outliers first
- Use median-based methods

### High Dimensions, Few Samples

When $d >> n$:
- Covariance estimate is rank-deficient
- At most $n-1$ non-zero eigenvalues
- Need regularization

---

## 8. Scaling and Computational Reality

### Time Complexity

| Method | Time | When to Use |
|--------|------|-------------|
| Full SVD | $O(\min(nd^2, n^2d))$ | $n, d < 10^4$ |
| Truncated SVD | $O(ndk)$ | Large $n$ or $d$ |
| Randomized SVD | $O(ndk)$ | Very large, $k$ small |
| Power iteration | $O(ndt)$ | Streaming data |

### Memory Complexity

- Store $X$: $O(nd)$
- Store $C$: $O(d^2)$ — avoid if $d$ large
- SVD approach: $O(nk)$ for $k$ components

### Incremental PCA

For data that doesn't fit in memory:
- Process in mini-batches
- Update running estimate of covariance
- `sklearn.decomposition.IncrementalPCA`

---

## 9. Real-World Deployment Considerations

### Preprocessing Pipeline

```
1. Handle missing values
2. Remove outliers (or use robust PCA)
3. Decide: standardize or not?
4. Fit PCA on training data
5. Apply same transformation to test data
```

**Critical**: Never fit PCA on test data — leaks information!

### Choosing Number of Components

Methods:
1. **Explained variance threshold**: Keep 95% variance
2. **Elbow method**: Plot eigenvalues, find "elbow"
3. **Cross-validation**: Choose $k$ that minimizes downstream error
4. **Parallel analysis**: Compare to random data eigenvalues

### Interpretation Challenges

Principal components are linear combinations:
$$PC_1 = 0.3 \cdot \text{height} + 0.5 \cdot \text{weight} + ...$$

Often hard to interpret — not a named concept.

### Latency Considerations

Inference (projection) is just matrix multiply:
$$z = V^T(x - \mu)$$

Very fast — $O(dk)$ per sample.

---

## 10. Comparison With Alternatives

### PCA vs Other Linear Methods

| Method | Objective | Supervised? |
|--------|-----------|-------------|
| PCA | Max variance | No |
| LDA | Max class separation | Yes |
| ICA | Max independence | No |
| NMF | Non-negative factors | No |

### PCA vs Nonlinear Methods

| Method | Handles Nonlinearity | Preserves |
|--------|---------------------|-----------|
| PCA | No | Global variance |
| Kernel PCA | Yes | Kernel distances |
| t-SNE | Yes | Local structure |
| UMAP | Yes | Local + some global |
| Autoencoders | Yes | Learned objective |

### When PCA Wins

1. Linear relationships in data
2. Need interpretable, reversible transform
3. Fast inference required
4. Preprocessing for downstream models

### When Alternatives Win

- **LDA**: Classification task, want class separation
- **t-SNE/UMAP**: Visualization, nonlinear structure
- **Autoencoders**: Complex nonlinear patterns

---

## 11. Mental Model Checkpoint

### Without Equations

PCA finds the directions in which your data varies most. Imagine a cloud of points — PCA finds the longest axis of this cloud, then the second longest perpendicular to the first, and so on. Projecting onto these axes gives you the most informative low-dimensional view.

**Analogy**: Finding the best angle to photograph a 3D sculpture — the angle that shows the most shape variation.

### With Equations

$$\max_{\|v\|=1} v^T C v = \lambda_1$$

Principal components are eigenvectors of covariance matrix: $Cv_i = \lambda_i v_i$

### Predict Behavior

1. **All features perfectly correlated**: One PC captures 100% variance
2. **All features independent, same variance**: All PCs have equal eigenvalue
3. **Adding noise feature**: One more small eigenvalue
4. **Doubling one feature's scale**: That feature dominates PC1
5. **10x more samples**: Same PCs, more stable estimates

---

## References

### Foundational
- Pearson (1901) - "On lines and planes of closest fit"
- Hotelling (1933) - "Analysis of complex statistical variables"

### Modern
- Jolliffe (2002) - *Principal Component Analysis* (textbook)
- Tipping & Bishop (1999) - Probabilistic PCA

### Applications
- Turk & Pentland (1991) - Eigenfaces
- Wall et al. (2003) - SVD in genomics
