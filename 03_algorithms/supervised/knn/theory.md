# K-Nearest Neighbors: Complete Mathematical Theory

A rigorous treatment of KNN covering non-parametric classification and regression.

---

## 1. Problem Definition

### What Problem Is Being Solved?

Given labeled examples $(x_i, y_i)$, predict $y$ for a new point $x$ by finding the "most similar" training examples.

**Key idea**: Points close in feature space have similar labels.

### Why Is This Problem Non-Trivial?

1. **Distance metric**: What does "similar" mean?
2. **Choosing $k$**: How many neighbors to use?
3. **Curse of dimensionality**: Distance concentrates in high dimensions
4. **Computational cost**: Must compare to all training points
5. **Feature scaling**: Features on different scales dominate

---

## 2. Mathematical Formulation

### Distance Function

Most common: Euclidean distance
$$d(x, x') = \|x - x'\|_2 = \sqrt{\sum_{j=1}^d (x_j - x'_j)^2}$$

**Alternatives**:
- Manhattan: $\|x - x'\|_1 = \sum_j |x_j - x'_j|$
- Minkowski: $\|x - x'\|_p = (\sum_j |x_j - x'_j|^p)^{1/p}$
- Mahalanobis: $(x-x')^T \Sigma^{-1} (x-x')$ (accounts for correlation)

### KNN Classification

For query point $x$, find $k$ nearest neighbors $\mathcal{N}_k(x)$:
$$\hat{y} = \arg\max_c \sum_{i \in \mathcal{N}_k(x)} \mathbb{1}(y_i = c)$$

(Majority vote among neighbors)

**Weighted version**:
$$\hat{y} = \arg\max_c \sum_{i \in \mathcal{N}_k(x)} w_i \cdot \mathbb{1}(y_i = c)$$

Common weights: $w_i = 1/d(x, x_i)$ or $w_i = 1/d(x, x_i)^2$

### KNN Regression

$$\hat{y} = \frac{1}{k}\sum_{i \in \mathcal{N}_k(x)} y_i$$

Weighted version:
$$\hat{y} = \frac{\sum_{i \in \mathcal{N}_k(x)} w_i \cdot y_i}{\sum_{i \in \mathcal{N}_k(x)} w_i}$$

---

## 3. Why This Formulation?

### Theoretical Foundation: Cover & Hart (1967)

**Theorem** (1-NN Error Bound):

As $n \to \infty$, the error rate $R$ of 1-NN satisfies:
$$R^* \leq R_{1NN} \leq 2R^*(1 - R^*)$$

Where $R^*$ is the Bayes optimal error rate.

**Implication**: 1-NN error is at most twice optimal! Remarkable for such a simple method.

### What Assumptions Are Required?

1. **Smoothness**: Similar points have similar labels
2. **Meaningful distance**: Euclidean distance captures similarity
3. **Sufficient coverage**: Training data covers the space
4. **Comparable scales**: Features contribute appropriately

### What Breaks If Assumptions Fail?

| Violation | Consequence |
|-----------|-------------|
| Non-smooth decision boundary | High error near boundary |
| Irrelevant features | Distance dominated by noise |
| Sparse data | No nearby neighbors |
| Skewed scales | Large-scale features dominate |

---

## 4. Derivation and Optimization

### Connection to Kernel Density Estimation

KNN regression is a **local averaging** estimator:
$$\hat{f}(x) = \frac{1}{k}\sum_{i \in \mathcal{N}_k(x)} y_i$$

This is a special case of kernel regression with a **box kernel** that adapts spatially.

### Optimal $k$ Selection

**Bias-Variance Tradeoff**:
- Small $k$: Low bias, high variance (sensitive to noise)
- Large $k$: High bias, low variance (over-smooth)

**Asymptotic optimality**: For $n \to \infty$, $k \to \infty$, $k/n \to 0$, KNN is consistent.

Typical choice: $k = O(\sqrt{n})$

**Cross-validation**: Best practical approach

### Efficient Nearest Neighbor Search

**Brute force**: $O(nd)$ per query

**Data structures**:
- KD-tree: $O(d \log n)$ average (but $O(dn)$ in high dimensions)
- Ball tree: Better in higher dimensions
- LSH: Approximate nearest neighbors, sublinear

---

## 5. Geometric Interpretation

### Voronoi Diagram (1-NN)

For 1-NN, space is partitioned into Voronoi cells:
- Each training point has a region
- All points in region are closest to that training point
- Decision boundary: perpendicular bisectors

### Decision Boundary

The decision boundary is:
- Piecewise linear (Euclidean distance)
- Potentially complex
- Adapts to local data density

**Comparison**:
- Linear classifier: Straight line
- KNN: Arbitrarily complex local boundaries

### High-Dimensional Geometry

**Curse of dimensionality**:
- In high dimensions, points are nearly equidistant
- Nearest neighbor may not be meaningful closer than others

**Theorem** (Beyer et al., 1999):
$$\lim_{d \to \infty} \frac{\text{dist}_{max} - \text{dist}_{min}}{\text{dist}_{min}} = 0$$

Distances concentrate → nearest neighbor becomes arbitrary.

---

## 6. Probabilistic Interpretation

### Density Estimation View

KNN estimates local density:
$$\hat{p}(x) = \frac{k}{n \cdot V_k(x)}$$

Where $V_k(x)$ is the volume of the ball containing $k$ neighbors.

### Posterior Probability Estimation

For classification, estimate:
$$\hat{p}(y=c|x) = \frac{1}{k}\sum_{i \in \mathcal{N}_k(x)} \mathbb{1}(y_i = c)$$

This is a consistent estimator of the class posterior.

### Bayesian KNN

Place prior on $k$, distance metric, or neighbors:
- Uncertainty in predictions
- Automatic complexity control
- But computationally expensive

---

## 7. Failure Modes and Limitations

### Curse of Dimensionality

In high dimensions:
- Data is sparse
- All points are far from each other
- "Nearest" neighbors aren't nearby

**Mitigation**:
- Dimensionality reduction (PCA) first
- Feature selection
- Manifold learning

### Imbalanced Classes

If class A has 99% of samples:
- Most neighbors are class A
- Class B rarely predicted

**Solutions**:
- Weighted voting
- Resampling
- Adjust distance metric

### Noisy Features

Irrelevant features:
- Add noise to distances
- Drown out meaningful features

**Solutions**:
- Feature selection
- Feature weighting (learned metric)
- Domain knowledge

### Expensive Prediction

Unlike trained models, KNN stores all training data:
- Memory: $O(nd)$
- Query time: $O(nd)$ brute force
- Not suitable for low-latency applications

---

## 8. Scaling and Computational Reality

### Computational Complexity

| Operation | Brute Force | KD-tree | Approximate |
|-----------|-------------|---------|-------------|
| Build | $O(1)$ | $O(n \log n)$ | $O(n)$ |
| Query | $O(nd)$ | $O(d \log n)$ average | $O(d)$ |
| Memory | $O(nd)$ | $O(nd)$ | $O(nd)$ |

### Approximate Nearest Neighbors (ANN)

For large-scale KNN:
- **LSH**: Hash similar points to same bucket
- **HNSW**: Hierarchical navigable small world graphs
- **FAISS**: Facebook's library for billion-scale search

Trade exact for approximate → orders of magnitude faster.

### Online/Streaming KNN

When data changes:
- Naive: Rebuild data structure
- Better: Delete + insert operations
- Cover trees support efficient updates

---

## 9. Real-World Deployment Considerations

### Feature Preprocessing

**Critical for KNN**:
1. Scale features (standardization or normalization)
2. Remove or downweight irrelevant features
3. Consider learned distance metrics

### Memory Constraints

Storing all training data is expensive:
- Data reduction: Keep only prototypes
- Compression: Quantize features
- Approximate methods: Trade accuracy for space

### Latency vs Accuracy

For real-time predictions:
- Use approximate NN
- Reduce dimensions
- Use smaller $k$

### When to Use KNN

**Good for**:
- Quick baselines
- Small to medium datasets
- When interpretability matters (can show neighbors)
- Recommendation systems (item similarity)

**Not good for**:
- Very high dimensions (without reduction)
- Large training sets
- Low-latency requirements
- When model updates are frequent

---

## 10. Comparison With Alternatives

### KNN vs Parametric Models

| Aspect | KNN | Parametric (Logistic, SVM) |
|--------|-----|---------------------------|
| Training | None (store data) | Fit parameters |
| Prediction | Expensive | Cheap |
| Decision boundary | Flexible | Fixed form |
| Sample efficiency | Poor | Better |

### KNN vs Decision Trees

| Aspect | KNN | Decision Trees |
|--------|-----|----------------|
| Interpretability | Show neighbors | Show rules |
| Speed | Slow query | Fast query |
| Feature importance | All features used | Natural selection |
| Handles categorical | Needs encoding | Natural |

### When Alternatives Win

- **Neural networks**: Large data, complex patterns
- **Random Forest**: Better accuracy, feature importance
- **SVM**: High dimensions with kernels
- **Linear models**: Simple relationships, interpretability

---

## 11. Mental Model Checkpoint

### Without Equations

KNN is like asking your neighbors for advice:
- To make a decision, look at the $k$ most similar training examples
- Go with the majority (classification) or average (regression)
- The more neighbors you ask, the more averaged your answer

**Analogy**: House pricing — look at similar houses in the neighborhood.

### With Equations

$$\hat{y} = \frac{1}{k}\sum_{i \in \mathcal{N}_k(x)} y_i$$

Where $\mathcal{N}_k(x)$ = indices of $k$ closest points to $x$.

### Predict Behavior

1. **$k=1$**: Every training point gets its own region, may overfit
2. **$k=n$**: Predict mean everywhere, ignores input
3. **Scaling features**: Essential, otherwise large features dominate
4. **Adding noise features**: Performance degrades
5. **More data**: Slower queries but better predictions

---

## References

### Foundational
- Cover & Hart (1967) - "Nearest neighbor pattern classification"
- Fix & Hodges (1951) - Original formulation

### Algorithms
- Friedman et al. (1977) - KD-trees
- Indyk & Motwani (1998) - Locality-sensitive hashing

### Analysis
- Beyer et al. (1999) - Curse of dimensionality
- Hastie, Tibshirani, Friedman - *ESL* Chapter 13
