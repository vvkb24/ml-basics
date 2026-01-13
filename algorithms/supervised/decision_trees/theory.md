# Decision Trees: Complete Mathematical Theory

A rigorous treatment of decision trees covering recursive partitioning for classification and regression.

---

## 1. Problem Definition

### What Problem Is Being Solved?

Partition feature space into regions using axis-aligned splits, assigning each region a prediction.

**Goal**: Create interpretable, hierarchical rules that map inputs to outputs.

### Why Is This Problem Non-Trivial?

1. **Exponentially many trees**: Can't enumerate all possible trees
2. **Greedy optimization**: Local decisions may be globally suboptimal
3. **Overfitting**: Can achieve perfect training accuracy
4. **Instability**: Small data changes → completely different tree
5. **Axis alignment**: Only rectangular partitions

---

## 2. Mathematical Formulation

### Tree Structure

A decision tree is a function $T: \mathbb{R}^d \to \mathbb{R}$ (or class label):
- **Internal nodes**: Split on feature $x_j < \theta$
- **Leaf nodes**: Predict constant value $c$

$$T(x) = \sum_{m=1}^M c_m \cdot \mathbb{1}(x \in R_m)$$

Where $R_1, ..., R_M$ are the leaf regions partitioning the space.

### Splitting Criterion: Classification

**Impurity measures** for node with class proportions $p_c$:

**Gini Impurity**:
$$G = \sum_{c=1}^C p_c(1 - p_c) = 1 - \sum_{c=1}^C p_c^2$$

**Entropy**:
$$H = -\sum_{c=1}^C p_c \log_2 p_c$$

**Misclassification Error**:
$$E = 1 - \max_c p_c$$

### Information Gain

For split dividing node into left ($L$) and right ($R$):
$$\text{Gain} = I(\text{parent}) - \frac{n_L}{n}I(L) - \frac{n_R}{n}I(R)$$

Greedy: Choose split maximizing gain.

### Splitting Criterion: Regression

**Mean Squared Error**:
$$\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \bar{y})^2$$

Minimize weighted MSE of children.

---

## 3. Why This Formulation?

### Greedy Recursive Partitioning

Finding optimal tree is NP-hard. Greedy algorithm:
1. Find best split for current node
2. Partition data
3. Recurse on children
4. Stop when criterion met

### Gini vs Entropy

Both are concave functions of class proportions:
- Maximum impurity when uniform ($p_c = 1/C$)
- Minimum when pure ($p_c = 1$ for some $c$)

**In practice**: Very similar results. Gini slightly faster (no log).

### Stopping Criteria

- Maximum depth
- Minimum samples per leaf
- Minimum impurity decrease
- Minimum samples for split

**Without stopping**: Tree can memorize training data perfectly.

---

## 4. Derivation and Optimization

### Finding Optimal Split

For continuous feature $x_j$:
1. Sort samples by $x_j$: $O(n \log n)$
2. Try each threshold between consecutive values
3. Compute impurity decrease for each
4. Keep best

Total: $O(n \cdot d \cdot n \log n) = O(dn^2 \log n)$ worst case.

### Optimal Predictions in Leaves

**Classification**: Majority class
$$c_m = \arg\max_c \sum_{i: x_i \in R_m} \mathbb{1}(y_i = c)$$

**Regression**: Mean
$$c_m = \frac{1}{|R_m|}\sum_{i: x_i \in R_m} y_i$$

### Pruning

**Pre-pruning**: Stop growing early
**Post-pruning**: Grow full tree, then remove subtrees

**Cost-complexity pruning**:
$$\text{Cost}_\alpha(T) = \sum_{m=1}^{|T|} \text{Error}(R_m) + \alpha |T|$$

Increase $\alpha$ → prune more → simpler tree.

---

## 5. Geometric Interpretation

### Axis-Aligned Partitions

Each split creates a line parallel to an axis:
- Split on $x_1 < 3$: Vertical line at $x_1 = 3$
- Regions are hyperrectangles

**Limitation**: Cannot represent diagonal boundaries efficiently.

### Decision Boundary

For binary classification:
- Boundary is staircase-shaped
- Approximates any boundary with enough splits
- But may need exponentially many splits for simple diagonals

### Depth and Expressiveness

- Depth $d$ tree: At most $2^d$ leaves
- Can represent any function if deep enough
- Deeper = more complex = higher variance

---

## 6. Probabilistic Interpretation

### Class Probability Estimates

Leaf $m$ estimates:
$$\hat{p}(y=c|x \in R_m) = \frac{n_{m,c}}{n_m}$$

Often poorly calibrated (overconfident). Use calibration post-hoc.

### Bayesian Perspective

**Prior**: Penalize deep trees
**Posterior**: Integrate over all trees

Computationally intractable → use MCMC or approximations.

### Connection to Kernel Methods

Tree-induced kernel:
$$k(x, x') = \mathbb{1}(\text{same leaf})$$

Random forests create smoother kernel through averaging.

---

## 7. Failure Modes and Limitations

### Overfitting

Without regularization, decision trees perfectly fit training data:
- One sample per leaf
- Zero training error, terrible generalization

**Solutions**: Pruning, ensembles.

### Instability

Small data change → completely different tree:
- Remove one sample → different splits
- Add one sample → different structure

**High variance**! Ensembles (Random Forest, Boosting) address this.

### Axis Alignment

Cannot efficiently capture:
- Diagonal decision boundaries
- Smooth functions

**Solution**: Oblique trees (split on linear combinations), but harder to fit.

### Biased toward Many-Category Features

Feature with many categories:
- More possible splits
- Higher chance of finding good split (by chance)
- Leads to overfitting on that feature

**Solution**: Use information gain ratio (normalize by split entropy).

---

## 8. Scaling and Computational Reality

### Time Complexity

| Operation | Complexity |
|-----------|------------|
| Training (per node) | $O(d \cdot n \log n)$ |
| Training (full tree) | $O(d \cdot n^2 \log n)$ worst case |
| Training (balanced tree) | $O(d \cdot n \log^2 n)$ |
| Prediction | $O(\text{depth})$ |

### Memory

- Store tree structure: $O(\text{nodes})$
- No need to store training data after building
- Very memory-efficient for prediction

### Parallelization

Can parallelize over:
- Features (try different splits in parallel)
- Subtrees (build children in parallel)

---

## 9. Real-World Deployment Considerations

### Interpretability

Each prediction has a path:
```
IF age > 30 AND income > 50k AND education = "Graduate"
THEN approved = True
```

**Highly interpretable**: Can explain to stakeholders.

### Feature Importance

Measured by total impurity decrease:
$$\text{Importance}(j) = \sum_{\text{nodes splitting on } j} n_m \cdot \Delta I_m$$

Useful for feature selection.

### Handling Missing Data

**Surrogate splits**: Find similar splits when primary feature missing
**Separate category**: Treat missing as its own category

### Categorical Features

**Options**:
1. One-vs-rest splits
2. All possible subsets ($2^{k-1} - 1$ for $k$ categories)
3. Target encoding

---

## 10. Comparison With Alternatives

### Decision Trees vs Linear Models

| Aspect | Decision Trees | Linear Models |
|--------|----------------|---------------|
| Boundary | Rectangular | Hyperplane |
| Interpretability | Rule-based | Coefficient-based |
| Interactions | Automatic | Must specify |
| Extrapolation | Constant | Linear |

### Decision Trees vs Neural Networks

| Aspect | Trees | Neural Nets |
|--------|-------|-------------|
| Data needed | Less | More |
| Interpretability | High | Low |
| Boundary | Axis-aligned | Arbitrary |
| Training | Fast | Slow |

### Ensembles: The Solution to Tree Weaknesses

| Ensemble | How It Helps |
|----------|--------------|
| Random Forest | Reduces variance through averaging |
| Gradient Boosting | Reduces bias through sequential fitting |
| AdaBoost | Focuses on hard examples |

---

## 11. Mental Model Checkpoint

### Without Equations

A decision tree is a flowchart of yes/no questions:
- Start at the root
- Answer a question about one feature
- Go left or right based on answer
- Repeat until reaching a leaf
- The leaf gives your prediction

**Analogy**: 20 questions game — each question narrows down possibilities.

### With Equations

Tree partitions space into regions:
$$T(x) = \sum_{m=1}^M c_m \cdot \mathbb{1}(x \in R_m)$$

Splits minimize impurity (Gini or entropy).

### Predict Behavior

1. **No pruning**: Overfits, perfect training accuracy
2. **Max depth = 1**: Underfits (just one split)
3. **Removing a feature**: Tree may grow deeper to compensate
4. **Duplicating a feature**: Split might happen on either copy
5. **Scaling features**: Decision trees are scale-invariant!

---

## References

### Foundational
- Breiman et al. (1984) - CART
- Quinlan (1986) - ID3
- Quinlan (1993) - C4.5

### Ensembles
- Breiman (2001) - Random Forests
- Friedman (2001) - Gradient Boosting

### Theory
- Hastie, Tibshirani, Friedman - *ESL* Chapter 9
