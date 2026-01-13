# K-Nearest Neighbors (KNN)

> **Status:** Stub - Implementation coming soon

## Overview

KNN is a non-parametric algorithm that classifies points based on their k nearest neighbors.

## Structure (To Be Implemented)

```
knn/
├── README.md           ← You are here
├── theory.md           # Distance metrics, curse of dimensionality
├── scratch.py          # NumPy implementation
├── sklearn_impl.py     # scikit-learn implementation
└── experiments.ipynb   # Visualizations
```

## Key Concepts

### Algorithm
1. Compute distances to all training points
2. Find k nearest neighbors
3. Classification: Majority vote
4. Regression: Average of neighbors

### Distance Metrics
- **Euclidean:** $d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum(x_i - y_i)^2}$
- **Manhattan:** $d(\mathbf{x}, \mathbf{y}) = \sum|x_i - y_i|$
- **Minkowski:** $d(\mathbf{x}, \mathbf{y}) = \left(\sum|x_i - y_i|^p\right)^{1/p}$

### Computational Complexity
- Training: O(1)
- Prediction: O(nd) per sample

## Contributing

Want to implement this module? See [CONTRIBUTING.md](../../../../CONTRIBUTING.md).
