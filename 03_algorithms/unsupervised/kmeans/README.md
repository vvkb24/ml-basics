# K-Means Clustering

> **Status:** Stub - Implementation coming soon

## Overview

K-Means partitions data into k clusters by minimizing within-cluster variance.

## Structure (To Be Implemented)

```
kmeans/
├── README.md           ← You are here
├── theory.md           # Lloyd's algorithm, convergence, initialization
├── scratch.py          # NumPy implementation
├── sklearn_impl.py     # scikit-learn implementation
└── experiments.ipynb   # Visualizations
```

## Key Concepts

### Objective
$$\min_{\mu, C} \sum_{k=1}^{K} \sum_{i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

### Algorithm (Lloyd's)
1. Initialize centroids
2. Assign points to nearest centroid
3. Update centroids as cluster means
4. Repeat until convergence

### Initialization Methods
- Random
- K-Means++ (improved)
- Multiple restarts

### Choosing K
- Elbow method
- Silhouette score
- Gap statistic

## Contributing

Want to implement this module? See [CONTRIBUTING.md](../../../../CONTRIBUTING.md).
