# Hierarchical Clustering

> **Status:** Stub - Implementation coming soon

## Overview

Hierarchical clustering creates a tree of nested clusters without specifying k.

## Structure (To Be Implemented)

```
hierarchical_clustering/
├── README.md           ← You are here
├── theory.md           # Linkage methods, dendrograms
├── scratch.py          # NumPy implementation
├── sklearn_impl.py     # scikit-learn implementation
└── experiments.ipynb   # Visualizations
```

## Key Concepts

### Agglomerative (Bottom-Up)
1. Each point is its own cluster
2. Merge closest clusters
3. Repeat until one cluster remains

### Linkage Methods
- **Single:** $d(A, B) = \min_{a \in A, b \in B} d(a, b)$
- **Complete:** $d(A, B) = \max_{a \in A, b \in B} d(a, b)$
- **Average:** $d(A, B) = \frac{1}{|A||B|}\sum_{a,b} d(a, b)$
- **Ward:** Minimize variance increase

### Dendrogram
Visual representation of cluster hierarchy.

## Contributing

Want to implement this module? See [CONTRIBUTING.md](../../../../CONTRIBUTING.md).
