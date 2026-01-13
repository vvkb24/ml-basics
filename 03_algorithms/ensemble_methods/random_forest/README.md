# Random Forest

> **Status:** Stub - Implementation coming soon

## Overview

Random Forest is an ensemble of decision trees using bagging and random feature selection.

## Structure (To Be Implemented)

```
random_forest/
├── README.md           ← You are here
├── theory.md           # Bagging, OOB error, feature importance
├── scratch.py          # NumPy implementation
├── sklearn_impl.py     # scikit-learn implementation
└── experiments.ipynb   # Visualizations
```

## Key Concepts

### Algorithm
1. Bootstrap samples from training data
2. Train decision tree on each sample
3. At each split, consider random subset of features
4. Aggregate predictions (vote/average)

### Key Ideas
- **Bagging:** Reduces variance via averaging
- **Feature randomness:** Decorrelates trees
- **OOB Error:** Free validation using out-of-bag samples

### Feature Importance
- Mean decrease in Gini impurity
- Permutation importance

## Contributing

Want to implement this module? See [CONTRIBUTING.md](../../../../CONTRIBUTING.md).
