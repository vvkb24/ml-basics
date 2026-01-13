# Decision Trees

> **Status:** Stub - Implementation coming soon

## Overview

Decision trees learn hierarchical if-then rules for classification and regression.

## Structure (To Be Implemented)

```
decision_trees/
├── README.md           ← You are here
├── theory.md           # Entropy, information gain, Gini impurity
├── scratch.py          # NumPy implementation
├── sklearn_impl.py     # scikit-learn implementation
└── experiments.ipynb   # Visualizations
```

## Key Concepts

### Entropy
$$H(S) = -\sum_{c} p_c \log_2(p_c)$$

### Information Gain
$$IG(S, A) = H(S) - \sum_{v} \frac{|S_v|}{|S|} H(S_v)$$

### Gini Impurity
$$G(S) = 1 - \sum_{c} p_c^2$$

### Splitting Criteria
- ID3: Information Gain
- C4.5: Gain Ratio
- CART: Gini Impurity (classification), MSE (regression)

## Contributing

Want to implement this module? See [CONTRIBUTING.md](../../../../CONTRIBUTING.md).
