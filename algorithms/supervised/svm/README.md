# Support Vector Machines (SVM)

> **Status:** Stub - Implementation coming soon

## Overview

SVMs find the optimal hyperplane that maximizes margin between classes.

## Structure (To Be Implemented)

```
svm/
├── README.md           ← You are here
├── theory.md           # Margin, dual formulation, kernels
├── scratch.py          # NumPy implementation
├── sklearn_impl.py     # scikit-learn implementation
└── experiments.ipynb   # Visualizations
```

## Key Concepts

### Primal Problem
$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2$$
$$\text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$$

### Soft Margin
$$\min_{\mathbf{w}, b, \xi} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum\xi_i$$

### Kernel Trick
$$K(\mathbf{x}, \mathbf{y}) = \phi(\mathbf{x})^T\phi(\mathbf{y})$$

Common kernels: Linear, RBF, Polynomial

## Contributing

Want to implement this module? See [CONTRIBUTING.md](../../../../CONTRIBUTING.md).
