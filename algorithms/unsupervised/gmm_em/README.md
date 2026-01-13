# Gaussian Mixture Models & EM Algorithm

> **Status:** Stub - Implementation coming soon

## Overview

GMMs model data as a mixture of Gaussians, fitted using Expectation-Maximization.

## Structure (To Be Implemented)

```
gmm_em/
├── README.md           ← You are here
├── theory.md           # MLE, EM algorithm derivation
├── scratch.py          # NumPy implementation
├── sklearn_impl.py     # scikit-learn implementation
└── experiments.ipynb   # Visualizations
```

## Key Concepts

### Model
$$p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

### EM Algorithm

**E-Step:** Compute responsibilities
$$\gamma_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}_i|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j \pi_j \mathcal{N}(\mathbf{x}_i|\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

**M-Step:** Update parameters
$$\boldsymbol{\mu}_k = \frac{\sum_i \gamma_{ik}\mathbf{x}_i}{\sum_i \gamma_{ik}}$$

### GMM vs K-Means
- K-Means: Hard assignment
- GMM: Soft (probabilistic) assignment

## Contributing

Want to implement this module? See [CONTRIBUTING.md](../../../../CONTRIBUTING.md).
