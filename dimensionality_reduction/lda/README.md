# Linear Discriminant Analysis (LDA)

> **Status:** Stub - Implementation coming soon

## Overview

LDA finds projections that maximize class separability.

## Key Concepts

### Objective
$$\max_{\mathbf{w}} \frac{\mathbf{w}^T\mathbf{S}_B\mathbf{w}}{\mathbf{w}^T\mathbf{S}_W\mathbf{w}}$$

- $\mathbf{S}_B$: Between-class scatter
- $\mathbf{S}_W$: Within-class scatter

### vs PCA
- PCA: Unsupervised, max variance
- LDA: Supervised, max class separation

## Contributing

See [CONTRIBUTING.md](../../../CONTRIBUTING.md).
