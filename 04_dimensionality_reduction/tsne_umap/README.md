# t-SNE and UMAP

> **Status:** Stub - Implementation coming soon

## Overview

Non-linear dimensionality reduction for visualization.

## t-SNE

### Algorithm
1. Compute pairwise similarities in high-D (Gaussian)
2. Compute similarities in low-D (t-distribution)
3. Minimize KL divergence

### Perplexity
Balance between local and global structure.

## UMAP

### Key Differences from t-SNE
- Faster computation
- Better preservation of global structure
- Theoretically grounded (topology)

## Contributing

See [CONTRIBUTING.md](../../../CONTRIBUTING.md).
