# Attention Mechanisms

> **Status:** Stub - Implementation coming soon

## Overview

Attention allows models to focus on relevant parts of input.

## Key Concepts

### Bahdanau Attention (Additive)
$$\alpha_{ij} = \text{softmax}(\mathbf{v}^T \tanh(\mathbf{W}_1\mathbf{h}_j + \mathbf{W}_2\mathbf{s}_i))$$

### Luong Attention (Multiplicative)
$$\alpha_{ij} = \text{softmax}(\mathbf{s}_i^T \mathbf{W} \mathbf{h}_j)$$

### Context Vector
$$\mathbf{c}_i = \sum_j \alpha_{ij}\mathbf{h}_j$$

## Contributing

See [CONTRIBUTING.md](../../../CONTRIBUTING.md).
