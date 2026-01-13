# Perceptron

> **Status:** Stub - Implementation coming soon

## Overview

The perceptron is the simplest neural network unit—a linear classifier.

## Key Concepts

### Model
$$\hat{y} = \text{sign}(\mathbf{w}^T\mathbf{x} + b)$$

### Update Rule (Perceptron Learning Algorithm)
$$\mathbf{w} \leftarrow \mathbf{w} + y_i\mathbf{x}_i \quad \text{if } y_i(\mathbf{w}^T\mathbf{x}_i) \leq 0$$

### Convergence
Converges in finite steps if data is linearly separable.

## Contributing

See [CONTRIBUTING.md](../../../CONTRIBUTING.md).
