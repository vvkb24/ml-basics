# Recurrent Neural Networks (RNN)

> **Status:** Stub - Implementation coming soon

## Overview

RNNs process sequential data by maintaining hidden state.

## Key Concepts

### Hidden State Update
$$\mathbf{h}_t = \tanh(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b})$$

### Backpropagation Through Time (BPTT)
Unroll RNN and apply standard backprop.

### Issues
- Vanishing/exploding gradients
- Limited long-range dependencies

## Contributing

See [CONTRIBUTING.md](../../../CONTRIBUTING.md).
