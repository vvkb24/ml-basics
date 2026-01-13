# Self-Attention

> **Status:** Stub - Implementation coming soon

## Overview

Self-attention computes relationships between all positions in a sequence.

## Key Concepts

### Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Query, Key, Value
- **Q (Query):** What I'm looking for
- **K (Key):** What I have to offer
- **V (Value):** What I actually give

### Multi-Head Attention
$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

## Contributing

See [CONTRIBUTING.md](../../../CONTRIBUTING.md).
