# Positional Encoding

> **Status:** Stub - Implementation coming soon

## Overview

Inject position information since attention is permutation-invariant.

## Sinusoidal Encoding

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

## Other Methods
- Learned positional embeddings
- Rotary Position Embedding (RoPE)
- ALiBi (Attention with Linear Biases)

## Contributing

See [CONTRIBUTING.md](../../../CONTRIBUTING.md).
