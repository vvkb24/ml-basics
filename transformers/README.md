# Transformers: From Theory to Implementation

This module provides a comprehensive, mathematically rigorous guide to Transformer architectures, from self-attention fundamentals to modern Large Language Models.

## 📚 Learning Path

| Order | Topic | Description |
|-------|-------|-------------|
| 1 | [Self-Attention](./self_attention/) | The core attention mechanism |
| 2 | [Positional Encoding](./positional_encoding/) | How transformers understand position |
| 3 | [Transformer Architecture](./transformer_from_scratch/) | Full encoder-decoder implementation |
| 4 | [LLM Fundamentals](./llm_fundamentals/) | Modern large language models |

## 🎯 What You'll Learn

### Mathematical Foundations
- Query, Key, Value intuition and derivation
- Scaled dot-product attention
- Multi-head attention mathematics
- Positional encoding (sinusoidal, RoPE, ALiBi)
- Layer normalization and residual connections

### Practical Implementations
- Self-attention from scratch (NumPy)
- Full Transformer encoder-decoder
- GPT-style decoder-only model
- Multiple positional encoding schemes

### Modern LLM Concepts
- Training pipelines (pretraining, SFT, RLHF)
- Inference optimization (KV cache, quantization)
- Prompting techniques (few-shot, CoT)

## 📁 Module Structure

```
transformers/
├── README.md                      # This file
├── self_attention/
│   ├── theory.md                  # Complete attention mathematics
│   ├── scratch.py                 # NumPy implementation
│   └── README.md
├── positional_encoding/
│   ├── theory.md                  # Sinusoidal, RoPE, ALiBi theory
│   ├── scratch.py                 # All encoding implementations
│   └── README.md
├── transformer_from_scratch/
│   ├── theory.md                  # Full architecture guide
│   ├── scratch.py                 # Complete transformer + GPT
│   └── README.md
└── llm_fundamentals/
    ├── theory.md                  # LLM training, scaling, RLHF
    ├── scratch.py                 # Minimal GPT implementation
    └── README.md
```

## 🚀 Quick Start

```python
# Run the self-attention demo
python self_attention/scratch.py

# Run the positional encoding demo
python positional_encoding/scratch.py

# Run the full transformer demo
python transformer_from_scratch/scratch.py

# Run the GPT demo
python llm_fundamentals/scratch.py
```

## 📖 Essential Resources

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [BERT](https://arxiv.org/abs/1810.04805) (Devlin et al., 2018)
- [GPT-3](https://arxiv.org/abs/2005.14165) (Brown et al., 2020)
- [LLaMA](https://arxiv.org/abs/2302.13971) (Touvron et al., 2023)

### Visual Guides
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Lilian Weng's Attention Guide](https://lilianweng.github.io/posts/2018-06-24-attention/)

### Video Courses
- [Stanford CS224N](https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ)
- [Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [3Blue1Brown: Attention](https://www.youtube.com/watch?v=eMlx5fFNoYc)

### Code Repositories
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Minimal GPT training
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [LLaMA](https://github.com/facebookresearch/llama)

## 🧮 Prerequisites

Before diving in, ensure familiarity with:
- Linear algebra (matrix multiplication, eigenvalues)
- Calculus (gradients, chain rule)
- Basic neural networks (MLPs, backpropagation)
- Python and NumPy

See [docs/math_prerequisites/](../docs/math_prerequisites/) for refreshers.
