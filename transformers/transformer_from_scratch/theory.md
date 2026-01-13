# Transformer Architecture: Complete Guide

## Overview

The Transformer is a neural network architecture that relies entirely on self-attention mechanisms, introduced in the seminal paper "Attention Is All You Need" (Vaswani et al., 2017).

---

## 1. Architecture Overview

```
                    ┌─────────────────────┐
                    │   Output Probabilities   │
                    └───────────┬─────────┘
                                │
                    ┌───────────┴───────────┐
                    │     Linear + Softmax      │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │      Decoder Stack        │
                    │   (N × Decoder Block)     │
                    └───────────┬───────────┘
                                │
              ┌────────────────┴────────────────┐
              │                                  │
    ┌─────────┴─────────┐            ┌─────────┴─────────┐
    │   Encoder Stack      │            │    Shifted Output     │
    │  (N × Encoder Block) │            │    Embeddings         │
    └─────────┬─────────┘            └─────────────────────┘
              │
    ┌─────────┴─────────┐
    │   Input Embeddings    │
    │  + Positional Enc.    │
    └───────────────────┘
```

---

## 2. Encoder Block

Each encoder block contains:

### 2.1 Multi-Head Self-Attention

$$\text{MultiHead}(X, X, X) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

### 2.2 Layer Normalization

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Where:
- $\mu = \frac{1}{d}\sum_{i=1}^d x_i$ (mean)
- $\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$ (variance)
- $\gamma, \beta$ are learnable parameters

### 2.3 Feed-Forward Network

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

Or with GELU (modern transformers):
$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

### 2.4 Residual Connections

$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

### Complete Encoder Block

```python
def encoder_block(x):
    # Self-attention with residual
    attn_out = multi_head_attention(x, x, x)
    x = layer_norm(x + attn_out)
    
    # FFN with residual
    ffn_out = feed_forward(x)
    x = layer_norm(x + ffn_out)
    
    return x
```

---

## 3. Decoder Block

Decoder blocks have three sublayers:

### 3.1 Masked Self-Attention
- Same as encoder self-attention
- Mask prevents attending to future positions

### 3.2 Cross-Attention
- Query from decoder, Key/Value from encoder output
- Allows decoder to attend to encoder representations

### 3.3 Feed-Forward Network
- Same as encoder

### Complete Decoder Block

```python
def decoder_block(x, encoder_output, causal_mask):
    # Masked self-attention
    self_attn = multi_head_attention(x, x, x, mask=causal_mask)
    x = layer_norm(x + self_attn)
    
    # Cross-attention
    cross_attn = multi_head_attention(x, encoder_output, encoder_output)
    x = layer_norm(x + cross_attn)
    
    # FFN
    ffn_out = feed_forward(x)
    x = layer_norm(x + ffn_out)
    
    return x
```

---

## 4. Mathematical Details

### 4.1 Embedding Layer

$$E = \text{Embed}(x) \cdot \sqrt{d_{model}}$$

Scaling by $\sqrt{d_{model}}$ ensures embeddings and positional encodings have similar magnitudes.

### 4.2 Attention Computation

For $h$ heads, each with dimension $d_k = d_{model}/h$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 4.3 Output Projection

$$P(\text{next token}) = \text{softmax}(xW_{out})$$

Where $W_{out} \in \mathbb{R}^{d_{model} \times |V|}$ and $|V|$ is vocabulary size.

---

## 5. Training

### 5.1 Loss Function

**Cross-entropy loss:**
$$\mathcal{L} = -\sum_{t=1}^T \log P(y_t | y_{<t}, X)$$

### 5.2 Label Smoothing

Instead of hard targets (0 or 1), use soft targets:
$$y_{smooth} = (1 - \epsilon) \cdot y_{hard} + \frac{\epsilon}{|V|}$$

### 5.3 Learning Rate Schedule

Warmup + inverse square root decay:
$$lr = d_{model}^{-0.5} \cdot \min(step^{-0.5}, step \cdot warmup^{-1.5})$$

---

## 6. Variants

### 6.1 Encoder-Only (BERT)
- Bidirectional attention
- Masked language modeling objective
- Good for: Classification, NER, QA

### 6.2 Decoder-Only (GPT)
- Causal (left-to-right) attention
- Next token prediction objective
- Good for: Text generation, completion

### 6.3 Encoder-Decoder (T5, BART)
- Full transformer architecture
- Sequence-to-sequence tasks
- Good for: Translation, summarization

---

## 7. Key Hyperparameters

| Parameter | BERT-base | GPT-2 | GPT-3 |
|-----------|-----------|-------|-------|
| Layers ($N$) | 12 | 12 | 96 |
| Hidden size ($d_{model}$) | 768 | 768 | 12288 |
| Attention heads ($h$) | 12 | 12 | 96 |
| FFN size | 3072 | 3072 | 49152 |
| Parameters | 110M | 117M | 175B |

---

## 8. Implementation Tips

### 8.1 Pre-LayerNorm vs Post-LayerNorm

**Post-LN (Original):**
$$x = x + \text{Sublayer}(x)$$
$$x = \text{LayerNorm}(x)$$

**Pre-LN (Modern, more stable):**
$$x = x + \text{Sublayer}(\text{LayerNorm}(x))$$

### 8.2 Initialization

- Xavier/Glorot for most weights
- Scale output projections by $1/\sqrt{2N}$ for $N$ layers
- Initialize bias to zero

### 8.3 Gradient Checkpointing

Trade compute for memory by recomputing activations during backward pass.

---

## 9. Resources

### Original Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT](https://arxiv.org/abs/1810.04805)
- [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT-3](https://arxiv.org/abs/2005.14165)

### Implementation Guides
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

### Video Courses
- [Stanford CS224N](https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ)
- [Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
