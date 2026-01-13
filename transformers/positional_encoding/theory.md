# Positional Encoding: Mathematical Deep Dive

## Overview

Transformers have no inherent notion of sequence order. Positional encoding injects position information into token embeddings, enabling the model to understand word order.

---

## 1. The Problem: Why Position Matters

Consider these sentences:
- "The cat sat on the mat"
- "The mat sat on the cat"

Same words, different meanings! Without position information, self-attention treats these identically.

### Why Attention is Position-Agnostic

Attention computes: $\text{Attention}(Q, K, V) = \text{softmax}(QK^T)V$

This is a **set operation** — permuting input tokens just permutes outputs. There's no notion of "first" or "last" token.

---

## 2. Sinusoidal Positional Encoding

The original Transformer uses sine and cosine functions of different frequencies.

### 2.1 Mathematical Formulation

For position $pos$ and dimension $i$:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Where:
- $pos$ = position in sequence (0, 1, 2, ...)
- $i$ = dimension index (0 to $d_{model}/2 - 1$)
- $d_{model}$ = embedding dimension

### 2.2 Why Sinusoids?

**Property 1: Bounded values**
$$-1 \leq PE_{(pos, i)} \leq 1$$
Prevents position from dominating token embeddings.

**Property 2: Unique encoding**
Each position gets a unique encoding due to different wavelengths.

**Property 3: Relative position as linear function**
For any fixed offset $k$:
$$PE_{pos+k} = f(PE_{pos})$$

This is because:
$$\sin(a + b) = \sin(a)\cos(b) + \cos(a)\sin(b)$$

The model can learn to attend to relative positions!

### 2.3 Wavelength Interpretation

For dimension $i$, the wavelength is:
$$\lambda_i = 2\pi \cdot 10000^{2i/d_{model}}$$

- Low dimensions ($i$ small): Short wavelengths, fast oscillation
- High dimensions ($i$ large): Long wavelengths, slow oscillation

This creates a **position spectrum** from fine-grained to coarse-grained.

---

## 3. Learned Positional Embeddings

Alternative: Learn a position embedding matrix.

$$PE = \text{Embedding}(pos) \in \mathbb{R}^{max\_len \times d_{model}}$$

**Pros:**
- More flexible, can learn complex patterns
- Used in BERT, GPT-2

**Cons:**
- Limited to maximum sequence length seen during training
- More parameters

---

## 4. Rotary Position Embedding (RoPE)

Modern approach used in LLaMA, GPT-NeoX, PaLM.

### 4.1 Key Idea

Instead of adding position to embeddings, **rotate** them based on position.

### 4.2 Mathematical Formulation

For a 2D subspace, apply rotation:

$$\begin{pmatrix} q'_0 \\ q'_1 \end{pmatrix} = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix}$$

Where $m$ is the position and $\theta$ is a learnable or fixed angle.

For full $d$-dimensional vectors, apply rotations to pairs:
$$(q_0, q_1), (q_2, q_3), ..., (q_{d-2}, q_{d-1})$$

### 4.3 Why RoPE Works

The dot product of rotated queries and keys depends only on **relative** position:

$$q_m^T k_n = f(m - n)$$

This naturally encodes relative position in attention!

### 4.4 RoPE Formula

$$f_q(x_m, m) = \begin{pmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \\ \vdots \end{pmatrix} \otimes \begin{pmatrix} \cos m\theta_1 \\ \cos m\theta_1 \\ \cos m\theta_2 \\ \cos m\theta_2 \\ \vdots \end{pmatrix} + \begin{pmatrix} -x_2 \\ x_1 \\ -x_4 \\ x_3 \\ \vdots \end{pmatrix} \otimes \begin{pmatrix} \sin m\theta_1 \\ \sin m\theta_1 \\ \sin m\theta_2 \\ \sin m\theta_2 \\ \vdots \end{pmatrix}$$

---

## 5. Relative Position Bias

Used in T5, Swin Transformer.

### 5.1 Approach

Add learnable bias to attention scores based on relative position:

$$A_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} + b_{i-j}$$

Where $b_{i-j}$ is a learned bias for relative distance $i-j$.

### 5.2 Advantages
- Generalizes to longer sequences than training
- Directly models relationships between positions

---

## 6. ALiBi (Attention with Linear Biases)

Used in BLOOM, MPT.

### 6.1 Formula

Add linear penalty based on distance:

$$A_{ij} = q_i \cdot k_j - m \cdot |i - j|$$

Where $m$ is a head-specific slope. Closer tokens have higher attention scores.

### 6.2 Properties
- No learned parameters for position
- Excellent length extrapolation
- Simple to implement

---

## 7. Comparison

| Method | Parameters | Length Generalization | Relative Position |
|--------|------------|----------------------|-------------------|
| Sinusoidal | 0 | Moderate | Implicit |
| Learned | $L \times d$ | Poor | No |
| RoPE | 0 | Good | Explicit |
| Relative Bias | $O(L)$ | Good | Explicit |
| ALiBi | $h$ slopes | Excellent | Explicit |

---

## 8. Implementation Considerations

### 8.1 Adding vs. Concatenating

- **Adding**: $\text{Embed}(x) + PE(pos)$
  - Standard approach
  - Position and content share same space

- **Concatenating**: $[\text{Embed}(x); PE(pos)]$
  - Doubles dimension
  - Keeps position separate

### 8.2 When to Apply

- Before attention: Most common
- Only to Q, K (not V): Used with RoPE
- As attention bias: Relative methods

---

## 9. Resources

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Sinusoidal encoding
- [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864) — Rotary embeddings
- [ALiBi](https://arxiv.org/abs/2108.12409) — Linear biases
- [Train Short, Test Long](https://arxiv.org/abs/2108.12409) — Position extrapolation

### Blog Posts
- [Rotary Embeddings (EleutherAI)](https://blog.eleuther.ai/rotary-embeddings/)
- [Positional Encoding Tutorial](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
- [A Gentle Introduction to Positional Encoding](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/)
