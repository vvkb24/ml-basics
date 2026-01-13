# Self-Attention: Complete Mathematical Deep Dive

A research-level treatment of the attention mechanism, covering mathematical foundations, gradient analysis, computational complexity, and modern variants.

---

## Part I: Foundations

### 1. The Attention Mechanism: First Principles

#### 1.1 From Sequence-to-Sequence to Attention

**Problem**: In seq2seq with RNNs, the encoder compresses entire input into fixed-size vector:

$$h_{enc} = \text{RNN}_{enc}(x_1, ..., x_n)$$

This creates an **information bottleneck** for long sequences.

**Solution**: Allow decoder to "look back" at all encoder states:

$$c_t = \sum_{s=1}^{n} \alpha_{t,s} \cdot h_s$$

Where $\alpha_{t,s}$ is the attention weight from decoder step $t$ to encoder step $s$.

#### 1.2 Attention as Soft Dictionary Lookup

Think of attention as a differentiable dictionary:
- **Keys** ($K$): What does each position contain?
- **Queries** ($Q$): What am I looking for?
- **Values** ($V$): What information should I retrieve?

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\text{similarity}(Q, K)\right) \cdot V$$

Different similarity functions:
- **Dot-product**: $s(q, k) = q^T k$
- **Scaled dot-product**: $s(q, k) = \frac{q^T k}{\sqrt{d_k}}$
- **Additive**: $s(q, k) = v^T \tanh(W_q q + W_k k)$

---

### 2. Scaled Dot-Product Attention: Complete Derivation

#### 2.1 Matrix Formulation

For queries $Q \in \mathbb{R}^{n \times d_k}$, keys $K \in \mathbb{R}^{m \times d_k}$, values $V \in \mathbb{R}^{m \times d_v}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Dimensions**:
- $QK^T \in \mathbb{R}^{n \times m}$: Attention scores
- $\text{softmax}(\cdot) \in \mathbb{R}^{n \times m}$: Attention weights (rows sum to 1)
- $\text{Output} \in \mathbb{R}^{n \times d_v}$

#### 2.2 Per-Position Analysis

For query position $i$:

$$\text{output}_i = \sum_{j=1}^{m} \alpha_{ij} \cdot v_j$$

Where:
$$\alpha_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_k})}{\sum_{l=1}^{m} \exp(q_i \cdot k_l / \sqrt{d_k})}$$

**Properties**:
- $\alpha_{ij} \geq 0$ (non-negative)
- $\sum_{j=1}^{m} \alpha_{ij} = 1$ (normalized)
- High $q_i \cdot k_j$ → high $\alpha_{ij}$ → position $i$ attends to position $j$

#### 2.3 Why Scale by $\sqrt{d_k}$? Rigorous Analysis

**Theorem**: For $q, k \in \mathbb{R}^{d_k}$ with independent components $q_i, k_i \sim \mathcal{N}(0, 1)$:

$$\mathbb{E}[q \cdot k] = 0, \quad \text{Var}(q \cdot k) = d_k$$

**Proof**:
$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

Since $q_i, k_i$ are independent with mean 0:
$$\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$$

For variance:
$$\text{Var}(q_i k_i) = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1 \cdot 1 = 1$$

By independence:
$$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$

**Problem**: For large $d_k$, dot products have large variance. Softmax applied to large values:
- Near 1 for max
- Near 0 for others
- Gradients vanish!

**Solution**: Scale by $\sqrt{d_k}$ to get unit variance:
$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{\text{Var}(q \cdot k)}{d_k} = 1$$

---

### 3. Gradient Flow Through Attention

#### 3.1 Forward Pass Recap

$$S = \frac{QK^T}{\sqrt{d_k}}$$
$$A = \text{softmax}(S)$$
$$O = AV$$

#### 3.2 Backward Pass Derivation

Given $\frac{\partial \mathcal{L}}{\partial O}$, we need $\frac{\partial \mathcal{L}}{\partial Q}, \frac{\partial \mathcal{L}}{\partial K}, \frac{\partial \mathcal{L}}{\partial V}$.

**Gradient w.r.t. V**:
$$\frac{\partial \mathcal{L}}{\partial V} = A^T \frac{\partial \mathcal{L}}{\partial O}$$

**Gradient w.r.t. A**:
$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial O} V^T$$

**Gradient through Softmax**:

For softmax output $a_i = \text{softmax}(s)_i$:
$$\frac{\partial a_i}{\partial s_j} = a_i(\delta_{ij} - a_j)$$

Where $\delta_{ij}$ is Kronecker delta.

In matrix form:
$$\frac{\partial \mathcal{L}}{\partial S} = A \odot \left(\frac{\partial \mathcal{L}}{\partial A} - \text{rowsum}\left(A \odot \frac{\partial \mathcal{L}}{\partial A}\right)\right)$$

**Gradient w.r.t. Q and K**:
$$\frac{\partial \mathcal{L}}{\partial Q} = \frac{1}{\sqrt{d_k}} \frac{\partial \mathcal{L}}{\partial S} K$$
$$\frac{\partial \mathcal{L}}{\partial K} = \frac{1}{\sqrt{d_k}} \left(\frac{\partial \mathcal{L}}{\partial S}\right)^T Q$$

---

### 4. Multi-Head Attention: Deep Analysis

#### 4.1 Mathematical Formulation

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Each head operates in a subspace:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Parameters**:
- $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$
- $W^O \in \mathbb{R}^{hd_v \times d_{model}}$

Typically: $d_k = d_v = d_{model}/h$.

#### 4.2 Why Multiple Heads Work

**Intuition 1: Different Relationship Types**

Head 1: 
$$W_1^Q, W_1^K \text{ project to subspace capturing syntactic relationships}$$

Head 2:
$$W_2^Q, W_2^K \text{ project to subspace capturing semantic relationships}$$

**Intuition 2: Ensemble Effect**

$$\text{MultiHead} = \sum_{i=1}^{h} \text{head}_i W_i^O$$

Each head is an "expert"; output combines their opinions.

**Intuition 3: Rank Increase**

Single attention has limited rank:
$$\text{rank}(AV) \leq \min(\text{rank}(A), \text{rank}(V))$$

Multi-head can achieve higher effective rank through concatenation.

#### 4.3 Parameter Count

Single head: $3d^2 + d^2 = 4d^2$
Multi-head ($h$ heads): $h \cdot 3(d/h)^2 + d^2 = 3d^2/h + d^2 ≈ 4d^2$

Same parameter count, more expressivity!

---

## Part II: Computational Analysis

### 5. Complexity Analysis

#### 5.1 Time Complexity

| Operation | FLOPs |
|-----------|-------|
| $Q = XW^Q$ | $O(nd^2)$ |
| $K = XW^K$ | $O(nd^2)$ |
| $V = XW^V$ | $O(nd^2)$ |
| $QK^T$ | $O(n^2 d)$ |
| $\text{softmax}$ | $O(n^2)$ |
| $AV$ | $O(n^2 d)$ |

**Total**: $O(n^2 d + nd^2)$

For typical transformer layers where $n \approx d$: $O(n^2 d)$

#### 5.2 Memory Complexity

- Store $Q, K, V$: $O(nd)$
- Store attention matrix $A$: $O(n^2)$ — **bottleneck!**
- Store output: $O(nd)$

**Total**: $O(n^2 + nd)$

The $O(n^2)$ attention matrix is the main memory bottleneck.

#### 5.3 Implications for Long Sequences

For sequence length 4096 with FP16:
- Attention matrix: $4096^2 \times 2 = 33.5 \text{ MB}$ per head
- With 32 heads: ~1 GB just for attention!

This motivates efficient attention mechanisms.

---

### 6. Efficient Attention Mechanisms

#### 6.1 Flash Attention

**Key insight**: Compute attention block-by-block, never materializing full $n \times n$ matrix.

**Algorithm**:
1. Tile $Q, K, V$ into blocks
2. For each $Q$ block:
   - Compute partial attention with each $K, V$ block
   - Accumulate results using online softmax
3. Never store full attention matrix

**Complexity**: Same time $O(n^2d)$, but memory $O(n)$!

#### 6.2 Linear Attention

Replace softmax with kernel feature maps:
$$\text{Attention}(Q, K, V) = \phi(Q)(\phi(K)^T V)$$

**Complexity**: $O(nd^2)$ instead of $O(n^2d)$!

Examples: Performer, Linear Transformer.

#### 6.3 Sparse Attention

Only compute attention for subset of positions:

**Local attention**: Each position attends to window of $w$ neighbors
$$\text{Complexity}: O(nwd)$$

**Strided attention**: Attend to every $k$-th position
$$\text{Complexity}: O(n^2d/k)$$

**Combination** (Longformer, BigBird):
- Local window + global tokens + random connections

---

## Part III: Attention Variants

### 7. Cross-Attention

Used in encoder-decoder models:

$$\text{CrossAttention}(X_{dec}, X_{enc}) = \text{Attention}(X_{dec}W^Q, X_{enc}W^K, X_{enc}W^V)$$

- Query from decoder
- Key, Value from encoder
- Allows decoder to "look at" encoder representations

### 8. Causal (Masked) Attention

For autoregressive generation, prevent looking at future:

$$\tilde{S}_{ij} = \begin{cases} S_{ij} & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

After softmax, future positions have zero weight.

### 9. Grouped Query Attention (GQA)

Share K, V projections across groups of heads:

```
Standard MHA:  32 Q heads, 32 K heads, 32 V heads
GQA:           32 Q heads,  8 K heads,  8 V heads
MQA:           32 Q heads,  1 K head,   1 V head
```

**Benefit**: Smaller KV cache during inference.

LLaMA 2 70B uses GQA with 8 KV heads.

### 10. Sliding Window Attention

Each position only attends to $w$ previous positions:

$$A_{ij} = 0 \text{ if } |i - j| > w/2$$

**Benefit**: $O(nw)$ complexity.

Used in Mistral, combined with interleaved full attention layers.

---

## Part IV: Theoretical Analysis

### 11. Attention as Kernel Smoother

Attention can be viewed as a **kernel density estimator**:

$$\text{output}_i = \frac{\sum_j K(q_i, k_j) v_j}{\sum_j K(q_i, k_j)}$$

Where the kernel is:
$$K(q, k) = \exp\left(\frac{q^T k}{\sqrt{d_k}}\right)$$

This is a **Nadaraya-Watson estimator** with learned queries and keys.

### 12. Attention as Gradient Descent

**Theorem** (von Oswald et al., 2022): Self-attention can implement one step of gradient descent on a linear regression objective.

Given training data $(X, Y)$ in context, for query $x_q$:
$$\text{Attention outputs} \approx X(X^TX)^{-1}X^T Y$$

This is exactly the least-squares solution!

### 13. Rank of Attention Outputs

**Observation**: Attention outputs have bounded rank.

$$\text{rank}(\text{Attention}(Q, K, V)) \leq \min(n, d_v)$$

This limits expressivity. Multi-head attention helps by combining subspaces.

### 14. Attention and Turing Completeness

**Theorem** (Pérez et al., 2019): Transformers with hard attention (argmax instead of softmax) are Turing complete.

Soft attention transformers are computationally universal under reasonable assumptions.

---

## Resources

### Foundational Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original transformer
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — Original attention
- [Flash Attention](https://arxiv.org/abs/2205.14135) — Memory-efficient attention

### Analysis Papers
- [What Can Transformers Learn In-Context?](https://arxiv.org/abs/2208.01066)
- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- [Are Transformers Universal Approximators?](https://arxiv.org/abs/1912.10077)

### Visual Resources
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)
- [3Blue1Brown: Attention in Transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc)
