# Large Language Models: Advanced Mathematical Guide

A comprehensive, research-level treatment of Large Language Models covering architecture, training dynamics, optimization, and theoretical foundations.

---

## Part I: Mathematical Foundations

### 1. Language Modeling: Formal Definition

#### 1.1 Probabilistic Framework

A language model defines a probability distribution over sequences of tokens from vocabulary $\mathcal{V}$:

$$P_\theta(x_1, x_2, ..., x_T) = \prod_{t=1}^{T} P_\theta(x_t | x_{<t})$$

Where:
- $x_t \in \mathcal{V}$ is the token at position $t$
- $x_{<t} = (x_1, ..., x_{t-1})$ is the context
- $\theta$ are model parameters

#### 1.2 Cross-Entropy Loss

Training minimizes the negative log-likelihood:

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(x_t | x_{<t})$$

This is equivalent to minimizing cross-entropy between the true distribution $P_{data}$ and model distribution $P_\theta$:

$$H(P_{data}, P_\theta) = -\mathbb{E}_{x \sim P_{data}}[\log P_\theta(x)]$$

#### 1.3 Perplexity

Perplexity measures how "surprised" the model is:

$$\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(x_t | x_{<t})\right) = \exp(\mathcal{L})$$

**Interpretation**:
- PPL = 1: Perfect prediction
- PPL = $|\mathcal{V}|$: Random guessing
- Lower is better

---

### 2. Transformer Architecture: Deep Dive

#### 2.1 Input Representation

For input sequence $x = (x_1, ..., x_T)$:

$$H^{(0)} = \text{Embed}(x) + \text{PE}$$

Where:
- $\text{Embed}(x_t) = E_{x_t} \in \mathbb{R}^d$ (lookup from embedding matrix $E \in \mathbb{R}^{|\mathcal{V}| \times d}$)
- $\text{PE} \in \mathbb{R}^{T \times d}$ is positional encoding

#### 2.2 Self-Attention: Complete Derivation

**Step 1: Linear Projections**

For input $H \in \mathbb{R}^{T \times d}$:

$$Q = HW^Q, \quad K = HW^K, \quad V = HW^V$$

Where $W^Q, W^K \in \mathbb{R}^{d \times d_k}$ and $W^V \in \mathbb{R}^{d \times d_v}$.

**Step 2: Attention Scores**

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{T \times T}$$

Element-wise:
$$A_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_k})}{\sum_{l=1}^{T} \exp(q_i \cdot k_l / \sqrt{d_k})}$$

**Step 3: Weighted Aggregation**

$$\text{Attention}(Q, K, V) = AV$$

Output row $i$:
$$\text{out}_i = \sum_{j=1}^{T} A_{ij} v_j$$

**Why $\sqrt{d_k}$ Scaling?**

Without scaling, for random $q, k \sim \mathcal{N}(0, 1)$:

$$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$

Large variance → softmax saturates → vanishing gradients. Scaling normalizes variance to 1.

#### 2.3 Multi-Head Attention: Why Multiple Heads?

**Mathematical Motivation**

Single attention computes:
$$\text{head} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The softmax creates a **convex combination** of values. This limits expressivity because:
1. Each position can only attend to a weighted average
2. Cannot simultaneously capture different relationship types

**Multi-Head Solution**

$$\text{MultiHead}(H) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Each head uses different projections:
$$\text{head}_i = \text{Attention}(HW_i^Q, HW_i^K, HW_i^V)$$

**Information-Theoretic View**

Each head can encode different "subspaces" of information:
- Head 1: Syntactic relationships (subject-verb agreement)
- Head 2: Semantic relationships (synonyms, antonyms)
- Head 3: Positional patterns (nearby tokens)

Empirically, attention heads learn interpretable patterns.

#### 2.4 Feed-Forward Networks: Expressivity Analysis

**Standard FFN**

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

Where $W_1 \in \mathbb{R}^{d \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d}$, typically $d_{ff} = 4d$.

**Why Large FFN?**

The FFN can be viewed as a **key-value memory**:
- $W_1$ columns are "keys" (patterns to match)
- $W_2$ columns are "values" (information to retrieve)
- ReLU selects which keys are activated

**SwiGLU Activation (Modern LLMs)**

$$\text{SwiGLU}(x, W, V, W_2) = (\text{Swish}(xW) \odot xV)W_2$$

Where $\text{Swish}(x) = x \cdot \sigma(x)$ and $\odot$ is element-wise product.

More expressive than ReLU, used in LLaMA, PaLM.

#### 2.5 Layer Normalization: Deep Analysis

**Definition**

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Where:
$$\mu = \frac{1}{d}\sum_{i=1}^{d} x_i, \quad \sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2$$

**Why LayerNorm Works**

1. **Normalization**: Keeps activations in stable range
2. **Re-centering**: $\gamma, \beta$ allow model to learn optimal distribution
3. **Gradient flow**: Normalized gradients prevent exploding/vanishing

**Pre-LN vs Post-LN**

Post-LN (original):
$$x = \text{LayerNorm}(x + \text{Sublayer}(x))$$

Pre-LN (modern, more stable):
$$x = x + \text{Sublayer}(\text{LayerNorm}(x))$$

Pre-LN has better gradient flow, enabling deeper models without warmup.

**RMSNorm (LLaMA, Gemma)**

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma$$

Removes mean subtraction → 15% faster, similar performance.

---

## Part II: Training Dynamics

### 3. Pretraining

#### 3.1 Next Token Prediction

**Objective:**
$$\mathcal{L}_{LM} = -\mathbb{E}_{x \sim \mathcal{D}}\left[\sum_{t=1}^{T} \log P_\theta(x_t | x_{<t})\right]$$

**Causal Masking:**

Attention matrix $A$ is masked:
When $j \leq i$: $\tilde{A}_{ij} = A_{ij}$, otherwise $\tilde{A}_{ij} = -\infty$

This ensures $P(x_t | x_{<t})$ only depends on past tokens.

#### 3.2 Training Data Composition

Modern LLMs train on diverse mixtures:

| Source | Example | Weight |
|--------|---------|--------|
| Web text | Common Crawl | 60% |
| Books | Books3, Gutenberg | 15% |
| Wikipedia | All languages | 5% |
| Code | GitHub, StackOverflow | 10% |
| Scientific | ArXiv, PubMed | 5% |
| Curated | InstructGPT data | 5% |

**Data Quality Matters**

Chinchilla finding: Smaller model + more tokens beats larger model + fewer tokens at fixed compute.

$$\text{Optimal}: N \propto C^{0.5}, \quad D \propto C^{0.5}$$

Rule: ~20 tokens per parameter.

#### 3.3 Optimization

**Adam Optimizer**

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\hat{m}_t = m_t/(1-\beta_1^t), \quad \hat{v}_t = v_t/(1-\beta_2^t)$$
$$\theta_t = \theta_{t-1} - \alpha \cdot \hat{m}_t/(\sqrt{\hat{v}_t} + \epsilon)$$

Typical: $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 10^{-8}$.

**AdamW (Weight Decay)**

$$\theta_t = \theta_{t-1} - \alpha \cdot (\hat{m}_t/(\sqrt{\hat{v}_t} + \epsilon) + \lambda\theta_{t-1})$$

Decouples weight decay from gradient-based updates.

**Learning Rate Schedule**

Warmup + cosine decay:

For $t < T_{warmup}$:
$$\eta(t) = \eta_{max} \cdot \frac{t}{T_{warmup}}$$

Otherwise:
$$\eta(t) = \eta_{min} + \frac{\eta_{max} - \eta_{min}}{2}\left(1 + \cos\left(\frac{\pi(t - T_{warmup})}{T_{total} - T_{warmup}}\right)\right)$$

Typical: warmup = 2000 steps, final LR = 0.1 × peak.

#### 3.4 Gradient Accumulation & Distributed Training

**Effective Batch Size**

$$B_{eff} = B_{micro} \times \text{accumulation\_steps} \times \text{num\_gpus}$$

Large batch (millions of tokens) is critical for stable training.

**3D Parallelism**

1. **Data Parallel**: Same model, different data
2. **Tensor Parallel**: Split layers across GPUs
3. **Pipeline Parallel**: Different layers on different GPUs

**ZeRO (Zero Redundancy Optimizer)**

Stage 1: Partition optimizer states
Stage 2: + Partition gradients
Stage 3: + Partition parameters

Enables training 100B+ models.

---

## Part III: Alignment

### 4. Supervised Fine-Tuning (SFT)

#### 4.1 Instruction Following

Train on (instruction, response) pairs:

$$\mathcal{L}_{SFT} = -\sum_{t=1}^{T_{response}} \log P_\theta(y_t | x_{instruction}, y_{<t})$$

Only compute loss on response tokens, not instruction.

#### 4.2 Data Quality for SFT

Key principles:
1. **Diversity**: Cover many tasks
2. **Quality**: Human-written or model-generated + filtered
3. **Formatting**: Consistent structure

Example datasets: FLAN, Alpaca, ShareGPT.

---

### 5. Reinforcement Learning from Human Feedback (RLHF)

#### 5.1 Reward Modeling

Train reward model $R_\phi(x, y)$ from human preferences:

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma(R_\phi(x, y_w) - R_\phi(x, y_l))\right]$$

Where:
- $y_w$ = preferred response
- $y_l$ = rejected response
- $\sigma$ = sigmoid function

This is the **Bradley-Terry model** of pairwise preferences.

#### 5.2 Policy Optimization (PPO)

Maximize reward while staying close to SFT model:

$$\mathcal{L}_{PPO} = \mathbb{E}_{x, y \sim \pi_\theta}\left[R_\phi(x, y)\right] - \beta \cdot D_{KL}[\pi_\theta(y|x) || \pi_{ref}(y|x)]$$

**PPO Objective:**

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

Where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ and $\hat{A}_t$ is advantage estimate.

#### 5.3 KL Divergence Constraint

$$D_{KL}[\pi_\theta || \pi_{ref}] = \mathbb{E}_{y \sim \pi_\theta}\left[\log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}\right]$$

Prevents:
- **Reward hacking**: Exploiting reward model weaknesses
- **Mode collapse**: Generating only one type of response

---

### 6. Direct Preference Optimization (DPO)

#### 6.1 Theoretical Foundation

DPO eliminates the reward model by deriving the optimal policy directly:

**Key insight**: Under the RLHF objective, the optimal policy satisfies:

$$\pi^*(y|x) = \frac{1}{Z(x)}\pi_{ref}(y|x)\exp\left(\frac{1}{\beta}R^*(x, y)\right)$$

Rearranging:
$$R^*(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

#### 6.2 DPO Loss

Substituting into Bradley-Terry model:

$$\mathcal{L}_{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

**Interpretation**: Increase probability of preferred responses, decrease probability of rejected responses, relative to reference model.

**Advantages over RLHF**:
- No reward model needed
- No RL instability
- Simpler implementation
- Single training stage

---

## Part IV: Inference

### 7. Efficient Inference

#### 7.1 KV Cache

During autoregressive generation, cache computed keys and values:

```
Step 1: Compute K_1, V_1 for token 1
Step 2: Compute K_2, V_2, use cached K_1, V_1
Step n: Compute K_n, V_n, use cached K_{1:n-1}, V_{1:n-1}
```

**Memory**: $O(n \cdot L \cdot d)$ where $L$ = layers, $n$ = sequence length.

#### 7.2 Grouped Query Attention (GQA)

Reduce KV cache by sharing K, V across heads:

| Type | Query heads | KV heads | KV cache |
|------|-------------|----------|----------|
| MHA | 32 | 32 | 100% |
| GQA | 32 | 8 | 25% |
| MQA | 32 | 1 | 3% |

LLaMA 2 70B uses GQA with 8 KV heads.

#### 7.3 Quantization

**Post-Training Quantization (PTQ)**

Map weights from FP16/FP32 to INT8/INT4:

$$W_{quant} = \text{round}\left(\frac{W - \text{min}(W)}{\text{max}(W) - \text{min}(W)} \times (2^b - 1)\right)$$

**QLoRA**: Quantize base model to 4-bit, fine-tune with LoRA adapters.

#### 7.4 Speculative Decoding

Use small "draft" model to propose $k$ tokens:
1. Draft model generates $x_1, ..., x_k$ quickly
2. Large model verifies in parallel
3. Accept prefix that matches, reject rest

Speedup: ~2-3x with 10-20% overhead.

---

### 8. Decoding Strategies

#### 8.1 Greedy Decoding

$$x_t = \arg\max_{x} P_\theta(x | x_{<t})$$

Fast but often repetitive/boring.

#### 8.2 Beam Search

Maintain $k$ best partial sequences:

$$\text{score}(x_{1:t}) = \sum_{i=1}^{t} \log P_\theta(x_i | x_{<i})$$

Better for translation, worse for open-ended generation.

#### 8.3 Top-k Sampling

Sample from top $k$ tokens:

If $x_t \in \text{top-}k$: $P_{top-k}(x_t | x_{<t}) \propto P_\theta(x_t | x_{<t})$, otherwise $P_{top-k}(x_t | x_{<t}) = 0$

#### 8.4 Nucleus (Top-p) Sampling

Sample from smallest set with cumulative probability $\geq p$:

$$V_p = \text{smallest } V \text{ s.t. } \sum_{x \in V} P_\theta(x | x_{<t}) \geq p$$

More adaptive than top-k.

#### 8.5 Temperature Scaling

$$P_{temp}(x_t | x_{<t}) = \frac{\exp(\text{logit}_t / \tau)}{\sum_{x'} \exp(\text{logit}_{x'} / \tau)}$$

- $\tau < 1$: Sharper (more deterministic)
- $\tau > 1$: Flatter (more random)
- $\tau = 0$: Greedy

---

## Part V: Advanced Topics

### 9. Emergent Abilities

Abilities that appear suddenly at scale:

| Ability | Emerges at |
|---------|------------|
| Few-shot learning | ~6B params |
| Chain-of-thought | ~60B params |
| Instruction following | ~1B + SFT |
| Code generation | ~10B params |

**Hypothesis**: Phase transitions in capability as model capacity crosses task thresholds.

### 10. In-Context Learning

LLMs can learn from examples in the prompt without weight updates:

$$P(y | x, (x_1, y_1), ..., (x_k, y_k))$$

**Theoretical analysis**: Transformers can implement gradient descent implicitly through attention.

### 11. Mechanistic Interpretability

Understanding how models compute:

- **Induction heads**: Copy patterns from context
- **Attention patterns**: Positional, syntactic, semantic
- **MLP as memory**: Key-value lookup

### 12. Constitutional AI (Claude)

Self-improvement through AI feedback:
1. Generate response
2. Critique against principles
3. Revise response
4. Train on improved responses

---

## Resources

### Foundational Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)
- [GPT-3](https://arxiv.org/abs/2005.14165) (2020)
- [Chinchilla Scaling Laws](https://arxiv.org/abs/2203.15556) (2022)
- [InstructGPT](https://arxiv.org/abs/2203.02155) (2022)
- [DPO](https://arxiv.org/abs/2305.18290) (2023)
- [LLaMA](https://arxiv.org/abs/2302.13971) (2023)

### Technical Deep Dives
- [The Transformer Family v2](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [Rotary Position Embeddings](https://arxiv.org/abs/2104.09864)

### Courses
- [Stanford CS324: Large Language Models](https://stanford-cs324.github.io/winter2022/)
- [Princeton COS 597G: Understanding LLMs](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/)
- [Karpathy: Neural Networks Zero to Hero](https://karpathy.ai/zero-to-hero.html)

### Books
- "Speech and Language Processing" - Jurafsky & Martin (ch. 10-11)
- "Deep Learning" - Goodfellow, Bengio, Courville
- "Understanding Deep Learning" - Simon Prince (2023)
