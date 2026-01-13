# Convolutional Neural Networks: Complete Mathematical Theory

A rigorous treatment of CNNs covering the mathematical foundations of spatial feature learning.

---

## 1. Problem Definition

### What Problem Is Being Solved?

Given input with **spatial structure** (images, audio, text), learn features that:
1. Are **translation invariant**: Pattern detection works everywhere
2. Exploit **local connectivity**: Nearby elements are related
3. Enable **hierarchical representation**: Simple features → complex concepts

### Why Is This Problem Non-Trivial?

1. **High dimensionality**: 224×224×3 image = 150,528 dimensions
2. **Spatial structure**: Pixels near each other are correlated
3. **Translation**: Object can be anywhere in image
4. **Hierarchy**: Edges → textures → parts → objects

**Fully connected networks fail** because they:
- Ignore spatial structure
- Have too many parameters ($O(d^2)$)
- Don't generalize across positions

---

## 2. Mathematical Formulation

### The Convolution Operation

For 2D input $X \in \mathbb{R}^{H \times W}$ and kernel $K \in \mathbb{R}^{k \times k}$:

$$(X * K)_{i,j} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} X_{i+m, j+n} \cdot K_{m,n}$$

**Terminology**:
- $K$: Kernel/filter (learned weights)
- $k$: Kernel size (e.g., 3×3)
- $(X * K)$: Feature map / activation map

### Multi-Channel Convolution

For input $X \in \mathbb{R}^{H \times W \times C_{in}}$ (e.g., RGB image, $C_{in}=3$):

$$Y_{i,j,c_{out}} = \sum_{c=1}^{C_{in}} \sum_{m,n} X_{i+m, j+n, c} \cdot K_{m,n,c,c_{out}} + b_{c_{out}}$$

**Parameters**:
- Kernel: $K \in \mathbb{R}^{k \times k \times C_{in} \times C_{out}}$
- Bias: $b \in \mathbb{R}^{C_{out}}$
- Total: $k^2 \cdot C_{in} \cdot C_{out} + C_{out}$

### Full CNN Layer

$$Z = X * K + b$$
$$H = \sigma(Z)$$

Where $\sigma$ is activation (ReLU, etc.).

### Pooling Operation

**Max pooling** with pool size $p$:
$$Y_{i,j,c} = \max_{m,n \in [0,p)} X_{pi+m, pj+n, c}$$

**Average pooling**:
$$Y_{i,j,c} = \frac{1}{p^2}\sum_{m,n \in [0,p)} X_{pi+m, pj+n, c}$$

---

## 3. Why This Formulation?

### The Three Properties of Convolution

**1. Sparse Connectivity**
Each output depends only on local region:
- FC layer: $O(H \cdot W \cdot C_{in})$ connections per output
- Conv layer: $O(k^2 \cdot C_{in})$ connections per output

For $k=3$, $H=W=224$: 9 vs 150,528 connections!

**2. Parameter Sharing**
Same kernel applied everywhere:
- FC layer: Different weights for each position
- Conv layer: Same weights, different positions

Parameters reduced from $O(d^2)$ to $O(k^2 \cdot C_{in} \cdot C_{out})$.

**3. Translation Equivariance**
If input shifts, output shifts the same way:
$$\tau_s(X) * K = \tau_s(X * K)$$

Pattern detection at position $(i,j)$ works at any position.

### What Breaks If Assumptions Fail?

| Assumption | Violation | Consequence |
|------------|-----------|-------------|
| Local patterns matter | Global context needed | Use attention, larger receptive field |
| Translation invariance | Position matters | Use coordinate convolution, attention |
| Hierarchy exists | Flat structure | Random features might work |

---

## 4. Derivation and Optimization

### Gradient Through Convolution

For loss $\mathcal{L}$, let $\delta = \frac{\partial \mathcal{L}}{\partial Y}$ (upstream gradient).

**Gradient w.r.t. kernel**:
$$\frac{\partial \mathcal{L}}{\partial K_{m,n,c,c'}} = \sum_{i,j} \delta_{i,j,c'} \cdot X_{i+m,j+n,c}$$

This is the **correlation** of input with upstream gradient.

**Gradient w.r.t. input**:
$$\frac{\partial \mathcal{L}}{\partial X_{i,j,c}} = \sum_{c'} \sum_{m,n} \delta_{i-m,j-n,c'} \cdot K_{m,n,c,c'}$$

This is **convolution with flipped kernel** — called "transposed convolution."

### Efficient Implementation

Convolution is computed as matrix multiplication:
1. **im2col**: Reshape input patches into matrix rows
2. **GEMM**: Matrix multiply with reshaped kernels
3. **col2im**: Reshape back to feature map

Modern GPUs are optimized for GEMM → convolution is fast.

### Receptive Field

The **receptive field** is the input region affecting an output:

For $L$ conv layers with kernel size $k$:
$$\text{RF} = 1 + L(k-1)$$

With stride $s$:
$$\text{RF} = 1 + \sum_{l=1}^L (k_l - 1) \prod_{i=1}^{l-1} s_i$$

**Example** (VGG-style):
- 3 layers of 3×3 conv: RF = 7×7
- Equivalent to one 7×7 conv but fewer parameters!

---

## 5. Geometric Interpretation

### Feature Maps as Representations

Each channel in a feature map detects one "concept":
- Early layers: Edges, colors, textures
- Middle layers: Parts, motifs
- Late layers: Objects, scenes

### The Manifold View

Images live on a low-dimensional manifold in pixel space:
- CNN learns a mapping from this manifold to feature space
- Each layer "untangles" the manifold
- Final layer: Linearly separable classes

### Invariance vs Equivariance

**Equivariant**: Feature map shifts with input (convolution)
**Invariant**: Output unchanged when input shifts (achieved by pooling + FC)

Full CNN path:
$$\text{Convolve (equivariant)} \to \text{Pool (local invariance)} \to \text{FC (global invariance)}$$

---

## 6. Probabilistic Interpretation

### CNN as Prior

Choosing CNN architecture encodes prior beliefs:
- Local correlations matter
- Same pattern can appear anywhere
- Hierarchy of abstraction exists

A Gaussian process with CNN-like kernel:
$$k(x, x') = \int K_\theta(x) K_\theta(x') d\theta$$

### Bayesian CNNs

Weight uncertainty can be modeled:
- Dropout as approximate inference
- Variational inference for posterior
- Uncertainty in predictions

### Information Theoretic View

Successive layers:
- **Compress** irrelevant information
- **Preserve** task-relevant information

InfoMax principle: Maximize $I(\text{output}; \text{label})$ while compressing $I(\text{hidden}; \text{input})$.

---

## 7. Failure Modes and Limitations

### Translation Sensitivity

Despite equivariance, small shifts can change predictions:
- Pooling and stride break exact equivariance
- Anti-aliasing (blur before pool) helps

### Adversarial Vulnerability

Small, imperceptible perturbations cause misclassification:
$$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L})$$

CNNs rely on texture more than shape → easily fooled.

### Data Efficiency

CNNs need lots of data:
- ImageNet: 1.2M images
- Insufficient data → overfitting

**Solutions**: Transfer learning, data augmentation, self-supervised pretraining.

### Global Context

Convolution has limited receptive field:
- Long-range dependencies missed
- **Solutions**: Dilated convolutions, attention, larger kernels

---

## 8. Scaling and Computational Reality

### Computational Cost

For conv layer: $O(H \cdot W \cdot C_{in} \cdot C_{out} \cdot k^2)$

**Bottlenecks**:
- Memory: Feature maps for all layers during backprop
- Compute: Matrix multiplications dominate

### Memory Optimization

**Gradient checkpointing**: Recompute instead of store
**Batch size**: Larger = faster training, more memory

### Efficient Architectures

| Architecture | Innovation |
|--------------|------------|
| MobileNet | Depthwise separable conv |
| EfficientNet | Compound scaling |
| ShuffleNet | Channel shuffle |

**Depthwise separable**: Instead of $C_{in} \cdot C_{out} \cdot k^2$, use $C \cdot k^2 + C_{in} \cdot C_{out}$.

---

## 9. Real-World Deployment Considerations

### Transfer Learning

Standard approach:
1. Pretrain on ImageNet (1000 classes)
2. Remove top classifier
3. Add new classifier for your task
4. Fine-tune

Works because early features (edges, textures) are universal.

### Data Augmentation

Critical for generalization:
- Random crop, flip, rotation
- Color jitter, cutout
- Mixup, CutMix
- AutoAugment (learned policies)

### Quantization and Pruning

For deployment on edge devices:
- INT8 quantization: 4× smaller, 2-4× faster
- Pruning: Remove small weights
- Knowledge distillation: Small model mimics large

---

## 10. Comparison With Alternatives

### CNN vs Vision Transformer (ViT)

| Aspect | CNN | ViT |
|--------|-----|-----|
| Inductive bias | Local, translation equivariant | Global, permutation equivariant |
| Data efficiency | Better with small data | Needs large data |
| Scaling | Diminishing returns | Scales better |
| Interpretability | Feature maps | Attention maps |

### CNN vs MLP-Mixer

MLP-Mixer: Only MLPs, no convolution or attention
- Competitive with proper training
- Suggests inductive bias less important at scale

### When CNNs Win

1. Limited data (strong inductive bias helps)
2. Edge deployment (efficient architectures)
3. Well-understood domain

### When Alternatives Win

- **Transformers**: Large data, long-range dependencies
- **Graph networks**: Non-grid structured data
- **Capsule networks**: Viewpoint invariance (experimental)

---

## 11. Mental Model Checkpoint

### Without Equations

A CNN is like a pattern detector that slides across an image:
- Small detector (kernel) looks for specific patterns (edges, curves)
- Same detector works everywhere (parameter sharing)
- Multiple detectors find different patterns
- Stack detectors: simple patterns combine into complex ones
- Pool: Summarize regions, throw away position

**Analogy**: Many flashlights (kernels) scanning a dark room, each tuned to different features.

### With Equations

$$(X * K)_{i,j} = \sum_{m,n} X_{i+m,j+n} \cdot K_{m,n}$$

Equivariance: $\tau_a(X) * K = \tau_a(X * K)$

### Predict Behavior

1. **Larger kernel size**: Bigger receptive field, more params
2. **More channels**: More patterns detected, more compute
3. **Pooling with stride 2**: Halves spatial resolution
4. **No pooling**: No spatial invariance, larger memory
5. **3 layers of 3×3 vs 1 layer of 7×7**: Same RF, fewer params

---

## References

### Foundational
- LeCun et al. (1998) - LeNet, "Gradient-based learning applied to document recognition"
- Krizhevsky et al. (2012) - AlexNet, ImageNet breakthrough

### Architecture Evolution
- Simonyan & Zisserman (2014) - VGG
- He et al. (2016) - ResNet
- Tan & Le (2019) - EfficientNet

### Theory
- Cohen & Welling (2016) - Group equivariant CNNs
- Goodfellow et al. - *Deep Learning* Chapter 9
