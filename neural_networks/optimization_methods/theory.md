# Gradient Descent Optimization: Complete Mathematical Theory

A rigorous treatment of gradient-based optimization methods for machine learning.

---

## 1. Problem Definition

### What Problem Is Being Solved?

Given a differentiable loss function $\mathcal{L}: \mathbb{R}^d \to \mathbb{R}$, find:
$$\theta^* = \arg\min_\theta \mathcal{L}(\theta)$$

### Why Is This Problem Non-Trivial?

1. **High dimensionality**: $d$ can be billions (modern neural networks)
2. **Non-convexity**: Many local minima and saddle points
3. **Stochasticity**: Only noisy gradient estimates available
4. **Ill-conditioning**: Different parameters need different step sizes
5. **Computational cost**: Cannot afford exact methods

---

## 2. Mathematical Formulation

### Gradient Descent Update

$$\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \mathcal{L}(\theta_t)$$

Where:
- $\eta_t$: Learning rate (step size)
- $\nabla_\theta \mathcal{L}$: Gradient of loss

### Stochastic Gradient Descent (SGD)

For minibatch $\mathcal{B}$ of size $b$:
$$g_t = \frac{1}{b}\sum_{i \in \mathcal{B}} \nabla_\theta \ell(\theta_t; x_i, y_i)$$
$$\theta_{t+1} = \theta_t - \eta_t g_t$$

**Properties**:
- $\mathbb{E}[g_t] = \nabla_\theta \mathcal{L}(\theta_t)$ (unbiased)
- $\text{Var}(g_t) = O(1/b)$ (decreases with batch size)

### Momentum

Add "velocity" term:
$$v_{t+1} = \mu v_t + g_t$$
$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$

Alternate form (Nesterov-style):
$$v_{t+1} = \mu v_t + \eta g_t$$
$$\theta_{t+1} = \theta_t - v_{t+1}$$

Typical: $\mu = 0.9$

---

## 3. Why This Formulation?

### Gradient as Steepest Descent

The gradient $\nabla \mathcal{L}$ points in the direction of steepest increase.

**Proof**: For unit vector $v$, the directional derivative is:
$$D_v \mathcal{L} = \nabla \mathcal{L} \cdot v = \|\nabla \mathcal{L}\| \cos\theta$$

Maximized when $v = \nabla \mathcal{L} / \|\nabla \mathcal{L}\|$.

Therefore, $-\nabla \mathcal{L}$ is the steepest descent direction.

### Why Stochastic?

Full gradient requires all $n$ samples:
$$\nabla \mathcal{L} = \frac{1}{n}\sum_{i=1}^n \nabla \ell_i$$

For $n = 10^9$, this is prohibitive.

Stochastic gradient: $O(b)$ instead of $O(n)$ per step.

### Why Momentum?

SGD oscillates in narrow valleys:
- High gradient perpendicular to valley
- Low gradient along valley

Momentum averages gradients → oscillations cancel, progress accumulates.

---

## 4. Derivation and Optimization

### Convergence Analysis

**For convex, L-smooth functions**:

$$\mathcal{L}(\theta_{t+1}) \leq \mathcal{L}(\theta_t) - \eta\|\nabla\mathcal{L}\|^2 + \frac{L\eta^2}{2}\|\nabla\mathcal{L}\|^2$$

With $\eta \leq 1/L$:
$$\mathcal{L}(\theta_t) - \mathcal{L}^* = O(1/t)$$

**For strongly convex (μ-strongly convex)**:
$$\|\theta_t - \theta^*\|^2 = O((1-\mu\eta)^t)$$

Linear convergence!

### Learning Rate Selection

**Too small**: Slow convergence, stuck in local minima
**Too large**: Divergence, oscillation
**Just right**: Fast convergence to good solution

**Learning rate schedules**:
- Constant: Simple but suboptimal
- Step decay: Reduce by factor every $k$ epochs
- Cosine annealing: Smooth decay with warm restarts
- Warmup: Start small, increase, then decay

### Numerical Stability

**Gradient clipping**: Prevent exploding gradients
$$g_t \leftarrow \min\left(1, \frac{c}{\|g_t\|}\right) g_t$$

**Weight decay**: Regularization
$$\theta_{t+1} = (1 - \lambda\eta)\theta_t - \eta g_t$$

---

## 5. Geometric Interpretation

### The Loss Landscape

Loss function defines a high-dimensional surface:
- **Valleys**: Long narrow regions (slow progress)
- **Plateaus**: Flat regions (vanishing gradients)
- **Saddle points**: Some directions up, some down
- **Local minima**: Gradient = 0, all directions up

### Curvature and Conditioning

**Hessian** $H = \nabla^2 \mathcal{L}$ captures curvature.

**Condition number**: $\kappa = \lambda_{max} / \lambda_{min}$
- Large $\kappa$: Narrow valleys, oscillation
- Small $\kappa$: Round bowl, easy optimization

**Newton's method** (second-order):
$$\theta_{t+1} = \theta_t - H^{-1} \nabla \mathcal{L}$$

One step for quadratic! But $O(d^3)$ to invert Hessian.

### Momentum as Ball Rolling

Imagine a ball rolling on the loss surface:
- Momentum = ball mass
- Gradient = slope
- Ball accelerates downhill, coasts through flat regions
- Overshoots sometimes but smooths trajectory

---

## 6. Probabilistic Interpretation

### Langevin Dynamics

Add noise to gradient descent:
$$\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L} + \sqrt{2\eta}\epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)$$

This samples from $p(\theta) \propto \exp(-\mathcal{L}(\theta))$!

SGD noise acts similarly → implicit regularization.

### Bayesian View

SGD explores the loss surface stochastically:
- Finds flat minima (low curvature)
- Flat minima generalize better
- Implicit bias toward simple solutions

### The Noise Scale

Effective "temperature" of SGD:
$$T \propto \frac{\eta}{b}$$

Large learning rate or small batch → more exploration.

---

## 7. Failure Modes and Limitations

### Saddle Points

In high dimensions, most critical points are saddle points.

Gradient = 0 but not a minimum:
- GD: Gets stuck
- SGD: Noise helps escape

### Sharp vs Flat Minima

**Sharp minima**: High curvature, poor generalization
**Flat minima**: Low curvature, good generalization

Large batch training finds sharp minima → worse test performance.

### Learning Rate Sensitivity

**Too low**: 
- Takes forever to converge
- May get stuck in bad local minimum

**Too high**:
- Diverges
- Oscillates without converging

### Vanishing/Exploding Gradients

In deep networks:
- Gradients multiply through layers
- Can shrink to 0 (vanishing) or grow to ∞ (exploding)

**Solutions**: Careful initialization, normalization, skip connections.

---

## 8. Scaling and Computational Reality

### Comparison of Optimizers

| Optimizer | Memory | Computation | Typical Use |
|-----------|--------|-------------|-------------|
| SGD | $O(d)$ | $O(d)$ per step | Simple, proven |
| SGD+Momentum | $O(2d)$ | $O(d)$ | Default for vision |
| Adam | $O(3d)$ | $O(d)$ | Default for NLP |
| LBFGS | $O(md)$ | $O(md)$ | Small models |

### Large Batch Training

Increase batch size with learning rate:
$$\eta_{large} = \eta_{small} \times \frac{b_{large}}{b_{small}}$$

**Linear scaling rule**: Works up to a point.

**Large batch challenges**:
- Finds sharp minima
- Need learning rate warmup
- Distributed communication overhead

### Distributed Training

**Synchronous SGD**:
1. Each worker computes gradient on local batch
2. Aggregate gradients (AllReduce)
3. Update parameters

**Asynchronous SGD**:
- No synchronization
- Faster but stale gradients
- Can diverge

---

## 9. Real-World Deployment Considerations

### Hyperparameter Tuning

Key hyperparameters:
- Learning rate (most important!)
- Momentum
- Batch size
- Learning rate schedule
- Weight decay

**Grid search**: Expensive but thorough
**Random search**: Often better per-trial
**Bayesian optimization**: Sample-efficient

### Common Training Recipe

```
1. Start with learning rate finder
2. Use warmup for first few epochs
3. Train with max LR
4. Decay toward end (cosine or step)
5. Use early stopping on validation
```

### Debugging Training

**Loss not decreasing**:
- Learning rate too high or too low
- Bug in data pipeline
- Wrong loss function

**Loss stuck**:
- Learning rate too small
- Bad initialization
- Gradient problems

**Overfitting**:
- Regularization (dropout, weight decay)
- More data (augmentation)
- Smaller model

---

## 10. Comparison With Alternatives

### SGD vs Adam

| Aspect | SGD | Adam |
|--------|-----|------|
| Convergence | Slower initially | Faster initially |
| Final solution | Often better generalization | May find sharp minima |
| Tuning | Learning rate critical | More robust |
| Memory | Less | More |

### Adaptive Learning Rate Methods

**AdaGrad**: Accumulate squared gradients, divide
$$\theta_t \propto g_t / \sqrt{\sum_{i=1}^t g_i^2}$$
Good for sparse; learning rate shrinks forever.

**RMSprop**: Exponential moving average
$$v_t = \beta v_{t-1} + (1-\beta)g_t^2$$
$$\theta_t \propto g_t / \sqrt{v_t}$$
Fixes AdaGrad's shrinking rate.

**Adam**: RMSprop + Momentum
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\theta_t \propto m_t / \sqrt{v_t}$$

### Second-Order Methods

- **Newton's method**: Uses Hessian, $O(d^3)$
- **Quasi-Newton (L-BFGS)**: Approximates Hessian
- **Natural gradient**: Uses Fisher information matrix
- **Shampoo**: Block-diagonal approximation

In practice, first-order methods dominate due to simplicity and scalability.

---

## 11. Mental Model Checkpoint

### Without Equations

Gradient descent is like walking downhill in fog:
- You can only feel the local slope (gradient)
- Take small steps in the steepest direction
- Eventually reach a valley (minimum)
- Momentum: Keep moving in your recent direction
- Adaptive rates: Take bigger steps where the ground is flat

**Analogy**: Ball rolling on a bumpy surface, eventually settling in a dip.

### With Equations

$$\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t)$$

With momentum: $v_{t+1} = \mu v_t + g_t$, $\theta_{t+1} = \theta_t - \eta v_{t+1}$

### Predict Behavior

1. **Learning rate 10x too large**: Divergence, NaN loss
2. **Learning rate 10x too small**: Slow training, may get stuck
3. **Momentum 0.99 vs 0.9**: Smoother but slower to adapt
4. **Batch size 32 → 256**: Need higher learning rate, faster epoch
5. **Adam vs SGD**: Adam converges faster initially, SGD often wins long-term

---

## References

### Classical
- Robbins & Monro (1951) - Stochastic approximation
- Rumelhart, Hinton, Williams (1986) - Backpropagation

### Modern
- Kingma & Ba (2014) - Adam optimizer
- Loshchilov & Hutter (2016) - SGDR (warm restarts)
- Goyal et al. (2017) - Large batch training

### Theory
- Bottou, Curtis, Nocedal (2018) - "Optimization methods for large-scale ML"
- Keskar et al. (2017) - Sharp minima and generalization
