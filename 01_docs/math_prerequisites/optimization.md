# Optimization for Machine Learning

Optimization is the process of finding parameters that minimize (or maximize) an objective function.

---

## 1. Problem Formulation

**Optimization Problem:**
$$\min_{\boldsymbol{\theta}} f(\boldsymbol{\theta})$$

In ML, $f$ is typically a **loss function** and $\boldsymbol{\theta}$ are model parameters.

**With constraints:**
$$\min_{\boldsymbol{\theta}} f(\boldsymbol{\theta}) \quad \text{s.t.} \quad g_i(\boldsymbol{\theta}) \leq 0, \quad h_j(\boldsymbol{\theta}) = 0$$

---

## 2. Convexity

### Convex Functions

A function $f$ is **convex** if:
$$f(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda)f(\mathbf{y})$$

for all $\mathbf{x}, \mathbf{y}$ and $\lambda \in [0, 1]$.

**Equivalently:** The Hessian $\mathbf{H} \succeq 0$ (positive semi-definite)

### Why Convexity Matters
- **Local minimum = Global minimum**
- Gradient descent converges to optimal solution
- No local minima traps

### Examples

| Function | Convex? |
|----------|---------|
| $f(x) = x^2$ | ✅ Yes |
| MSE Loss | ✅ Yes |
| Cross-entropy | ✅ Yes (in parameters) |
| Neural network loss | ❌ No (highly non-convex) |

---

## 3. Gradient Descent

### Algorithm

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)$$

where $\eta$ is the **learning rate**.

### Intuition
1. Compute gradient (direction of steepest ascent)
2. Move in opposite direction (steepest descent)
3. Repeat until convergence

### Convergence Criteria
- Gradient norm: $\|\nabla f\| < \epsilon$
- Parameter change: $\|\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}_t\| < \epsilon$
- Loss change: $|f_{t+1} - f_t| < \epsilon$

### Learning Rate Selection

| Too Small | Too Large |
|-----------|-----------|
| Slow convergence | Oscillation, divergence |
| May get stuck | Overshoot minimum |

---

## 4. Stochastic Gradient Descent (SGD)

For large datasets, compute gradient on mini-batch:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f_{\mathcal{B}}(\boldsymbol{\theta}_t)$$

where $\mathcal{B}$ is a mini-batch of size $b$.

**Advantages:**
- Faster per iteration
- Noise helps escape local minima
- Better generalization

**Variance Reduction:**
True gradient ≈ Mini-batch gradient (in expectation)

---

## 5. Momentum

Add "velocity" to smooth updates:

$$\mathbf{v}_{t+1} = \beta \mathbf{v}_t - \eta \nabla f(\boldsymbol{\theta}_t)$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \mathbf{v}_{t+1}$$

**Why it helps:**
- Accelerates convergence in consistent directions
- Dampens oscillations
- Typical $\beta = 0.9$

### Nesterov Momentum

Look ahead before computing gradient:
$$\mathbf{v}_{t+1} = \beta \mathbf{v}_t - \eta \nabla f(\boldsymbol{\theta}_t + \beta \mathbf{v}_t)$$

---

## 6. Adaptive Learning Rates

### AdaGrad

Adapt learning rate per parameter:
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\mathbf{G}_t + \epsilon}} \odot \nabla f_t$$

where $\mathbf{G}_t = \sum_{i=1}^{t} (\nabla f_i)^2$

**Problem:** Learning rate monotonically decreases.

### RMSprop

Use exponential moving average:
$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1-\beta)(\nabla f_t)^2$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\mathbf{v}_t + \epsilon}} \nabla f_t$$

### Adam (Adaptive Moment Estimation)

Combines momentum and RMSprop:

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\nabla f_t \quad \text{(first moment)}$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)(\nabla f_t)^2 \quad \text{(second moment)}$$

Bias correction:
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}$$

Update:
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}\hat{\mathbf{m}}_t$$

**Default hyperparameters:** $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$

---

## 7. Learning Rate Schedules

### Step Decay
$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$$

### Exponential Decay
$$\eta_t = \eta_0 \cdot e^{-\lambda t}$$

### Cosine Annealing
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\frac{t\pi}{T}))$$

### Warmup
Start with small LR, gradually increase:
$$\eta_t = \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}}$$

---

## 8. Second-Order Methods

### Newton's Method

Use Hessian for better step:
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \mathbf{H}^{-1}\nabla f$$

**Advantages:** Faster convergence near optimum
**Disadvantages:** Computing/inverting Hessian is expensive

### Quasi-Newton (L-BFGS)

Approximate Hessian from gradient history.

---

## 9. Constrained Optimization

### Lagrange Multipliers

For $\min f(\mathbf{x})$ s.t. $g(\mathbf{x}) = 0$:

$$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda g(\mathbf{x})$$

Solve: $\nabla_\mathbf{x} \mathcal{L} = 0$ and $\nabla_\lambda \mathcal{L} = 0$

### KKT Conditions

For inequality constraints $g_i(\mathbf{x}) \leq 0$:
1. Stationarity: $\nabla f + \sum \lambda_i \nabla g_i = 0$
2. Primal feasibility: $g_i(\mathbf{x}) \leq 0$
3. Dual feasibility: $\lambda_i \geq 0$
4. Complementary slackness: $\lambda_i g_i(\mathbf{x}) = 0$

**Why it matters:** Foundation for SVM dual formulation.

---

## 10. Regularization as Optimization

### L2 Regularization (Ridge)
$$\min_{\boldsymbol{\theta}} f(\boldsymbol{\theta}) + \lambda\|\boldsymbol{\theta}\|_2^2$$

Gradient: $\nabla f + 2\lambda\boldsymbol{\theta}$

### L1 Regularization (Lasso)
$$\min_{\boldsymbol{\theta}} f(\boldsymbol{\theta}) + \lambda\|\boldsymbol{\theta}\|_1$$

Promotes sparsity (proximal gradient methods).

---

## 11. Key Applications

| Method | When to Use |
|--------|-------------|
| SGD + Momentum | Deep learning default |
| Adam | Fast convergence, less tuning |
| L-BFGS | Small datasets, convex problems |
| Newton's method | Second-order optimization |

---

## Python Implementation

```python
import numpy as np

class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
    
    def step(self, params, grads):
        return params - self.lr * grads

class SGDMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        self.velocity = self.momentum * self.velocity - self.lr * grads
        return params + self.velocity

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0
    
    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2
        
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

---

## Further Reading

- "Convex Optimization" by Boyd and Vandenberghe
- Sebastian Ruder: "An overview of gradient descent optimization algorithms"
- Stanford CS229 Optimization notes
