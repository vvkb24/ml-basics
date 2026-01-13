# Calculus for Machine Learning

Calculus is essential for understanding optimization in ML.

---

## 1. Derivatives

The derivative measures the rate of change:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

### Common Derivatives

| Function | Derivative |
|----------|------------|
| $c$ (constant) | $0$ |
| $x^n$ | $nx^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln(x)$ | $\frac{1}{x}$ |
| $\sin(x)$ | $\cos(x)$ |
| $\cos(x)$ | $-\sin(x)$ |

### Derivative Rules

**Sum Rule:**
$$(f + g)' = f' + g'$$

**Product Rule:**
$$(fg)' = f'g + fg'$$

**Chain Rule:**
$$(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$$

**Why it matters:** Chain rule is the foundation of backpropagation.

---

## 2. Partial Derivatives

For functions of multiple variables:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(\ldots, x_i + h, \ldots) - f(\ldots, x_i, \ldots)}{h}$$

**Example:** For $f(x, y) = x^2 + 3xy + y^2$:
- $\frac{\partial f}{\partial x} = 2x + 3y$
- $\frac{\partial f}{\partial y} = 3x + 2y$

---

## 3. Gradients

The gradient is the vector of all partial derivatives:

$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

**Key Properties:**
- Points in direction of steepest ascent
- Perpendicular to level curves
- Magnitude indicates steepness

**Example:** For $f(x, y) = x^2 + y^2$:
$$\nabla f = \begin{bmatrix} 2x \\ 2y \end{bmatrix}$$

---

## 4. Jacobian Matrix

For vector-valued function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$:

$$\mathbf{J} = \begin{bmatrix} 
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}$$

**Why it matters:** Used in neural network backpropagation.

---

## 5. Hessian Matrix

Matrix of second-order partial derivatives:

$$\mathbf{H} = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}$$

**Why it matters:**
- Positive definite Hessian → local minimum
- Negative definite Hessian → local maximum
- Used in second-order optimization (Newton's method)

---

## 6. Chain Rule in Depth

### Scalar Case
If $y = f(u)$ and $u = g(x)$:
$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

### Multivariate Case
If $z = f(x, y)$ where $x = g(t)$ and $y = h(t)$:
$$\frac{dz}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt}$$

### Backpropagation Example

For a two-layer network:
$$L = \text{loss}(y, \hat{y})$$
$$\hat{y} = \sigma(\mathbf{W}_2 \mathbf{h})$$
$$\mathbf{h} = \sigma(\mathbf{W}_1 \mathbf{x})$$

Gradients via chain rule:
$$\frac{\partial L}{\partial \mathbf{W}_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial \mathbf{h}} \cdot \frac{\partial \mathbf{h}}{\partial \mathbf{W}_1}$$

---

## 7. Important Derivatives for ML

### Sigmoid Function

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

### Softmax Function

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

$$\frac{\partial \text{softmax}_i}{\partial z_j} = \text{softmax}_i (\delta_{ij} - \text{softmax}_j)$$

### ReLU Function

$$\text{ReLU}(x) = \max(0, x)$$

$$\frac{d\text{ReLU}}{dx} = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}$$

### Log Loss

$$L = -[y \log(\hat{y}) + (1-y)\log(1-\hat{y})]$$

$$\frac{\partial L}{\partial \hat{y}} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$$

---

## 8. Taylor Series

Approximate function near a point:

$$f(x) \approx f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots$$

**Multivariate (first order):**
$$f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^T (\mathbf{x} - \mathbf{a})$$

**Why it matters:** 
- Justifies gradient descent (linear approximation)
- Newton's method uses quadratic approximation

---

## 9. Integrals

### Definite Integral
$$\int_a^b f(x) \, dx = F(b) - F(a)$$

where $F'(x) = f(x)$ (antiderivative).

### Key Applications in ML
- Computing expectations: $\mathbb{E}[X] = \int x \cdot p(x) \, dx$
- Normalization: $\int p(x) \, dx = 1$
- Marginalization: $p(x) = \int p(x, y) \, dy$

---

## 10. Key Applications in ML

| Concept | ML Application |
|---------|----------------|
| Gradient | Gradient descent optimization |
| Chain rule | Backpropagation |
| Hessian | Newton's method, curvature |
| Taylor series | Understanding optimization |
| Activation derivatives | Neural network training |
| Integrals | Probabilistic models |

---

## Python Examples

```python
import numpy as np

# Numerical gradient
def numerical_gradient(f, x, eps=1e-5):
    """Compute gradient numerically."""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# ReLU and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)
```

---

## Further Reading

- 3Blue1Brown: "Essence of Calculus"
- "Mathematics for Machine Learning" (Chapter 5)
- Stanford CS231n: Backpropagation notes
