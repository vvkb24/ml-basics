# Linear Algebra for Machine Learning

Linear algebra is the foundation of machine learning. This document covers essential concepts.

## Notation

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x}$ | Column vector (bold lowercase) |
| $\mathbf{A}$ | Matrix (bold uppercase) |
| $\mathbf{A}^T$ | Transpose of A |
| $\mathbf{A}^{-1}$ | Inverse of A |
| $\|\mathbf{x}\|$ | Norm of vector x |

---

## 1. Vectors

A vector is an ordered list of numbers:

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n$$

### Vector Operations

**Addition:**
$$\mathbf{x} + \mathbf{y} = \begin{bmatrix} x_1 + y_1 \\ x_2 + y_2 \\ \vdots \\ x_n + y_n \end{bmatrix}$$

**Scalar Multiplication:**
$$c\mathbf{x} = \begin{bmatrix} cx_1 \\ cx_2 \\ \vdots \\ cx_n \end{bmatrix}$$

**Dot Product:**
$$\mathbf{x} \cdot \mathbf{y} = \mathbf{x}^T\mathbf{y} = \sum_{i=1}^{n} x_i y_i$$

### Norms

The **L2 norm** (Euclidean norm):
$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2} = \sqrt{\mathbf{x}^T\mathbf{x}}$$

The **L1 norm** (Manhattan norm):
$$\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i|$$

**Why it matters:** Regularization uses L1 (Lasso) and L2 (Ridge) norms.

---

## 2. Matrices

A matrix is a 2D array of numbers:

$$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

### Matrix Multiplication

For $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{B} \in \mathbb{R}^{n \times p}$:

$$(\mathbf{AB})_{ij} = \sum_{k=1}^{n} a_{ik}b_{kj}$$

The result is $\mathbf{C} \in \mathbb{R}^{m \times p}$.

**Key Properties:**
- Not commutative: $\mathbf{AB} \neq \mathbf{BA}$ in general
- Associative: $(\mathbf{AB})\mathbf{C} = \mathbf{A}(\mathbf{BC})$
- Distributive: $\mathbf{A}(\mathbf{B} + \mathbf{C}) = \mathbf{AB} + \mathbf{AC}$

---

## 3. Special Matrices

### Identity Matrix
$$\mathbf{I} = \begin{bmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{bmatrix}$$

Property: $\mathbf{AI} = \mathbf{IA} = \mathbf{A}$

### Diagonal Matrix
Only diagonal elements are non-zero.

### Symmetric Matrix
$\mathbf{A} = \mathbf{A}^T$

**Why it matters:** Covariance matrices are symmetric.

### Positive Definite Matrix
For all $\mathbf{x} \neq \mathbf{0}$: $\mathbf{x}^T\mathbf{A}\mathbf{x} > 0$

**Why it matters:** Ensures convexity in optimization.

---

## 4. Matrix Inverse

For square matrix $\mathbf{A}$, the inverse $\mathbf{A}^{-1}$ satisfies:

$$\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$$

**Properties:**
- $(\mathbf{AB})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$
- $(\mathbf{A}^T)^{-1} = (\mathbf{A}^{-1})^T$

**Why it matters:** Used in the normal equation for linear regression:
$$\boldsymbol{\theta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

---

## 5. Determinant

The determinant measures the "volume scaling" of a linear transformation.

For 2×2 matrix:
$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

**Properties:**
- $\det(\mathbf{AB}) = \det(\mathbf{A})\det(\mathbf{B})$
- $\det(\mathbf{A}^T) = \det(\mathbf{A})$
- $\det(\mathbf{A}^{-1}) = \frac{1}{\det(\mathbf{A})}$

**Why it matters:** 
- If $\det(\mathbf{A}) = 0$, matrix is singular (not invertible)
- Used in Gaussian probability density

---

## 6. Eigenvalues and Eigenvectors

An eigenvector $\mathbf{v}$ of matrix $\mathbf{A}$ satisfies:

$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

where $\lambda$ is the corresponding eigenvalue.

### Computing Eigenvalues

Solve the characteristic equation:
$$\det(\mathbf{A} - \lambda\mathbf{I}) = 0$$

### Eigendecomposition

For square matrix with $n$ linearly independent eigenvectors:

$$\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$$

where:
- $\mathbf{V}$: Matrix of eigenvectors (columns)
- $\boldsymbol{\Lambda}$: Diagonal matrix of eigenvalues

**Why it matters:**
- PCA uses eigendecomposition of covariance matrix
- Powers: $\mathbf{A}^n = \mathbf{V}\boldsymbol{\Lambda}^n\mathbf{V}^{-1}$

---

## 7. Singular Value Decomposition (SVD)

Any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ can be decomposed:

$$\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$$

where:
- $\mathbf{U} \in \mathbb{R}^{m \times m}$: Left singular vectors (orthogonal)
- $\boldsymbol{\Sigma} \in \mathbb{R}^{m \times n}$: Singular values (diagonal)
- $\mathbf{V} \in \mathbb{R}^{n \times n}$: Right singular vectors (orthogonal)

**Why it matters:**
- Works for any matrix (not just square)
- Used in dimensionality reduction, matrix factorization
- Low-rank approximation

---

## 8. Key Applications in ML

| Concept | ML Application |
|---------|----------------|
| Matrix multiplication | Neural network forward pass |
| Inverse | Normal equation |
| Transpose | Computing gradients |
| Eigendecomposition | PCA |
| SVD | Latent semantic analysis, compression |
| Positive definiteness | Kernel methods, covariance |
| Norms | Regularization |

---

## NumPy Examples

```python
import numpy as np

# Vector operations
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
dot_product = np.dot(x, y)  # 32
l2_norm = np.linalg.norm(x)  # 3.74

# Matrix operations
A = np.array([[1, 2], [3, 4]])
A_inv = np.linalg.inv(A)
det = np.linalg.det(A)  # -2.0

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(A)

# SVD
U, S, Vt = np.linalg.svd(A)
```

---

## Further Reading

- [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- Gilbert Strang: "Linear Algebra and Its Applications"
- Stephen Boyd: "Introduction to Applied Linear Algebra"
