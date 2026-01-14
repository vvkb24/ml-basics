# NumPy: The Foundation of Scientific Python

A concept-first, failure-aware guide to NumPy for serious practitioners.

---

## 1. Why NumPy Exists: Conceptual and Historical Context

### The Problem NumPy Solves

Python lists are:
- **Flexible** (heterogeneous types)
- **Dynamic** (growable)
- **Slow** (interpreted, boxed objects)

For numerical computing, this is catastrophic:
```python
# Pure Python: ~100x slower than NumPy
result = [a[i] * b[i] for i in range(len(a))]  # 1M elements: ~500ms

# NumPy: vectorized, compiled
result = a * b  # 1M elements: ~5ms
```

### Why This Matters

**CPU operations have hierarchical latency:**

| Operation | Latency |
|-----------|---------|
| L1 cache access | ~1 ns |
| L2 cache access | ~4 ns |
| RAM access | ~100 ns |
| Python interpreter overhead | ~1000 ns per operation |

NumPy eliminates Python interpreter overhead by:
1. Storing data in **contiguous memory blocks**
2. Executing operations in **compiled C code**
3. Using **SIMD instructions** (process 4-8 numbers simultaneously)

### Historical Context

- **1995**: Numeric Python (Jim Hugunin)
- **2001**: Numarray (better for large arrays)
- **2006**: NumPy unifies both (Travis Oliphant)
- **Today**: Foundation for entire scientific Python ecosystem

**Consequence**: Libraries like Pandas, sklearn, PyTorch all use NumPy's array protocols internally.

---

## 2. Mathematical Abstractions NumPy Encodes

### The N-Dimensional Array (ndarray)

NumPy's core abstraction is the **homogeneous, fixed-size, contiguous array**:

$$\mathbf{A} \in \mathbb{R}^{d_1 \times d_2 \times \ldots \times d_n}$$

This encodes several mathematical assumptions:

| Mathematical Concept | NumPy Implementation |
|---------------------|----------------------|
| Vector space | 1D array with dtype |
| Matrix | 2D array |
| Tensor | N-dimensional array |
| Linear transformation | Matrix multiplication (`@` or `np.dot`) |
| Inner product | `np.dot(a, b)` for 1D |
| Norm | `np.linalg.norm(x, ord=2)` |

### Memory Layout Mathematics

NumPy arrays store a **pointer + strides + shape + dtype**:

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.strides)  # (24, 8) - bytes to jump for each dimension
print(a.shape)    # (2, 3)
print(a.dtype)    # int64 = 8 bytes
```

**The stride formula for element access:**
$$\text{address}(i, j) = \text{base} + i \cdot \text{stride}_0 + j \cdot \text{stride}_1$$

This means:
- **Row-major (C order)**: Last index changes fastest
- **Column-major (Fortran order)**: First index changes fastest

**ML Implication**: Wrong memory layout causes cache misses and 10-100x slowdowns.

### Vectorization as Mathematical Mapping

NumPy's element-wise operations encode:
$$f: \mathbb{R}^n \to \mathbb{R}^n, \quad f(\mathbf{x})_i = g(x_i)$$

This is **NOT** the same as:
$$f: \mathbb{R}^n \to \mathbb{R}, \quad f(\mathbf{x}) = \sum_i g(x_i)$$

The first is embarrassingly parallel; the second requires reduction.

---

## 3. Assumptions Hidden in Common Functions

### `np.mean()` Assumptions

```python
np.mean(x)  # What does this assume?
```

**Hidden assumptions:**
1. **Numeric dtype**: Silently converts or fails on strings
2. **No NaN handling**: `np.mean([1, 2, np.nan])` → `nan` (not 1.5!)
3. **Overflow risk**: Large arrays of int32 can overflow during summation
4. **Statistical meaning**: Mean assumes symmetric distribution is representative

**Safe alternative:**
```python
np.nanmean(x)  # Ignores NaN
np.mean(x, dtype=np.float64)  # Prevents overflow
```

### `np.dot()` vs `@` vs `np.matmul()`

| Function | Behavior for 1D | Behavior for 2D | Behavior for ND |
|----------|-----------------|-----------------|-----------------|
| `np.dot(a, b)` | Inner product | Matrix multiplication | Complex rules |
| `a @ b` | Inner product | Matrix multiplication | Batch matmul |
| `np.matmul(a, b)` | Inner product | Matrix multiplication | Batch matmul |

**Hidden danger:**
```python
a = np.random.rand(3, 4, 5)
b = np.random.rand(3, 5, 2)

# np.dot broadcasts differently than @
np.dot(a, b).shape   # ERROR or unexpected shape
(a @ b).shape        # (3, 4, 2) - batch matrix multiply
```

**Rule**: Use `@` for matrix operations. Use `np.dot` only for explicit inner products.

### `np.sum()` Axis Confusion

```python
a = np.array([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)

np.sum(a, axis=0)  # [5, 7, 9]  - sum OVER axis 0 (rows collapse)
np.sum(a, axis=1)  # [6, 15]   - sum OVER axis 1 (columns collapse)
```

**The confusion**: "axis=0" means "collapse rows", not "sum rows".

**Mental model**: The specified axis **disappears** from the result:
- Input: (2, 3), axis=0 → Output: (3,)
- Input: (2, 3), axis=1 → Output: (2,)

### Broadcasting Rules (The Silent Killer)

Broadcasting enables this:
```python
a = np.array([[1], [2], [3]])  # (3, 1)
b = np.array([10, 20, 30])     # (3,)

a + b  # Shape: (3, 3) - creates 3x3 matrix!
```

**Broadcasting rules:**
1. Align shapes from the **right**
2. Dimensions of size 1 are stretched
3. Missing dimensions are prepended as size 1

**Silent failure example:**
```python
# Intended: subtract mean from each column
X = np.random.rand(100, 5)
mean = X.mean(axis=0)  # Shape: (5,)
X_centered = X - mean   # Correct! (100, 5) - (5,) broadcasts

# Bug: wrong axis
mean_wrong = X.mean(axis=1)  # Shape: (100,)
X_wrong = X - mean_wrong     # ERROR: shapes (100, 5) and (100,)
# But if mean_wrong was (100, 1):
X_wrong = X - mean_wrong.reshape(-1, 1)  # Broadcasts to (100, 5)
# This SILENTLY subtracts row means, not column means!
```

---

## 4. What NumPy Does NOT Protect Against

### 4.1 Numerical Instability

**Catastrophic cancellation:**
```python
a = 1e16
b = 1e16 + 1
c = a - b  # Expected: 1, Actual: 0.0 (precision loss)
```

**Overflow/Underflow:**
```python
np.exp(1000)   # inf
np.exp(-1000)  # 0.0 (underflow)

# In softmax, this is deadly:
logits = np.array([1000, 1001, 1002])
np.exp(logits) / np.sum(np.exp(logits))  # [nan, nan, nan]

# Safe version:
logits_shifted = logits - np.max(logits)
np.exp(logits_shifted) / np.sum(np.exp(logits_shifted))  # Works!
```

### 4.2 View vs Copy Confusion

```python
a = np.array([1, 2, 3, 4, 5])

# This creates a VIEW (shared memory)
b = a[1:4]
b[0] = 999
print(a)  # [1, 999, 3, 4, 5] - a is modified!

# This creates a COPY
c = a[[1, 2, 3]]  # Fancy indexing
c[0] = 888
print(a)  # [1, 999, 3, 4, 5] - a is NOT modified
```

**Rule of thumb:**
- Slicing (`a[1:4]`) → View
- Fancy indexing (`a[[1,2,3]]`) → Copy
- Boolean mask (`a[a > 2]`) → Copy

### 4.3 Silent Type Coercion

```python
a = np.array([1, 2, 3])  # int64
a[0] = 1.7               # Silently truncates to 1!
print(a)  # [1, 2, 3]

# No warning! No error! Just wrong.
```

### 4.4 Dimension Traps

```python
a = np.array([1, 2, 3])      # Shape: (3,) - 1D array
b = np.array([[1, 2, 3]])    # Shape: (1, 3) - 2D row vector
c = np.array([[1], [2], [3]]) # Shape: (3, 1) - 2D column vector

# These are NOT equivalent for matrix operations!
a @ a      # 14 (scalar - inner product)
b @ b.T    # [[14]] (1x1 matrix)
c @ c.T    # [[1, 2, 3], [2, 4, 6], [3, 6, 9]] (3x3 outer product!)
```

---

## 5. Failure Modes with Concrete Examples

### Failure Mode 1: The Wrong Reduction

**Scenario**: Computing per-class accuracy

```python
# Ground truth and predictions for 3 classes
y_true = np.array([0, 0, 1, 1, 2, 2])
y_pred = np.array([0, 1, 1, 1, 2, 0])

# Wrong: global accuracy
accuracy = (y_true == y_pred).mean()  # 0.667

# Problem: Class 2 has 50% accuracy, masked by others
# Correct: per-class accuracy
for c in [0, 1, 2]:
    mask = y_true == c
    acc = (y_pred[mask] == c).mean()
    print(f"Class {c}: {acc:.2f}")
# Class 0: 0.50, Class 1: 1.00, Class 2: 0.50
```

### Failure Mode 2: Memory Explosion

```python
# Creating all pairwise distances (n=50,000 points, d=100)
n = 50000
X = np.random.rand(n, 100)

# This creates (50000, 50000) matrix = 20GB!
diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # Memory error!

# Solution: chunked computation
from scipy.spatial.distance import cdist
distances = cdist(X[:1000], X[:1000])  # Process in chunks
```

### Failure Mode 3: Random Seed Trap

```python
np.random.seed(42)
a = np.random.rand(5)
b = np.random.rand(5)

np.random.seed(42)
c = np.random.rand(10)

# a != c[:5] if internal state differs!
# Use np.random.default_rng() instead:

rng = np.random.default_rng(42)
a = rng.random(5)
b = rng.random(5)

rng2 = np.random.default_rng(42)
c = rng2.random(5)  # c == a guaranteed
```

### Failure Mode 4: Integer Division Surprise

```python
# Python 3 division
3 / 2  # 1.5 (float division)

# NumPy with integer arrays
a = np.array([3])
b = np.array([2])
a / b  # array([1.5]) - OK in NumPy!

# But floor division:
a // b  # array([1]) - integer result
```

---

## 6. Performance and Scaling Trade-offs

### When NumPy is Fast

| Operation | Speed | Why |
|-----------|-------|-----|
| Element-wise math | ⚡ Very fast | SIMD vectorization |
| Matrix multiply | ⚡ Very fast | BLAS libraries (OpenBLAS, MKL) |
| Reductions (sum, mean) | ⚡ Fast | Compiled loops |
| Broadcasting | ⚡ Fast | Zero-copy views |

### When NumPy is Slow

| Operation | Speed | Why | Alternative |
|-----------|-------|-----|-------------|
| Python loops | 🐌 Very slow | Interpreter overhead | Vectorize or Numba |
| Small arrays | 🐌 Slow | Function call overhead | Python lists |
| Non-contiguous access | 🐌 Slow | Cache misses | Ensure C-contiguous |
| Growing arrays | 🐌 Slow | Reallocation | Preallocate |

### Memory Efficiency

```python
# 1 billion float64 = 8 GB
huge = np.zeros(10**9, dtype=np.float64)

# Same data, 4x smaller:
small = np.zeros(10**9, dtype=np.float16)  # 2 GB

# Trade-off: float16 has ~3 decimal digits of precision
np.float16(1.0001)  # 1.0 (precision loss!)
```

**Recommendation**:
- Use float32 for ML training (good balance)
- Use float64 for numerical algorithms (stability)
- Use float16 only for inference with proper scaling

---

## 7. When NOT to Use NumPy

### Use Pandas Instead When:
- Data has **heterogeneous types** (strings, dates, numbers)
- You need **labeled axes** (column names, row indices)
- Missing data handling is critical (`NaN` semantics)

### Use PyTorch/TensorFlow Instead When:
- You need **GPU acceleration**
- You need **automatic differentiation**
- You're doing deep learning

### Use Scipy Instead When:
- You need **sparse matrices** (most entries are zero)
- You need **specialized algorithms** (integration, optimization)
- You need **statistical distributions**

### Use Numba Instead When:
- You MUST use loops (unavoidable algorithm)
- NumPy's vectorization doesn't fit your problem
- You need JIT compilation for custom functions

```python
from numba import jit

@jit(nopython=True)
def custom_loop(x):
    result = 0.0
    for i in range(len(x)):
        result += x[i] ** 2  # This loop is now fast!
    return result
```

---

## 8. Real-World Anti-Patterns

### Anti-Pattern 1: Loop Instead of Vectorize

```python
# BAD: 100x slower
result = []
for i in range(len(x)):
    result.append(x[i] ** 2 + y[i])
result = np.array(result)

# GOOD: Vectorized
result = x ** 2 + y
```

### Anti-Pattern 2: Repeated Concatenation

```python
# BAD: O(n²) due to reallocation
results = np.array([])
for item in data:
    results = np.concatenate([results, [process(item)]])

# GOOD: Preallocate
results = np.empty(len(data))
for i, item in enumerate(data):
    results[i] = process(item)

# BEST: Vectorize or list comprehension
results = np.array([process(item) for item in data])
```

### Anti-Pattern 3: Ignoring Memory Order

```python
# When calling BLAS/LAPACK (e.g., matrix multiply):
A = np.random.rand(1000, 1000, order='C')  # C-contiguous
B = np.random.rand(1000, 1000, order='F')  # Fortran-contiguous

# This may copy B internally, slowing down:
C = A @ B

# Ensure consistent order:
B = np.ascontiguousarray(B)  # Convert to C-order
C = A @ B  # Faster
```

### Anti-Pattern 4: Misusing `np.array()` in Loops

```python
# BAD: Creates new array object each iteration
for i in range(1000000):
    temp = np.array([x[i], y[i]])  # Slow!
    result[i] = np.linalg.norm(temp)

# GOOD: Vectorize entirely
result = np.sqrt(x**2 + y**2)
```

### Anti-Pattern 5: Silent Broadcasting Bugs

```python
# Intended: Normalize each row
X = np.random.rand(100, 5)
norms = np.linalg.norm(X, axis=1)  # Shape: (100,)
X_normalized = X / norms  # BUG! Broadcasts wrongly

# Correct:
X_normalized = X / norms[:, np.newaxis]  # Shape: (100, 1) broadcasts correctly
```

---

## Summary: NumPy Decision Framework

| Question | Answer | Action |
|----------|--------|--------|
| Can I vectorize? | Yes | Use NumPy |
| Is data heterogeneous? | Yes | Use Pandas |
| Need GPU? | Yes | Use PyTorch |
| Need loops unavoidably? | Yes | Use Numba |
| Is array small (<100 elements)? | Yes | Consider Python lists |
| Need sparse data? | Yes | Use scipy.sparse |

---

## References

- [NumPy Documentation](https://numpy.org/doc/stable/)
- [From Python to NumPy](https://www.labri.fr/perso/nrougier/from-python-to-numpy/) - Nicolas Rougier
- [100 NumPy Exercises](https://github.com/rougier/numpy-100)
- [NumPy Illustrated](https://betterprogramming.pub/numpy-illustrated-the-visual-guide-to-numpy-3b1d4976de1d)
