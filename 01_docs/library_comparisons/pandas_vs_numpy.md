# Pandas vs NumPy: When to Use Which

A practical comparison for data practitioners.

---

## Core Philosophy

| Aspect | NumPy | Pandas |
|--------|-------|--------|
| **Purpose** | Numerical computation | Data wrangling |
| **Primary abstraction** | Homogeneous array | Labeled DataFrame |
| **Designed for** | Mathematical operations | Business/research data |
| **Performance** | Fastest | Slower but more convenient |

---

## When Pandas Wins

### 1. Heterogeneous Data

```python
# NumPy forces you to use object dtype (slow!)
data = np.array([["Alice", 25, 50000], ["Bob", 30, 60000]], dtype=object)

# Pandas handles naturally
df = pd.DataFrame({
    'name': ["Alice", "Bob"],
    'age': [25, 30],
    'salary': [50000, 60000]
})
```

### 2. Missing Data

```python
# NumPy: NaN only for floats, breaks int arrays
arr = np.array([1, 2, np.nan, 4])  # Converts to float64!

# Pandas: Nullable types
s = pd.Series([1, 2, pd.NA, 4], dtype='Int64')  # Stays integer
```

### 3. Labeled Axes

```python
# NumPy: Manual index tracking
revenue_2023 = data[0, 2]  # What does [0, 2] mean?

# Pandas: Self-documenting
revenue_2023 = df.loc['Alice', 'salary']  # Clear intent
```

### 4. GroupBy Operations

```python
# NumPy: Manual implementation required
unique_groups = np.unique(groups)
means = [data[groups == g].mean() for g in unique_groups]

# Pandas: One line
means = df.groupby('category')['value'].mean()
```

---

## When NumPy Wins

### 1. Pure Numerical Computation

```python
# Pandas overhead for math:
%timeit df['col'].values ** 2 + df['col'].values  # Fast

# But if already NumPy:
%timeit arr ** 2 + arr  # Faster
```

### 2. N-Dimensional Arrays (N > 2)

```python
# Pandas: Only 1D (Series) and 2D (DataFrame)

# NumPy: Any dimensions
tensor = np.random.rand(10, 20, 30, 40)  # 4D array
```

### 3. Custom Algorithms

```python
# Implementing custom numerical method (e.g., gradient descent):
# NumPy is cleaner:
theta = np.random.randn(n_features)
for _ in range(iterations):
    gradient = X.T @ (X @ theta - y) / m
    theta -= learning_rate * gradient
```

### 4. Memory Efficiency

```python
# NumPy: 8 bytes per float64
arr = np.zeros(1_000_000, dtype=np.float64)  # 8 MB

# Pandas: Extra overhead (index, metadata)
s = pd.Series(np.zeros(1_000_000))  # ~10 MB
```

---

## Hybrid Approach: Best of Both Worlds

```python
# Start with Pandas for data loading and cleaning
df = pd.read_csv('data.csv')
df = df.dropna().query('age > 18')

# Convert to NumPy for computation
X = df[['feature1', 'feature2']].values
y = df['target'].values

# NumPy for model training
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)

#Back to Pandas for results
df['predictions'] = model.predict(X)
```

---

## Performance Comparison

| Operation | NumPy | Pandas | Winner |
|-----------|-------|--------|--------|
| Element-wise math | 1x | 1.2x | NumPy |
| Filtering rows | 1x | 1.5x | NumPy |
| GroupBy aggregation | N/A (manual) | Built-in | Pandas |
| String operations | N/A | Built-in | Pandas |
| Reshaping | 1x | 2x | NumPy |

---

## Decision Framework

```
Is your data heterogeneous (mixed types)?
├── Yes → Pandas
└── No → Continue

Do you need row/column labels?
├── Yes → Pandas
└── No → Continue

Is it more than 2D?
├── Yes → NumPy
└── No → Continue

Is it pure math (matrix ops, FFT, etc.)?
├── Yes → NumPy
└── No → Pandas (for convenience)
```

---

## Common Pitfalls

### 1. Unnecessary DataFrame Creation

```python
# BAD: Creating DataFrame for simple array
df = pd.DataFrame({'col': [1, 2, 3]})
result = df['col'].sum()

# GOOD: Use NumPy directly
arr = np.array([1, 2, 3])
result = arr.sum()
```

### 2. Ignoring .values

```python
# SLOW: Pandas Series in tight loop
for val in df['column']:  # Iterator overhead
    compute(val)

# FAST: Extract to NumPy first
for val in df['column'].values:
    compute(val)
```

### 3. Wrong Tool for joins

```python
# NumPy: Terrible for joins
result = []
for i in range(len(arr1)):
    for j in range(len(arr2)):
        if arr1[i, 0] == arr2[j, 0]:
            result.append(np.hstack([arr1[i], arr2[j]]))

# Pandas: Built for this
result = pd.merge(df1, df2, on='key')
```

---

## Summary

| Use Case | Library |
|----------|---------|
| CSV with mixed types | Pandas |
| Images (H × W × C) | NumPy |
| Linear algebra | NumPy |
| Time series resampling | Pandas |
| Neural network tensors | NumPy (or PyTorch) |
| Business analytics | Pandas |
| Signal processing | NumPy |

**Golden Rule**: Start with Pandas for data prep, extract to NumPy for computation.
