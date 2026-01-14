# Pandas: The Language of Tabular Data

A concept-first, failure-aware guide to Pandas for serious practitioners.

---

## 1. Why Pandas Exists: Conceptual and Historical Context

### The Problem Pandas Solves

NumPy arrays have no notion of:
- **Column names** (just integer indices)
- **Mixed types** (homogeneous arrays only)
- **Missing values** (NaN handling is limited)
- **Labeled rows** (time series, entity IDs)

**Real data looks like this:**

| user_id | signup_date | revenue | country |
|---------|-------------|---------|---------|
| A001 | 2023-01-15 | 150.50 | USA |
| A002 | 2023-02-20 | NaN | UK |

You can't represent this naturally in NumPy.

### What Pandas Provides

1. **DataFrame**: 2D labeled data structure (like a spreadsheet)
2. **Series**: 1D labeled array (a single column)
3. **Index**: Row labels (dates, IDs, hierarchical)
4. **Missing data**: First-class `NaN` and `NA` support
5. **I/O**: Read/write CSV, Excel, SQL, JSON, Parquet

### Historical Context

- **2008**: Wes McKinney creates pandas at AQR Capital (finance)
- **2009**: Open-sourced
- **2012**: Becomes standard for data science in Python
- **Today**: 100M+ downloads/month, foundation of data science stack

**Finance DNA**: Pandas was designed for time series and tabular financial data—this shapes its API and assumptions.

---

## 2. Mathematical Abstractions Pandas Encodes

### The DataFrame as a Mathematical Object

A DataFrame is a **labeled matrix with heterogeneous columns**:

$$\mathbf{D} = \{(\text{index}_i, \mathbf{row}_i)\}_{i=1}^{n}$$

Where each row is a **named tuple**:
$$\mathbf{row}_i = (x_{i,c_1}, x_{i,c_2}, \ldots, x_{i,c_k})$$

### Index Alignment

**The killer feature**: Operations automatically align on indices.

```python
a = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
b = pd.Series([10, 20, 30], index=['y', 'z', 'w'])

a + b
# x    NaN  (x not in b)
# y    12.0  (1 + 10, wait no...)
```

Actually:
```python
a + b
# w     NaN
# x     NaN
# y    22.0  (2 + 20, aligned by index!)
# z    33.0  (3 + 30)
```

**This is alignment by label, not position!**

### GroupBy as Split-Apply-Combine

Pandas `groupby` implements the mathematical pattern:

$$\text{result}_g = f(\{x_i : \text{group}(x_i) = g\})$$

```python
df.groupby('category')['value'].mean()
```

This is:
1. **Split**: Partition rows by 'category'
2. **Apply**: Compute mean for each partition
3. **Combine**: Assemble results into Series

### Memory Model

Each column is stored as a **contiguous NumPy array**:

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4.0, 5.0, 6.0]})
df['A'].values  # Underlying NumPy array (int64)
df['B'].values  # Underlying NumPy array (float64)
```

**Implication**: Column-wise operations are fast; row-wise operations are slow.

---

## 3. Assumptions Hidden in Common Functions

### `df.dropna()` Assumptions

```python
df.dropna()  # What does this assume?
```

**Hidden assumptions:**
1. **Drops entire rows** by default (might lose 90% of data!)
2. **Any NaN in row** triggers drop (use `how='all'` for stricter)
3. **Doesn't modify original** (returns copy unless `inplace=True`)

**Safer approach:**
```python
# Check how much data you're losing
print(f"Before: {len(df)}, After: {len(df.dropna())}")
print(f"NaN by column:\n{df.isna().sum()}")

# Drop only if specific columns have NaN
df.dropna(subset=['critical_column'])
```

### `df.merge()` vs `df.join()` 

| Method | Default behavior | When to use |
|--------|------------------|-------------|
| `merge()` | Inner join on columns | When joining by column values |
| `join()` | Left join on index | When joining by row indices |

**Hidden danger:**
```python
# If keys aren't unique, merge creates CARTESIAN PRODUCT
df1 = pd.DataFrame({'key': ['A', 'A'], 'val1': [1, 2]})
df2 = pd.DataFrame({'key': ['A', 'A'], 'val2': [10, 20]})

pd.merge(df1, df2, on='key')
# 4 rows! (2 * 2 = Cartesian product)
```

**Always check:**
```python
# Before merge, verify key uniqueness
assert df1['key'].is_unique, "df1 keys not unique!"
# Or use validate parameter:
pd.merge(df1, df2, on='key', validate='one_to_one')
```

### `df.apply()` is Usually Wrong

```python
# BAD: apply with axis=1 (row-wise) is incredibly slow
df['new'] = df.apply(lambda row: row['a'] + row['b'], axis=1)

# GOOD: Vectorized
df['new'] = df['a'] + df['b']
```

**When apply is OK:**
- Complex string operations not in `.str` accessor
- Custom aggregations in groupby
- When vectorization is impossible

### `df.copy()` Deep vs Shallow

```python
df2 = df       # NO COPY - same object!
df2 = df.copy()  # Shallow copy (data is duplicated)
df2 = df.copy(deep=True)  # Same as above

# Slicing creates a view (usually):
df2 = df[['col1', 'col2']]  # May or may not be a copy!
```

**The SettingWithCopyWarning**:
```python
df2 = df[df['a'] > 5]  # Creates a copy? Maybe!
df2['b'] = 10  # Warning: might not modify df

# Safe: Always use .loc
df.loc[df['a'] > 5, 'b'] = 10
```

---

## 4. What Pandas Does NOT Protect Against

### 4.1 Silent Type Conversion

```python
df = pd.DataFrame({'A': [1, 2, 3]})
df.loc[0, 'A'] = 'hello'  # Silently converts entire column to object!
print(df['A'].dtype)  # object (string)
```

**Worse:**
```python
df = pd.DataFrame({'A': [1, 2, 3]})
df.loc[0, 'A'] = np.nan  # Converts int to float!
print(df['A'].dtype)  # float64
```

**Use nullable dtypes to prevent:**
```python
df = pd.DataFrame({'A': pd.array([1, 2, 3], dtype='Int64')})
df.loc[0, 'A'] = pd.NA  # Stays Int64!
```

### 4.2 Index Alignment Surprises

```python
s1 = pd.Series([1, 2, 3], index=[0, 1, 2])
s2 = pd.Series([10, 20, 30], index=[1, 2, 3])

s1 + s2
# 0     NaN
# 1    12.0
# 2    23.0
# 3     NaN

# If you wanted positional addition:
s1.values + s2.values  # [11, 22, 33]
# Or reset index:
s1.reset_index(drop=True) + s2.reset_index(drop=True)
```

### 4.3 Chained Indexing Failure

```python
# This might silently fail:
df[df['a'] > 5]['b'] = 10  # SettingWithCopyWarning!

# This always works:
df.loc[df['a'] > 5, 'b'] = 10
```

### 4.4 Memory Explosion with Categoricals

```python
# String column with high cardinality
df = pd.DataFrame({'user_id': [f'user_{i}' for i in range(1_000_000)]})
print(df.memory_usage(deep=True).sum() / 1e6)  # ~64 MB

# Converting to category WHEN CARDINALITY IS HIGH doesn't help:
df['user_id'] = df['user_id'].astype('category')
print(df.memory_usage(deep=True).sum() / 1e6)  # ~64 MB (codes + categories)

# Category only helps when CARDINALITY IS LOW:
df = pd.DataFrame({'status': ['active', 'inactive'] * 500_000})
print(df.memory_usage(deep=True).sum() / 1e6)  # ~32 MB
df['status'] = df['status'].astype('category')
print(df.memory_usage(deep=True).sum() / 1e6)  # ~0.5 MB (1000x smaller!)
```

---

## 5. Failure Modes with Concrete Examples

### Failure Mode 1: Merge Explosion

**Scenario**: Joining transaction data with user data

```python
transactions = pd.DataFrame({
    'user_id': ['A', 'A', 'B', 'B', 'B'],
    'amount': [100, 200, 50, 75, 25]
})

# Bug: user_info has duplicate user_ids
user_info = pd.DataFrame({
    'user_id': ['A', 'A', 'B'],  # Duplicate A!
    'segment': ['premium', 'standard', 'basic']
})

merged = pd.merge(transactions, user_info, on='user_id')
print(len(merged))  # 8 rows! (explosion due to duplicates)
```

**Fix:**
```python
# Always validate before merge
assert user_info['user_id'].is_unique, "Duplicate user_ids found!"

# Or use validate parameter
pd.merge(transactions, user_info, on='user_id', validate='many_to_one')
```

### Failure Mode 2: Silent Aggregation Bug

```python
df = pd.DataFrame({
    'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02']),
    'revenue': [100, 200, 150]
})

# Bug: Forgot to aggregate, so groupby returns grouped object
daily = df.groupby('date')  # NOT a DataFrame!

# This silently does nothing:
daily['revenue']  # Returns a SeriesGroupBy, not data!

# Must aggregate:
daily = df.groupby('date')['revenue'].sum()  # Now it's a Series
```

### Failure Mode 3: DateTime Index Slicing

```python
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=10),
    'value': range(10)
}).set_index('date')

# Partial string indexing:
df.loc['2023-01']  # Returns rows for January 2023

# But this fails if index is not DatetimeIndex:
df = df.reset_index()
df.loc['2023-01']  # KeyError!
```

### Failure Mode 4: GroupBy + NaN

```python
df = pd.DataFrame({
    'group': ['A', 'A', 'B', None],
    'value': [1, 2, 3, 4]
})

# By default, NaN groups are DROPPED:
df.groupby('group')['value'].sum()
# A    3
# B    3
# (No row for None!)

# To include NaN:
df.groupby('group', dropna=False)['value'].sum()
# A      3
# B      3
# NaN    4
```

---

## 6. Performance and Scaling Trade-offs

### When Pandas is Fast

| Operation | Speed | Why |
|-----------|-------|-----|
| Column operations | ⚡ Very fast | NumPy vectorization |
| Filtering | ⚡ Fast | Boolean indexing on NumPy |
| GroupBy with simple aggs | ⚡ Fast | Optimized C code |
| Read/write Parquet | ⚡ Fast | Columnar format, compression |

### When Pandas is Slow

| Operation | Speed | Why | Alternative |
|-----------|-------|-----|-------------|
| Row iteration | 🐌 Very slow | Python loops | Vectorize |
| `apply(axis=1)` | 🐌 Slow | Python function calls | Vectorize or Numba |
| String operations | 🐌 Slow | Python strings | pyarrow StringDtype |
| Large CSVs | 🐌 Slow | Text parsing | Use Parquet |
| Wide DataFrames | 🐌 Slow | Many small arrays | Reduce columns |

### Memory Efficiency

```python
# Check memory:
df.info(memory_usage='deep')

# Reduce int size:
df['small_int'] = df['small_int'].astype('int8')  # -128 to 127

# Use categories for low-cardinality strings:
df['category'] = df['category'].astype('category')

# Use PyArrow backend (pandas 2.0+):
df = pd.read_csv('data.csv', dtype_backend='pyarrow')
```

### At What Scale Does Pandas Break?

| Data Size | Pandas Performance | Alternative |
|-----------|-------------------|-------------|
| < 1 GB | ✅ Excellent | Pandas |
| 1-10 GB | ⚠️ Possible with optimization | Polars, Dask |
| 10-100 GB | ❌ Memory issues | Dask, Spark, DuckDB |
| > 100 GB | ❌ Not feasible | Distributed systems |

---

## 7. When NOT to Use Pandas

### Use Polars Instead When:
- You need **faster performance** (Polars is 10-100x faster)
- You have **large datasets** (>1GB)
- You want **lazy evaluation** (query optimization)

### Use SQL/DuckDB Instead When:
- You're doing **complex aggregations**
- Data is already in a **database**
- You need **join optimizations**

### Use NumPy Instead When:
- Data is **homogeneous numerical**
- You need **maximum performance**
- No need for labels or mixed types

### Use Spark Instead When:
- Data is **distributed across machines**
- You need **fault tolerance**
- Processing **terabytes+** of data

---

## 8. Real-World Anti-Patterns

### Anti-Pattern 1: Iterating Rows

```python
# BAD: ~1000x slower than vectorized
for idx, row in df.iterrows():
    df.loc[idx, 'new'] = row['a'] * row['b']

# GOOD: Vectorized
df['new'] = df['a'] * df['b']
```

### Anti-Pattern 2: Growing DataFrames in a Loop

```python
# BAD: O(n²) due to reallocation
result = pd.DataFrame()
for item in items:
    row = process(item)
    result = pd.concat([result, pd.DataFrame([row])])

# GOOD: Collect in list, create once
rows = [process(item) for item in items]
result = pd.DataFrame(rows)
```

### Anti-Pattern 3: Ignoring dtypes on Load

```python
# BAD: Loads everything as default types, high memory
df = pd.read_csv('huge.csv')

# GOOD: Specify dtypes upfront
df = pd.read_csv('huge.csv', dtype={
    'user_id': 'category',
    'amount': 'float32',
    'count': 'int16'
})
```

### Anti-Pattern 4: Using `inplace=True`

```python
# BAD: Deprecated pattern, unclear behavior
df.drop(columns=['col'], inplace=True)
df.fillna(0, inplace=True)

# GOOD: Method chaining, clear data flow
df = (df
    .drop(columns=['col'])
    .fillna(0)
)
```

### Anti-Pattern 5: Not Checking for NaN Before Operations

```python
# Silent bug: mean ignores NaN but count doesn't!
mean_val = df['revenue'].mean()  # Ignores NaN
count = len(df)  # Includes NaN rows

# Wrong average calculation:
total = mean_val * count  # WRONG if NaN exists!

# Correct:
total = df['revenue'].sum()  # Handles NaN correctly
```

---

## Summary: Pandas Decision Framework

| Question | Answer | Action |
|----------|--------|--------|
| Mixed types (strings, numbers)? | Yes | Use Pandas |
| Need labeled axes? | Yes | Use Pandas |
| Homogeneous numerical? | Yes | Consider NumPy |
| > 1 GB data? | Yes | Consider Polars/Dask |
| Complex aggregations? | Yes | Consider SQL/DuckDB |
| Row-wise operations? | Yes | Vectorize or use apply wisely |

---

## References

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Effective Pandas](https://www.amazon.com/Effective-Pandas-Patterns-Manipulation-Treading/dp/B09MYXXSFM) - Matt Harrison
- [Modern Pandas](https://tomaugspurger.github.io/posts/modern-1-intro/) - Tom Augspurger
- [Pandas Anti-Patterns](https://pandas.pydata.org/docs/user_guide/enhancingperf.html)
