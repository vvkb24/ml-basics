# scikit-learn: The Grammar of Machine Learning

A concept-first, failure-aware guide to scikit-learn for serious practitioners.

---

## 1. Why scikit-learn Exists: Conceptual and Historical Context

### The Problem scikit-learn Solves

Before scikit-learn (2007):
- Each algorithm had **different APIs** 
- No standard for **train/test/predict**
- **Preprocessing** was ad-hoc
- **Model comparison** required manual work

scikit-learn provides:
1. **Unified API**: `fit()`, `predict()`, `transform()`
2. **Consistent estimator interface**: All models work the same way
3. **Pipelines**: Chain preprocessing and modeling
4. **Cross-validation**: Built-in model selection
5. **Extensive algorithms**: 90% of classical ML in one package

### The Estimator API Philosophy

Every scikit-learn object follows this pattern:

```python
# Estimators (models)
model = SomeModel(hyperparameters)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Transformers (preprocessors)
transformer = SomeTransformer()
transformer.fit(X_train)
X_transformed = transformer.transform(X_test)

# Combined
preprocessed = transformer.fit_transform(X_train)  # fit + transform
```

**Why this matters**: You can swap models without changing code structure.

### Historical Context

- **2007**: Started by David Cournapeau (Google Summer of Code)
- **2010**: INRIA team takes over (France)
- **2011**: First stable release
- **Today**: Most widely used ML library in the world

**Design DNA**: scikit-learn was designed for **tabular data** and **classical ML**. This shapes its assumptions.

---

## 2. Mathematical Abstractions scikit-learn Encodes

### The Supervised Learning Framework

scikit-learn encodes:
$$\hat{f}: \mathbb{R}^{n \times d} \to \mathbb{R}^n \text{ (regression) or } \{0,1\}^n \text{ (classification)}$$

**Core assumption**: Data is a matrix $X$ with $n$ samples and $d$ features.

### The Fit-Transform Pattern

**Fit**: Learn parameters from training data
$$\theta^* = \arg\min_\theta \mathcal{L}(X_{train}, y_{train}; \theta)$$

**Transform/Predict**: Apply learned parameters to new data
$$\hat{y} = f(X_{test}; \theta^*)$$

### Preprocessing as Linear Algebra

| Transformer | Mathematical Operation |
|-------------|----------------------|
| StandardScaler | $x' = \frac{x - \mu}{\sigma}$ |
| MinMaxScaler | $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$ |
| PCA | $X' = X W_k$ (projection onto top-k eigenvectors) |
| OneHotEncoder | $x \mapsto e_i$ (categorical to binary vector) |

### Loss Functions Encoded

| Model | Loss Function |
|-------|---------------|
| LinearRegression | $\sum_i (y_i - X_i\beta)^2$ (MSE) |
| Ridge | MSE + $\lambda\|\beta\|_2^2$ |
| Lasso | MSE + $\lambda\|\beta\|_1$ |
| LogisticRegression | Cross-entropy + regularization |
| SVM | Hinge loss + margin maximization |

---

## 3. Assumptions Hidden in Common Functions

### `train_test_split()` Assumptions

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Hidden assumptions:**
1. **i.i.d. data**: Samples are independent and identically distributed
2. **No temporal structure**: Random split is valid (NOT for time series!)
3. **Shuffling is OK**: Data order doesn't matter

**When this breaks:**
```python
# Time series - WRONG
train_test_split(time_series_df)  # Future leaks into training!

# Correct for time series:
train = df[df['date'] < '2023-01-01']
test = df[df['date'] >= '2023-01-01']
```

### `StandardScaler()` Assumptions

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Uses train statistics!
```

**Hidden assumptions:**
1. **Features are approximately Gaussian** (or at least symmetric)
2. **No extreme outliers** (they skew mean/std)
3. **Train and test have similar distributions**

**What happens with outliers:**
```python
X = np.array([[1], [2], [3], [1000]])  # Outlier
scaler.fit_transform(X)
# Most values squeezed near zero, outlier dominates
```

**Alternative**: `RobustScaler()` uses median/IQR instead.

### `cross_val_score()` Assumptions

```python
scores = cross_val_score(model, X, y, cv=5)
```

**Hidden assumptions:**
1. **No data leakage**: Preprocessing must be inside CV loop!
2. **Folds are representative**: Stratification may be needed
3. **Model is fast enough**: K fits required

**Data leakage trap:**
```python
# WRONG: Scaling leaks information
X_scaled = StandardScaler().fit_transform(X)  # Uses ALL data
scores = cross_val_score(model, X_scaled, y, cv=5)  # Optimistic!

# CORRECT: Use Pipeline
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
scores = cross_val_score(pipe, X, y, cv=5)  # Scaler fit inside each fold
```

### `GridSearchCV()` Assumptions

```python
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train, y_train)
```

**Hidden assumptions:**
1. **Validation set is representative of test set**
2. **Parameter space is well-defined**
3. **Best CV score ≈ best test score** (often optimistic)

**The double-dipping trap:**
```python
# Tune hyperparameters
grid.fit(X, y)  # Uses ALL data for CV

# Report best score
print(grid.best_score_)  # Optimistic! This is validation, not test

# Correct: Separate holdout set
grid.fit(X_train, y_train)
test_score = grid.score(X_test, y_test)  # Honest estimate
```

---

## 4. What scikit-learn Does NOT Protect Against

### 4.1 Data Leakage

scikit-learn **cannot detect** if you:
- Scaled before splitting
- Used target information in features
- Included future data in time series

**You must enforce this yourself.**

### 4.2 Class Imbalance

Default behavior ignores imbalance:
```python
# 95% class 0, 5% class 1
LogisticRegression().fit(X, y)  # Biased toward majority class

# Solutions:
LogisticRegression(class_weight='balanced')
# or use SMOTE from imbalanced-learn
```

### 4.3 Feature Scale Sensitivity

Some algorithms are scale-sensitive, others aren't:

| Scale-Sensitive | Scale-Invariant |
|-----------------|-----------------|
| SVM | Decision Trees |
| KNN | Random Forest |
| Neural Networks | Naive Bayes |
| Linear Regression (for regularized) | |

**scikit-learn doesn't warn you!**

### 4.4 Extrapolation Danger

```python
# Train on houses $100k-$500k
model.fit(X_train, prices)

# Predict on mansion worth $5M
model.predict(mansion_features)  # Garbage! Extrapolation
```

Models assume test data is within training distribution.

---

## 5. Failure Modes with Concrete Examples

### Failure Mode 1: The Preprocessing Leak

**Scenario**: You want to compare models fairly.

```python
# WRONG: Scaler sees all data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

model.fit(X_train, y_train)
model.score(X_test, y_test)  # Optimistic by 2-5%!
```

**Why it's wrong**: Test set statistics influenced the scaler.

**Correct:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only
X_test_scaled = scaler.transform(X_test)  # Apply to test

model.fit(X_train_scaled, y_train)
model.score(X_test_scaled, y_test)  # Honest
```

### Failure Mode 2: Accuracy on Imbalanced Data

**Scenario**: Fraud detection (1% fraud, 99% normal)

```python
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2%}")  # 99%!!!
```

**But**: The model just predicts "not fraud" for everything.

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, model.predict(X_test)))
# Precision for fraud: 0%
# Recall for fraud: 0%
```

**Solution**: Use appropriate metrics (F1, PR-AUC, or custom threshold).

### Failure Mode 3: Hyperparameter Overfitting

**Scenario**: Extensive grid search

```python
param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    # ... many more
}

grid = GridSearchCV(RandomForest(), param_grid, cv=5)
grid.fit(X_train, y_train)

print(grid.best_score_)  # 0.92
print(grid.score(X_test, y_test))  # 0.87 - overfit to validation!
```

**The problem**: Trying many combinations, you eventually find one that does well on validation by chance.

**Solution**: Use nested cross-validation or holdout test set.

### Failure Mode 4: Same Random State Everywhere

```python
# Seems reproducible, but...
X_train, X_test = train_test_split(X, y, random_state=42)
model = RandomForest(random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=5)  # No random_state in cv!
```

**Problem**: CV shuffling varies between runs.

**Solution**: Use `cv=KFold(n_splits=5, shuffle=True, random_state=42)`

---

## 6. Performance and Scaling Trade-offs

### When scikit-learn is Fast

| Situation | Speed | Why |
|-----------|-------|-----|
| < 100k samples | ⚡ Fast | Optimized NumPy/Cython |
| Sparse data | ⚡ Fast | Sparse matrix support |
| Linear models | ⚡ Very fast | Closed-form or fast optimization |
| Ensemble parallel | ⚡ Fast | `n_jobs=-1` uses all cores |

### When scikit-learn is Slow

| Situation | Speed | Alternative |
|-----------|-------|-------------|
| > 1M samples | 🐌 Slow | Use incremental learning or XGBoost |
| High-dimensional | 🐌 Slow | Dimensionality reduction first |
| Deep learning | 🐌 Impossible | Use PyTorch/TensorFlow |
| GPU training | ❌ No support | Use cuML or PyTorch |

### Memory Considerations

```python
# OneHotEncoder can explode memory
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)  

# If X has column with 100k categories:
# Dense: 100k columns × n_samples × 8 bytes = huge!
# Sparse: Only stores non-zeros (much smaller)
```

**Always use sparse for high-cardinality categoricals.**

---

## 7. When NOT to Use scikit-learn

### Use XGBoost/LightGBM Instead When:
- You need **best tabular performance**
- You have **large datasets** (faster training)
- You want **built-in handling of missing values**

### Use statsmodels Instead When:
- You need **statistical inference** (p-values, confidence intervals)
- You're doing **econometrics or causal inference**
- You need **detailed model diagnostics**

### Use PyTorch/TensorFlow Instead When:
- You need **deep learning**
- You want **GPU acceleration**
- You need **custom architectures**

### Use Spark MLlib Instead When:
- Data is **distributed across machines**
- You have **terabytes** of data

---

## 8. Real-World Anti-Patterns

### Anti-Pattern 1: Fitting on Full Data

```python
# BAD: Leaks information
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test = train_test_split(X_scaled)

# GOOD: Pipeline handles it
pipe = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])
pipe.fit(X_train, y_train)
```

### Anti-Pattern 2: Ignoring Feature Types

```python
# BAD: Scaling categorical features
X = pd.DataFrame({'age': [25, 30], 'gender': [0, 1]})  # gender is categorical!
StandardScaler().fit_transform(X)  # Scales gender meaninglessly

# GOOD: Use ColumnTransformer
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([
    ('num', StandardScaler(), ['age']),
    ('cat', OneHotEncoder(), ['gender'])
])
```

### Anti-Pattern 3: Not Using Pipelines

```python
# BAD: Manual steps, easy to mess up
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
# ... later in production, did you remember the scaler?

# GOOD: Pipeline bundles everything
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)
# pipe is one object: easy to save, deploy, reproduce
```

### Anti-Pattern 4: Tuning Without Baseline

```python
# BAD: Jump to complex tuning
grid = GridSearchCV(RandomForest(), huge_param_grid)
grid.fit(X_train, y_train)

# GOOD: Establish baselines first
baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)
print(f"Baseline: {baseline.score(X_test, y_test)}")

simple = LogisticRegression()
simple.fit(X_train, y_train)
print(f"Simple model: {simple.score(X_test, y_test)}")

# Only then try complex models
```

### Anti-Pattern 5: Reporting CV Score as Final Result

```python
# BAD: CV score is validation, not test
grid = GridSearchCV(model, params, cv=5)
grid.fit(X, y)  # Uses ALL data
print(f"Score: {grid.best_score_}")  # This is NOT generalizable

# GOOD: Proper evaluation
grid.fit(X_train, y_train)
print(f"CV Score: {grid.best_score_}")  # Validation
print(f"Test Score: {grid.score(X_test, y_test)}")  # Honest estimate
```

---

## Summary: scikit-learn Decision Framework

| Question | Answer | Action |
|----------|--------|--------|
| Tabular data? | Yes | sklearn is ideal |
| Need statistical inference? | Yes | Use statsmodels |
| Need deep learning? | Yes | Use PyTorch |
| > 1M samples? | Yes | Consider XGBoost or incremental learning |
| Need reproducibility? | Yes | Use Pipeline + random_state everywhere |
| Class imbalanced? | Yes | Use class_weight or imbalanced-learn |

---

## Essential Imports Cheatsheet

```python
# Data splitting
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline

# Models
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Baseline
from sklearn.dummy import DummyClassifier, DummyRegressor
```

---

## References

- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- ["Hands-On Machine Learning"](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) - Aurélien Géron
- [scikit-learn MOOC](https://inria.github.io/scikit-learn-mooc/) - INRIA
- ["Python Machine Learning"](https://www.packtpub.com/product/python-machine-learning) - Sebastian Raschka
