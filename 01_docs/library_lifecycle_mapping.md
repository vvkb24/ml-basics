# Python Libraries in the Data Science Lifecycle

A strategic mapping of when and why to use each library across the ML workflow.

---

## The Data Science Lifecycle

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                   │
│  Problem    Data       Data         Feature    Modeling          │
│  Framing → Collection → Understanding → Engineering → Training   │
│                                                                   │
│     ↓                                                             │
│                                                                   │
│  Evaluation → Deployment → Monitoring → Iteration                │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Library Mapping by Stage

### Stage 1: Data Collection & Loading

| Task | Primary Library | Why This Choice |
|------|-----------------|-----------------|
| Read CSV/Excel | **Pandas** | Flexible I/O, handles encoding, missing values |
| Read JSON | **Pandas** | `pd.read_json()` normalizes nested structures |
| Read Parquet | **Pandas** / **PyArrow** | Columnar format, efficient for large data |
| Read from SQL | **Pandas** + SQLAlchemy | `pd.read_sql()` abstracts DB differences |
| Read images | **PIL** / **torchvision** | Image-specific decompression, transforms |
| Read text | **HuggingFace datasets** | Handles common NLP datasets with tokenization |

**Decision Point**: 
- Tabular → Pandas
- Images → PIL/torchvision
- Text → HuggingFace
- Very large → Dask/Spark

---

### Stage 2: Data Understanding (EDA)

| Task | Primary Library | Why This Choice |
|------|-----------------|-----------------|
| Summary statistics | **Pandas** | `.describe()`, `.info()`, `.value_counts()` |
| Distributions | **Pandas** + **Matplotlib** | Histograms, KDE plots |
| Correlations | **Pandas** + **Seaborn** | `.corr()` + heatmaps |
| Missing analysis | **Pandas** | `.isna().sum()`, `msno.matrix()` |
| Groupwise analysis | **Pandas** | `.groupby()` with aggregations |
| Quick plots | **Seaborn** | High-level API, beautiful defaults |

**Why Pandas Dominates EDA**:
- Named columns make code readable
- GroupBy is essential for segment analysis
- Integration with plotting libraries

```python
# Typical EDA flow
df.info()                          # Types, memory, nulls
df.describe()                      # Numeric summaries
df['target'].value_counts()        # Class distribution
df.groupby('segment')['revenue'].mean()  # Segment analysis
```

---

### Stage 3: Feature Engineering

| Task | Primary Library | Why This Choice |
|------|-----------------|-----------------|
| Numeric scaling | **sklearn** | `StandardScaler`, `MinMaxScaler` |
| Categorical encoding | **sklearn** | `OneHotEncoder`, `LabelEncoder` |
| Text vectorization | **sklearn** | `TfidfVectorizer`, `CountVectorizer` |
| Date features | **Pandas** | `.dt` accessor (day, month, dayofweek) |
| Custom transforms | **sklearn** | `FunctionTransformer`, custom classes |
| Imputation | **sklearn** | `SimpleImputer`, `IterativeImputer` |

**Why sklearn for Feature Engineering**:
- `fit()` / `transform()` pattern prevents data leakage
- Pipelines bundle preprocessing with model
- Consistent API across all transformers

```python
# Proper feature engineering with sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# This preprocessor is fitted on training data only
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
```

---

### Stage 4: Model Training

| Task | Primary Library | Why This Choice |
|------|-----------------|-----------------|
| Linear models | **sklearn** | Fast, interpretable, regularization built-in |
| Tree ensembles | **sklearn** / **XGBoost** | sklearn for prototyping, XGBoost for production |
| Neural networks | **PyTorch** / **TensorFlow** | Custom architectures, GPU support |
| LLMs | **HuggingFace Transformers** | Pre-trained models, fine-tuning |
| Hyperparameter tuning | **sklearn** / **Optuna** | GridSearchCV for small, Optuna for complex |

**Decision Tree for Model Selection**:

```
Is it tabular data?
├── Yes → Try XGBoost/LightGBM first
│         └── If interpretability needed → Logistic Regression / Trees
├── No → Is it images?
│        ├── Yes → Use PyTorch + pretrained CNN
│        └── No → Is it text?
│                 ├── Yes → Use Transformers (BERT/GPT)
│                 └── No → Custom PyTorch model
```

---

### Stage 5: Evaluation

| Task | Primary Library | Why This Choice |
|------|-----------------|-----------------|
| Basic metrics | **sklearn.metrics** | accuracy, precision, recall, F1 |
| Confusion matrix | **sklearn** + **Seaborn** | `confusion_matrix()` + heatmap |
| ROC curves | **sklearn** | `roc_curve()`, `roc_auc_score()` |
| Statistical tests | **scipy.stats** | t-test, chi-squared, Wilcoxon |
| Cross-validation | **sklearn** | `cross_val_score()`, `KFold` |

**Why sklearn for Evaluation**:
- Consistent interface across all metrics
- Works with any array-like (numpy, torch, etc.)
- Classification report gives everything at once

```python
from sklearn.metrics import classification_report, roc_auc_score

print(classification_report(y_true, y_pred))
# Shows precision, recall, f1, support for each class

auc = roc_auc_score(y_true, y_prob)
# Works with probabilities, not predictions
```

---

### Stage 6: Deployment

| Task | Primary Library | Why This Choice |
|------|-----------------|-----------------|
| Model serialization | **joblib** / **pickle** | Fast save/load for sklearn models |
| PyTorch models | **torch.save()** / **ONNX** | State dict or cross-platform export |
| API serving | **FastAPI** / **Flask** | Python web frameworks |
| Containerization | **Docker** | Reproducible environments |
| ML platform | **MLflow** | Experiment tracking, model registry |

**Export Pipeline Example**:
```python
# sklearn: Save entire pipeline
import joblib
joblib.dump(trained_pipeline, 'model.joblib')

# PyTorch: Save state dict
torch.save(model.state_dict(), 'model.pth')

# Cross-platform: ONNX export
torch.onnx.export(model, dummy_input, 'model.onnx')
```

---

## Library Comparison by Role

### Numerical Computation

| Library | Strengths | Weaknesses |
|---------|-----------|------------|
| **NumPy** | Fast, foundational, everywhere | No labels, homogeneous only |
| **Pandas** | Labels, mixed types, I/O | Slower, memory-heavy |
| **PyTorch** | GPU, gradients | Overkill for simple math |

**Rule**: NumPy for math, Pandas for data wrangling, PyTorch for learning.

### Machine Learning

| Library | Best For | Avoid When |
|---------|----------|------------|
| **sklearn** | Classical ML, tabular | Deep learning |
| **XGBoost** | Best tabular performance | Interpretability needed |
| **PyTorch** | Custom neural nets | Simple problems |
| **Transformers** | NLP, pretrained models | Non-text tasks |

### Visualization

| Library | Best For | Style |
|---------|----------|-------|
| **Matplotlib** | Fine control, publication | Verbose |
| **Seaborn** | Statistical plots, quick EDA | High-level |
| **Plotly** | Interactive dashboards | Web-based |

---

## Common Anti-Patterns

### Anti-Pattern: Using Wrong Library for the Job

```python
# BAD: PyTorch for linear regression
model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
for epoch in range(1000):
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()

# GOOD: sklearn is simpler and faster
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)
```

### Anti-Pattern: Pandas for Everything

```python
# BAD: Pandas for heavy numerical work
result = df.apply(lambda row: expensive_calc(row), axis=1)  # Very slow!

# GOOD: Extract to NumPy for computation
X = df.values  # or df[cols].values
result = vectorized_calc(X)  # Much faster
```

### Anti-Pattern: Ignoring Library Versions

```python
# Code that worked in sklearn 0.24 breaks in 1.0
from sklearn.preprocessing import StandardScaler
# DeprecationWarning: feature_names_in_ behavior changed

# GOOD: Pin versions in requirements.txt
# scikit-learn==1.0.2
```

---

## Quick Reference: Which Library When?

| "I need to..." | Use |
|----------------|-----|
| Read a CSV | `pandas.read_csv()` |
| Scale features | `sklearn.StandardScaler` |
| Train a random forest | `sklearn.RandomForestClassifier` |
| Train a neural network | `torch.nn` |
| Fine-tune BERT | `transformers.AutoModel` |
| Plot a heatmap | `seaborn.heatmap()` |
| Calculate correlation | `pandas.DataFrame.corr()` |
| Do matrix math | `numpy` |
| Serve a model via API | `FastAPI` + `joblib` |
| Track experiments | `MLflow` |

---

## Summary

```
Data Lifecycle:
Load (pandas) → Explore (pandas+seaborn) → Transform (sklearn) 
→ Train (sklearn/pytorch) → Evaluate (sklearn) → Deploy (fastapi)
```

Each library exists to solve a specific problem. Master the right tool for each stage.
