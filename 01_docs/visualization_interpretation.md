# Visualization Interpretation: A Complete Reference

A rigorous guide to interpreting statistical and ML visualizations correctly, with emphasis on when they mislead.

---

## Part I: Time Series Visualizations

### 1.1 Trend Curves

**What It Shows**:
- Long-term direction of a time series
- Often extracted via moving average or decomposition

**Mathematical Formulation**:

Moving average of window $k$:
$$MA_t = \frac{1}{k}\sum_{i=0}^{k-1} y_{t-i}$$

**When to Use**:
- Understanding long-term patterns
- Removing short-term noise
- Identifying regime changes

**When NOT to Use**:
- When recent values matter most (MA lags)
- Very short series (not enough for trend)
- When seasonality is primary interest

**How It Misleads**:
1. **Lag at turning points**: MA doesn't detect turns until after they happen
2. **Edge effects**: First/last k values are incomplete
3. **Window choice**: Different windows show different "trends"

```python
import pandas as pd
import matplotlib.pyplot as plt

# Simulated data with trend and noise
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=365, freq='D')
trend = np.linspace(100, 200, 365)
noise = np.random.normal(0, 10, 365)
y = trend + noise

series = pd.Series(y, index=dates)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(series, alpha=0.5, label='Raw')
ax.plot(series.rolling(7).mean(), label='7-day MA')
ax.plot(series.rolling(30).mean(), label='30-day MA')
ax.legend()
ax.set_title('Same data, different windows → different apparent trends')
```

---

### 1.2 Seasonality Plots

**What It Shows**:
- Repeating patterns at fixed intervals
- Daily, weekly, monthly, yearly cycles

**Types**:
1. **Seasonal decomposition**: Separate trend, seasonal, residual
2. **Seasonal subseries**: Plot each season separately
3. **Polar/radial**: Circular plot for cyclic patterns

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose
result = seasonal_decompose(series, model='additive', period=7)

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
result.observed.plot(ax=axes[0], title='Observed')
result.trend.plot(ax=axes[1], title='Trend')
result.seasonal.plot(ax=axes[2], title='Seasonal')
result.resid.plot(ax=axes[3], title='Residual')
```

**Dangers**:
- Wrong period chosen → spurious seasonality
- Trend not removed → seasonal appears damped
- Additive vs multiplicative confusion

---

### 1.3 Autocorrelation Function (ACF)

**Mathematical Definition**:

$$ACF(k) = \frac{\sum_{t=k+1}^{n}(y_t - \bar{y})(y_{t-k} - \bar{y})}{\sum_{t=1}^{n}(y_t - \bar{y})^2}$$

Correlation of series with itself at lag $k$.

**What to Look For**:

| Pattern | Interpretation |
|---------|----------------|
| Slow decay | Non-stationary (trend) |
| Cutoff after lag q | MA(q) process |
| Oscillating decay | AR process |
| Spike at lag s | Seasonality of period s |

**Confidence Bands**:

Blue bands show $\pm 1.96/\sqrt{n}$ (95% CI for white noise).

Spikes outside bands are "significant" (but multiple testing caveat!).

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(series, ax=axes[0], lags=40)
axes[0].set_title('ACF: Look for patterns')

plot_pacf(series, ax=axes[1], lags=40)
axes[1].set_title('PACF: Direct effects at each lag')
```

---

### 1.4 Partial Autocorrelation Function (PACF)

**What It Shows**:
- Correlation at lag $k$ **after** removing effects of lags 1 through k-1
- Direct effect of lag k

**What to Look For**:

| Pattern | Interpretation |
|---------|----------------|
| Cutoff after lag p | AR(p) process |
| Slow decay | MA process |
| Spike at lag s | Seasonal AR component |

**ACF vs PACF Decision**:

| ACF | PACF | Model |
|-----|------|-------|
| Cutoff | Decay | MA |
| Decay | Cutoff | AR |
| Decay | Decay | ARMA |

---

## Part II: Model Diagnostic Curves

### 2.1 Loss Curves (Training)

**What It Shows**:
- Loss vs training iteration/epoch
- Should decrease and plateau

**What to Look For**:

| Pattern | Issue | Solution |
|---------|-------|----------|
| Not decreasing | Learning rate too low, bug | Increase LR, debug |
| Increasing | Learning rate too high | Decrease LR |
| Oscillating | LR too high or batch too small | Adjust hyperparams |
| Plateau early | Stuck in local min or underfitting | Better init, more capacity |
| Diverging to NaN | Numerical issues | Gradient clipping, lower LR |

**Train vs Validation Loss**:

| Pattern | Diagnosis |
|---------|-----------|
| Both decreasing together | Good fit |
| Train low, val high | Overfitting |
| Both high | Underfitting |
| Val increases while train decreases | Overfitting, stop early |

```python
history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val),
                    epochs=100)

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Look for gap = overfitting')
```

---

### 2.2 Feature Importance Plots

**Types**:
1. **Built-in importance** (trees): Impurity decrease
2. **Permutation importance**: Drop in score when shuffling
3. **SHAP values**: Game-theoretic attribution

**Dangers**:

| Method | Problem |
|--------|---------|
| Impurity-based | Biased toward high-cardinality features |
| Permutation | Doesn't handle correlated features well |
| Single-point | Hides nonlinear effects |

**Best Practice**: Use multiple methods, compare.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

rf = RandomForestClassifier().fit(X_train, y_train)

# Built-in
plt.barh(feature_names, rf.feature_importances_)
plt.title('Built-in (may be biased)')

# Permutation
perm_imp = permutation_importance(rf, X_val, y_val, n_repeats=10)
plt.barh(feature_names, perm_imp.importances_mean)
plt.title('Permutation (more reliable)')
```

---

## Part III: Distribution Analysis

### 3.1 Q-Q Plot (Quantile-Quantile)

**What It Shows**:
- Compare data quantiles to theoretical distribution
- If data follows distribution → straight line

**How to Read**:

| Pattern | Interpretation |
|---------|----------------|
| Straight line | Data matches distribution |
| S-curve | Heavy tails |
| Inverted S | Light tails |
| Curve up at right | Right skew |
| Curve down at right | Left skew |

```python
from scipy import stats

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Normal data
normal_data = np.random.normal(0, 1, 500)
stats.probplot(normal_data, plot=axes[0])
axes[0].set_title('Normal: Straight line')

# Heavy-tailed
t_data = np.random.standard_t(3, 500)
stats.probplot(t_data, plot=axes[1])
axes[1].set_title('Heavy-tailed: S-curve')

# Right-skewed
exp_data = np.random.exponential(1, 500)
stats.probplot(exp_data, plot=axes[2])
axes[2].set_title('Right-skewed: Curve up')
```

---

### 3.2 Pair Plots

**What It Shows**:
- All pairwise scatter plots
- Often with histograms on diagonal

**When to Use**:
- Initial exploration of relationships
- Finding clusters
- Detecting outliers

**When NOT to Use**:
- Many features (>10 becomes unreadable)
- Very large datasets (slow to render)

**Reading Tips**:
- Look for linear/nonlinear relationships
- Check for clusters (possible subgroups)
- Identify outliers visible in 2D

```python
import seaborn as sns

# Color by target for classification
sns.pairplot(df, hue='target', diag_kind='kde')
plt.suptitle('Color coding reveals class separation')
```

---

### 3.3 Heatmaps (Correlation)

**What It Shows**:
- Correlation matrix as colored grid
- Typically Pearson correlation

**Interpretation**:
- +1 (red): Perfect positive correlation
- 0 (white): No linear correlation
- -1 (blue): Perfect negative correlation

**Dangers**:
1. **Pearson only captures linear**: Nonlinear relationships score 0
2. **Outlier sensitivity**: One outlier can create/destroy correlation
3. **Correlation ≠ causation**: Always!
4. **Many features**: Multiple testing issue

```python
corr = df.corr()

# With annotation
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0,
            vmin=-1, vmax=1)
plt.title('Correlation Matrix')

# Clustermap for patterns
sns.clustermap(corr, cmap='RdBu_r', center=0)
```

---

## Part IV: Choosing the Right Visualization

### 4.1 Decision Framework

```
Is data continuous or categorical?
├── Continuous
│   ├── One variable?
│   │   ├── Small n (<30): Strip plot, rug plot
│   │   ├── Medium n: Histogram, KDE
│   │   └── Large n: Histogram, ECDF
│   ├── Two continuous?
│   │   ├── Small n: Scatter
│   │   ├── Large n: Hexbin, 2D histogram
│   │   └── Time-ordered: Line plot
│   └── Many continuous?
│       ├── <10: Pair plot
│       └── >10: Correlation heatmap, PCA
├── Categorical
│   ├── One variable: Bar chart
│   ├── Two categorical: Heatmap, mosaic
│   └── Cat + Continuous: Box, violin
└── Time series
    ├── Raw: Line plot
    ├── Trend: Moving average
    ├── Seasonality: Decomposition
    └── Stationarity: ACF, PACF
```

### 4.2 Data Properties → Plot Choice

| Property | Preferred Visualization |
|----------|------------------------|
| Skewed distribution | Histogram with many bins, KDE |
| Heavy tails | Log scale, robust stats |
| Multimodal | Violin, KDE overlay |
| Many outliers | Box plot, but show points |
| Time structure | Line, never scatter |
| Group comparison | Box, violin by group |
| Large dataset | Hexbin, density, sample |
| High-dimensional | PCA, t-SNE, UMAP |

---

## Part V: Common Mistakes and Solutions

### 5.1 Truncated Axes

**Problem**: Y-axis doesn't start at 0, exaggerating differences.

**Detection**: Check axis labels.

**When It's Okay**: Small relative changes matter (stock prices).

### 5.2 Cherry-Picked Time Windows

**Problem**: Show only the period that supports your narrative.

**Solution**: Always show full context, or explain why truncated.

### 5.3 3D Charts

**Problem**: Perspective distorts proportions.

**Solution**: Almost always use 2D.

### 5.4 Pie Charts

**Problems**:
- Humans bad at comparing angles
- Can't compare across charts

**Solution**: Bar charts almost always better.

### 5.5 Dual Y-Axes

**Problems**:
- Can imply false correlation
- Scale choice manipulates perception

**Solution**: Use two separate plots, or normalize to same scale.

---

## Part VI: Code Templates

### 6.1 Standard EDA Suite

```python
def eda_numeric(df, col):
    """Complete EDA for numeric column."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Distribution
    df[col].hist(ax=axes[0, 0], bins=30, edgecolor='black')
    axes[0, 0].set_title(f'{col}: Histogram')
    
    # Box plot
    df.boxplot(column=col, ax=axes[0, 1])
    axes[0, 1].set_title(f'{col}: Box Plot')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(df[col].dropna(), plot=axes[1, 0])
    axes[1, 0].set_title(f'{col}: Q-Q Plot')
    
    # Stats
    axes[1, 1].axis('off')
    stats_text = f"""
    Count: {df[col].count():,}
    Missing: {df[col].isna().sum():,} ({df[col].isna().mean()*100:.1f}%)
    Mean: {df[col].mean():.2f}
    Median: {df[col].median():.2f}
    Std: {df[col].std():.2f}
    Min: {df[col].min():.2f}
    Max: {df[col].max():.2f}
    Skew: {df[col].skew():.2f}
    Kurtosis: {df[col].kurtosis():.2f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace')
    
    plt.tight_layout()
    return fig

def eda_categorical(df, col, max_categories=20):
    """Complete EDA for categorical column."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Value counts
    counts = df[col].value_counts()
    if len(counts) > max_categories:
        counts = counts.head(max_categories)
        axes[0].set_title(f'{col}: Top {max_categories} Categories')
    else:
        axes[0].set_title(f'{col}: All Categories')
    
    counts.plot(kind='barh', ax=axes[0])
    
    # Missing and cardinality stats
    axes[1].axis('off')
    stats_text = f"""
    Total: {len(df):,}
    Missing: {df[col].isna().sum():,} ({df[col].isna().mean()*100:.1f}%)
    Unique: {df[col].nunique():,}
    Top: {df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A'}
    Top Freq: {df[col].value_counts().iloc[0]:,}
    """
    axes[1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace')
    
    plt.tight_layout()
    return fig
```

### 6.2 Model Diagnostics Suite

```python
def model_diagnostics(y_true, y_pred, y_proba=None):
    """Complete model diagnostics."""
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.metrics import roc_curve, precision_recall_curve
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues')
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    if y_proba is not None:
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        axes[0, 1].plot(fpr, tpr)
        axes[0, 1].plot([0, 1], [0, 1], 'k--')
        axes[0, 1].set_xlabel('FPR')
        axes[0, 1].set_ylabel('TPR')
        axes[0, 1].set_title('ROC Curve')
        
        # PR curve
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        axes[1, 0].plot(rec, prec)
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
        
        # Calibration
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        axes[1, 1].plot(prob_pred, prob_true, 's-')
        axes[1, 1].plot([0, 1], [0, 1], 'k--')
        axes[1, 1].set_xlabel('Mean Predicted')
        axes[1, 1].set_ylabel('Fraction Positive')
        axes[1, 1].set_title('Calibration Curve')
    
    plt.tight_layout()
    return fig
```

---

## References

- Tufte, E. - *The Visual Display of Quantitative Information*
- Few, S. - *Show Me the Numbers*
- Wilke, C. - *Fundamentals of Data Visualization*
- Cairo, A. - *How Charts Lie*
