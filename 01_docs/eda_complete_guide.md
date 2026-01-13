# Exploratory Data Analysis: Scientific Foundations and Practice

A rigorous treatment of EDA as a scientific process for understanding data before modeling.

---

## Part I: Foundations of EDA

### 1.1 What Is Exploratory Data Analysis?

EDA is the systematic investigation of data to:
1. **Discover patterns** and relationships
2. **Detect anomalies** and outliers
3. **Test assumptions** required for modeling
4. **Generate hypotheses** for further investigation
5. **Guide decisions** about preprocessing and modeling

**Key Distinction**:
- **Confirmatory Data Analysis (CDA)**: Test pre-specified hypotheses
- **Exploratory Data Analysis (EDA)**: Generate hypotheses from data

### 1.2 Why EDA Exists Scientifically

**Historical Context**: John Tukey (1977) formalized EDA as a discipline.

**Scientific Motivation**:
1. Real data violates textbook assumptions
2. Visualizations reveal what summaries hide
3. Understanding precedes prediction
4. Garbage in = garbage out

**The EDA Philosophy**:
> "The greatest value of a picture is when it forces us to notice what we never expected to see." — Tukey

### 1.3 EDA vs Statistical Inference

| Aspect | EDA | Inference |
|--------|-----|-----------|
| Goal | Discover patterns | Confirm hypotheses |
| Approach | Flexible, visual | Formal, mathematical |
| Output | Hypotheses, insights | p-values, confidence intervals |
| Multiple testing | Expected | Must correct |
| Assumptions | To be checked | Must be satisfied |

**Critical Insight**: EDA findings are **hypotheses**, not conclusions. They require validation on independent data.

### 1.4 What Questions EDA Can and Cannot Answer

**EDA Can Answer**:
- What is the shape of this distribution?
- Are there outliers or anomalies?
- Which variables are correlated?
- What are the typical values?
- Is there missing data?

**EDA Cannot Answer**:
- Is this effect statistically significant?
- Is this relationship causal?
- Will this pattern generalize to new data?
- What is the uncertainty in my estimate?

---

## Part II: Univariate Analysis

### 2.1 Histogram

**What It Represents Mathematically**:

Frequency in bin $[a, b)$:
$$f_{[a,b)} = \frac{\#\{x_i : a \leq x_i < b\}}{n}$$

**Implicit Assumptions**:
- Bin width is appropriate
- Data is continuous or has many unique values
- Sample size is sufficient for chosen bins

**When to Use**:
- First look at continuous variable
- Assessing distribution shape
- Checking for multimodality

**When NOT to Use**:
- Small samples (< 30): too few per bin
- Discrete data with few values: use bar chart
- When comparing distributions: use overlaid KDE

**How It Can Mislead**:

1. **Bin width sensitivity**:
```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
data = np.concatenate([np.random.normal(0, 1, 500), 
                       np.random.normal(3, 0.5, 200)])

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, bins in zip(axes, [5, 30, 100]):
    ax.hist(data, bins=bins, edgecolor='black')
    ax.set_title(f'{bins} bins')
# Same data, very different appearance!
```

2. **Edge effects**: Where bins start matters

**Best Practices**:
- Try multiple bin widths
- Use Sturges' rule as starting point: $k = 1 + \log_2(n)$
- Or Freedman-Diaconis: $h = 2 \cdot IQR \cdot n^{-1/3}$

---

### 2.2 Kernel Density Estimation (KDE)

**Mathematical Formulation**:

$$\hat{f}(x) = \frac{1}{nh}\sum_{i=1}^n K\left(\frac{x - x_i}{h}\right)$$

Where:
- $K$: Kernel function (typically Gaussian)
- $h$: Bandwidth (smoothness parameter)

**Implicit Assumptions**:
- Underlying distribution is continuous and smooth
- Bandwidth is appropriate
- Data is not heavily discrete

**When to Use**:
- Smooth estimate of distribution
- Comparing multiple distributions
- When histogram looks choppy

**When NOT to Use**:
- Discrete data (produces misleading smooth curves)
- At distribution boundaries (KDE bleeds beyond)
- Very small samples (unreliable)

**Bandwidth Sensitivity**:
```python
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt

data = np.random.exponential(2, 200)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, bw in zip(axes, [0.1, 0.5, 2.0]):
    kde = gaussian_kde(data, bw_method=bw)
    x = np.linspace(0, 15, 200)
    ax.plot(x, kde(x))
    ax.hist(data, bins=30, density=True, alpha=0.3)
    ax.set_title(f'bandwidth = {bw}')
```

**Edge Bias Problem**:
- Exponential data is ≥ 0
- KDE shows probability for negative values!
- Use boundary-corrected KDE for bounded data

---

### 2.3 Box Plot

**What It Shows**:
- Box: Q1 (25th percentile) to Q3 (75th percentile)
- Line in box: Median (Q2)
- Whiskers: Q1 - 1.5×IQR to Q3 + 1.5×IQR
- Points beyond whiskers: Potential outliers

Where IQR = Q3 - Q1 (Interquartile Range)

**Implicit Assumptions**:
- Unimodal distribution (bimodal hidden)
- Symmetric outlier definition
- Whisker rule is appropriate

**When to Use**:
- Comparing distributions across groups
- Quick outlier detection
- Compact summary of spread

**When NOT to Use**:
- Multimodal distributions (hidden in box)
- When actual shape matters
- Very small samples (quartiles unreliable)

**How It Misleads**:
```python
# Box plots hide multimodality!
import seaborn as sns

# Bimodal data
bimodal = np.concatenate([np.random.normal(-2, 0.5, 100),
                          np.random.normal(2, 0.5, 100)])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].boxplot(bimodal)
axes[0].set_title('Box plot: Looks unimodal!')
axes[1].hist(bimodal, bins=30)
axes[1].set_title('Histogram: Clearly bimodal!')
```

---

### 2.4 Violin Plot

**What It Adds**:
- Full distribution shape via KDE
- Quartiles or box plot overlay
- Best of both worlds

**When to Use**:
- Comparing distributions with different shapes
- When box plot hides important features
- Moderate to large sample sizes

**When NOT to Use**:
- Very small samples (KDE unreliable)
- Discrete data
- Many groups (becomes cluttered)

```python
import seaborn as sns
import pandas as pd

# Create data with different shapes
np.random.seed(42)
data = pd.DataFrame({
    'Group A': np.random.normal(0, 1, 200),
    'Group B': np.random.exponential(1, 200),
    'Group C': np.concatenate([np.random.normal(-1, 0.3, 100),
                               np.random.normal(1, 0.3, 100)])
}).melt(var_name='Group', value_name='Value')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
data.boxplot(column='Value', by='Group', ax=axes[0])
sns.violinplot(data=data, x='Group', y='Value', ax=axes[1])
```

---

### 2.5 Empirical CDF (ECDF)

**Mathematical Definition**:

$$\hat{F}(x) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(X_i \leq x)$$

Proportion of observations ≤ x.

**Advantages**:
- No binning decisions
- Shows all data points
- Easy to compare distributions
- Natural for percentile questions

**When to Use**:
- Comparing distributions formally
- Percentile analysis
- Checking distribution fit

**When NOT to Use**:
- When shape is more intuitive than quantiles
- Very small samples (steps dominate)

```python
from statsmodels.distributions.empirical_distribution import ECDF

ecdf = ECDF(data)
x = np.linspace(min(data), max(data), 100)
plt.plot(x, ecdf(x))
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.title('What fraction of data is ≤ x?')
```

---

## Part III: Bivariate Analysis

### 3.1 Scatter Plot

**What It Shows**:
- Joint distribution of two continuous variables
- Each point = one observation

**Implicit Assumptions**:
- Points visible (not too many overlapping)
- Both variables are continuous

**When to Use**:
- Exploring relationship between two variables
- Detecting nonlinearity
- Finding outliers in 2D

**When NOT to Use**:
- Millions of points (overplotting)
- Discrete variables (use jitter or heatmap)

**Overplotting Solutions**:
```python
# PROBLEM: 100,000 points = black blob
plt.scatter(x, y)  # Useless!

# SOLUTIONS:
# 1. Transparency
plt.scatter(x, y, alpha=0.1)

# 2. Hexbin
plt.hexbin(x, y, gridsize=30, cmap='YlOrRd')

# 3. 2D histogram
plt.hist2d(x, y, bins=50)

# 4. Sample
sample_idx = np.random.choice(len(x), 1000, replace=False)
plt.scatter(x[sample_idx], y[sample_idx])
```

---

### 3.2 Residual Plot

**What It Shows**:
- Residuals ($y - \hat{y}$) vs predicted values or features
- Should show no pattern if model is good

**What to Look For**:

| Pattern | Indicates |
|---------|-----------|
| Random scatter | Good fit |
| Funnel shape | Heteroscedasticity |
| Curve | Nonlinearity missed |
| Clusters | Subgroups not modeled |
| Trend | Systematic bias |

```python
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_train)
residuals = y_train - predictions

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Residuals vs predicted
axes[0].scatter(predictions, residuals, alpha=0.5)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Residual')
axes[0].set_title('Look for patterns!')

# Q-Q plot of residuals
from scipy import stats
stats.probplot(residuals, plot=axes[1])
axes[1].set_title('Should be straight line')
```

---

## Part IV: Model-Related Curves

### 4.1 ROC Curve

**Mathematical Definition**:

For threshold $t$:
- TPR(t) = True Positive Rate = TP/(TP+FN)
- FPR(t) = False Positive Rate = FP/(FP+TN)

ROC curve plots TPR vs FPR as t varies.

**Interpretation**:
- Diagonal = random guessing
- Top-left corner = perfect classifier
- AUC = probability that random positive ranks higher than random negative

**When to Use**:
- Evaluating ranking quality
- Comparing classifiers at all thresholds
- When positive/negative costs unknown

**When NOT to Use**:
- Heavily imbalanced data (use PR curve)
- When specific threshold matters
- When costs are known (use cost curve)

**Why ROC Can Mislead on Imbalanced Data**:
```python
# 99% negative class
# Model predicts all negative
# FPR = 0, TPR = 0 → seems "reasonable" on ROC
# But Precision = 0 (useless for finding positives!)
```

---

### 4.2 Precision-Recall Curve

**Mathematical Definition**:

For threshold $t$:
- Precision(t) = TP/(TP+FP)
- Recall(t) = TP/(TP+FN)

**When to Use**:
- Imbalanced classification
- When false positives are costly
- When you care about finding positives

**Key Difference from ROC**:
- ROC uses FPR (denominator = actual negatives)
- PR uses Precision (denominator = predicted positives)

With many negatives, FP/TN is small even with many false positives!

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
ap = average_precision_score(y_true, y_scores)

plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'PR Curve (AP = {ap:.3f})')

# Baseline is class prior (horizontal line)
plt.axhline(y=sum(y_true)/len(y_true), linestyle='--', label='Random')
```

---

### 4.3 Calibration Curve

**What It Shows**:
- Do predicted probabilities match actual frequencies?
- X-axis: Predicted probability
- Y-axis: Observed frequency

**Perfect Calibration**: Diagonal line

**How to Read**:
- Above diagonal: Model underconfident (predicts 0.3, actually 0.5)
- Below diagonal: Model overconfident (predicts 0.7, actually 0.4)

```python
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)

plt.plot(prob_pred, prob_true, 's-')
plt.plot([0, 1], [0, 1], '--', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration curve')
```

**Why Calibration Matters**:
- Probability 0.8 should mean 80% of the time positive
- Uncalibrated models can't be trusted for decisions
- Neural networks and boosting often need calibration

---

### 4.4 Learning Curves

**What It Shows**:
- Training & validation performance vs training set size
- Diagnoses underfitting/overfitting

**Interpretation**:

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Both low, converging | Underfitting | More complex model |
| Train high, val low, gap stays | Overfitting | More data or regularization |
| Both high, converging | Good fit | Collect to threshold |

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='accuracy'
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
plt.xlabel('Training set size')
plt.ylabel('Score')
plt.legend()
plt.title('Learning Curve')
```

---

## Part V: Decision Logic for Plot Selection

### 5.1 Decision Table: Variable Types

| Data Type | One Variable | Two Variables |
|-----------|--------------|---------------|
| Continuous | Histogram, KDE, ECDF | Scatter, Hexbin |
| Categorical (few) | Bar chart | Grouped bar, Heatmap |
| Categorical (many) | Top-N bar | Heatmap only |
| Continuous + Cat | Violin, Box by group | Faceted scatter |
| Time series | Line plot | Multiple lines |

### 5.2 Decision Table: Data Size

| Sample Size | Histogram | KDE | Scatter |
|-------------|-----------|-----|---------|
| < 30 | Wide bins | Avoid | Use all points |
| 30-1000 | Standard | Good | Points visible |
| 1000-100K | Fine bins | Good | Use alpha |
| > 100K | Density shading | Sample | Hexbin/heatmap |

### 5.3 Pattern → Problem Mapping

| Pattern Observed | Likely Problem | Investigation |
|------------------|----------------|---------------|
| Long right tail | Skewed distribution | Log transform? |
| Bimodal | Mixture of populations | Stratify analysis |
| Gaps/clumping | Data quality issue | Check source |
| Linear trend with increasing spread | Heteroscedasticity | Weighted regression |
| Strong correlation | Multicollinearity | VIF analysis |
| Sudden level shifts | Concept drift | Time-based analysis |

---

## Part VI: Edge Cases and Failure Modes

### 6.1 Simpson's Paradox

**Definition**: Aggregate trend reverses in subgroups.

```python
# Overall: More ads → fewer sales (negative correlation)
# But within each region: More ads → more sales (positive correlation)
# Explanation: Rich regions buy less AND get fewer ads
```

**Lesson**: Always stratify by potential confounders.

### 6.2 Aggregation Bias

Aggregating data can create or hide patterns:
- Individual behavior ≠ group average
- Different scales can flip conclusions

### 6.3 Outlier Distortion

**Mean vs Median**:
```python
incomes = [40000, 45000, 50000, 55000, 10000000]
np.mean(incomes)   # 2,038,000 (distorted)
np.median(incomes) # 50,000 (representative)
```

**Correlation**:
- One outlier can create or destroy correlation
- Always check scatter plot!

### 6.4 Visualization Lies

| Trick | Effect | Defense |
|-------|--------|---------|
| Truncated Y-axis | Exaggerates differences | Check axis starts at 0 |
| 3D charts | Distorts proportions | Use 2D |
| Cherry-picked time window | Hides context | Request full data |
| Colored scales | Biases perception | Check legend |
| Log scale without label | Hides magnitude | Check axis type |

---

## Part VII: EDA → Modeling Connection

### 7.1 How EDA Determines Model Choice

| EDA Finding | Modeling Implication |
|-------------|----------------------|
| Linear relationship | Linear models appropriate |
| Nonlinear pattern | Trees, polynomial, neural nets |
| Outliers | Robust methods or removal |
| Multicollinearity | Regularization, PCA |
| Clusters in features | Consider interaction terms |
| Heavy tails | Log transform or robust loss |
| Class imbalance | Weighted loss, resampling |

### 7.2 How Curves Reveal Model Problems

| Curve | What It Reveals |
|-------|-----------------|
| Residual plot | Nonlinearity, heteroscedasticity |
| Learning curve | Overfitting, underfitting |
| Loss curve | Training issues (divergence, plateaus) |
| ROC/PR curve | Discrimination quality |
| Calibration curve | Probability reliability |

### 7.3 When Models Fail Due to Ignored EDA

**Case 1**: Trained on aggregated data, deployed on individual level
- EDA would show very different individual patterns

**Case 2**: Feature had impossible values (negative age)
- EDA histogram would immediately show this

**Case 3**: Target variable definition changed over time
- EDA time series would show the shift

---

## References

- Tukey, J. (1977) - *Exploratory Data Analysis*
- Wickham, H. - *ggplot2* and Grammar of Graphics
- Cleveland, W. - *Visualizing Data*
- Wilke, C. - *Fundamentals of Data Visualization*
