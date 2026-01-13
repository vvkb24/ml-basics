# The Complete Data Science Lifecycle: A Rigorous Guide

A comprehensive, practitioner-focused treatment of the data science lifecycle from problem framing to production monitoring.

---

## Part I: Problem Framing

### 1.1 Business vs Scientific Questions

**Business questions** are outcome-oriented:
- "How can we reduce churn by 10%?"
- "Which customers should we target?"
- "What's the expected revenue impact?"

**Scientific questions** are mechanism-oriented:
- "What factors influence churn?"
- "Is there a causal relationship between X and Y?"
- "What is the distribution of customer lifetime value?"

**Critical Insight**: Many projects fail because business questions are translated incorrectly into ML problems.

**Example Failure**:
- Business: "Predict which customers will churn"
- ML team builds: Churn classifier
- Reality: By the time model predicts churn, it's too late to intervene
- Better framing: "Identify customers in early stages of disengagement"

### 1.2 Target Variable Definition

**Questions to ask**:
1. What exactly are you predicting?
2. When is the target observed? (temporal leakage risk!)
3. How is the target measured? (measurement error)
4. Is the target stable over time? (concept drift)

**Common Mistakes**:

| Mistake | Example | Consequence |
|---------|---------|-------------|
| Predicting proxy, not outcome | Predict clicks, want purchases | Optimizes wrong thing |
| Leaky target | Target contains future information | Unrealistic performance |
| Unstable definition | Fraud label changes over time | Model degrades |
| Aggregated target | Predict monthly total, need daily | Wrong granularity |

### 1.3 What NOT to Model

**Don't model if**:
1. **Simple rules suffice**: If business logic is 90% accurate, ML may not help
2. **No actionable intervention**: Prediction without action is useless
3. **Target is unknowable**: Some things cannot be predicted
4. **Data is fundamentally insufficient**: Garbage in, garbage out
5. **Cost of errors is asymmetric and unacceptable**: Medical, legal

**Red Flags**:
- "We want AI to solve this" (no clear objective)
- "We have lots of data" (but is it relevant?)
- "Our competitor does this" (context matters)

### 1.4 Common Framing Mistakes

| Mistake | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Predicting the past | Confusing correlation with prediction | Always use temporal holdout |
| Ignoring base rates | Rare events seem unpredictable | Check class balance first |
| Wrong unit of analysis | Predicting person-level from aggregate | Match prediction to decision |
| Survivorship bias | Only see successful cases | Consider missing data |

---

## Part II: Data Collection

### 2.1 Structured vs Unstructured Data

**Structured Data**:
- Tables with defined schema
- Fixed columns, typed values
- Examples: SQL databases, CSV files

**Unstructured Data**:
- No predefined schema
- Examples: text, images, audio, logs
- Requires preprocessing to extract features

**Semi-Structured Data**:
- Has some structure but flexible
- Examples: JSON, XML, logs with patterns

**Real-World Reality**:
- Most projects combine multiple types
- Joining structured and unstructured is hard
- Schema evolution breaks pipelines

### 2.2 Sampling Bias

**Types of Sampling Bias**:

| Type | Description | Example |
|------|-------------|---------|
| Selection bias | Non-random sample | Only surveying customers who respond |
| Survivorship bias | Only observing "survivors" | Analyzing only successful startups |
| Self-selection | Subjects choose to participate | Voluntary health surveys |
| Temporal bias | Time-dependent patterns | Training on 2019, deploying in 2020 (COVID) |
| Geographic bias | Location-specific patterns | US data for global model |

**Detection Methods**:
- Compare sample distributions to known population
- Check for missing subgroups
- Analyze collection timestamp patterns
- Examine data source metadata

### 2.3 Data Leakage Risks

**Leakage**: When training data contains information that wouldn't be available at prediction time.

**Types**:

1. **Target leakage**: Feature derived from target
   ```
   # BAD: "days_since_purchase" when predicting "will_purchase"
   # The feature contains the target information!
   ```

2. **Temporal leakage**: Using future information
   ```
   # BAD: Training on shuffled time series
   # Model sees future data during training
   ```

3. **Train-test contamination**: Preprocessing on full data
   ```python
   # BAD
   scaler.fit(all_data)  # Leaks test info
   scaler.transform(train), scaler.transform(test)
   
   # GOOD
   scaler.fit(train)  # Only fit on train
   scaler.transform(train), scaler.transform(test)
   ```

4. **Feature leakage**: Feature is proxy for target
   ```
   # Predicting hospital readmission
   # Feature: "discharge_summary" contains "plan for readmission"
   ```

### 2.4 Real-World Data Issues

**Missing Data**:
- MCAR (Missing Completely at Random): Safe to drop
- MAR (Missing at Random): Imputation possible
- MNAR (Missing Not at Random): Dangerous! Missingness is informative

**Delayed Data**:
- Labels arrive days/weeks after events
- Cannot train on most recent data
- Production-training mismatch

**Corrupted Data**:
- Sensor malfunctions (sudden zeros, spikes)
- Encoding errors (UTF-8 issues, date formats)
- Schema changes (column meanings change)
- Duplicate records

**Data Quality Checklist**:
```python
def data_quality_check(df):
    report = {}
    report['n_rows'] = len(df)
    report['n_cols'] = len(df.columns)
    report['missing_pct'] = df.isnull().mean() * 100
    report['duplicates'] = df.duplicated().sum()
    report['dtypes'] = df.dtypes.value_counts()
    
    # Check for suspicious values
    for col in df.select_dtypes(include='number'):
        report[f'{col}_zeros'] = (df[col] == 0).sum()
        report[f'{col}_negatives'] = (df[col] < 0).sum()
        report[f'{col}_outliers'] = ((df[col] - df[col].mean()).abs() > 3 * df[col].std()).sum()
    
    return report
```

---

## Part III: Data Understanding

### 3.1 Schema Validation

**Critical Questions**:
- Do column names match documentation?
- Are data types correct (numeric vs string)?
- Are there unexpected null patterns?
- Do value ranges make sense?

**Automated Validation**:
```python
import pandera as pa

schema = pa.DataFrameSchema({
    "age": pa.Column(int, checks=[
        pa.Check.greater_than(0),
        pa.Check.less_than(120)
    ]),
    "income": pa.Column(float, nullable=True),
    "category": pa.Column(str, checks=pa.Check.isin(["A", "B", "C"]))
})

# Validate
validated_df = schema.validate(df)
```

### 3.2 Data Types: The Hidden Complexity

| Semantic Type | Common Mistake | Correct Handling |
|---------------|----------------|------------------|
| ID | Treated as numeric | Should be string/categorical |
| Date | Kept as string | Parse to datetime |
| Categorical | Left as string | Encode appropriately |
| Ordinal | Treated as nominal | Preserve order |
| Currency | Mixed formats | Standardize to float |
| Phone/ZIP | Treated as numeric | Should be string |

### 3.3 Units and Scale Problems

**Unit Mismatches**:
- Temperature: Celsius vs Fahrenheit
- Currency: USD vs local currency
- Time: Seconds vs milliseconds
- Distance: Miles vs kilometers

**Scale Check**:
```python
def check_scale(df, col, expected_min, expected_max):
    actual_min, actual_max = df[col].min(), df[col].max()
    if actual_min < expected_min or actual_max > expected_max:
        print(f"WARNING: {col} outside [{expected_min}, {expected_max}]")
        print(f"Actual range: [{actual_min}, {actual_max}]")
        print("Possible unit mismatch!")
```

### 3.4 Silent Errors

Errors that don't raise exceptions but corrupt analysis:

| Silent Error | Example | Detection |
|--------------|---------|-----------|
| Wrong join | Many-to-many creates duplicates | Check row count after join |
| Timezone confusion | UTC vs local mixed | Verify timestamp ranges |
| Encoding issues | "Ã©" instead of "é" | Check for non-ASCII |
| Floating point | 0.1 + 0.2 ≠ 0.3 | Use appropriate precision |
| Integer overflow | Large IDs as int32 | Check max values |

---

## Part IV: Exploratory Data Analysis

### 4.1 Why EDA Is Not Optional

**EDA reveals**:
1. Data quality issues (before they break models)
2. Distribution shapes (guides preprocessing)
3. Relationships between variables (informs feature engineering)
4. Outliers and anomalies (need special handling)
5. Class imbalance (affects model choice)

**Without EDA**:
- You might train on corrupted data
- Model assumptions may be violated
- Feature engineering is blind guessing
- Evaluation metrics may be misleading

### 4.2 How EDA Changes Modeling Choices

| EDA Finding | Modeling Implication |
|-------------|----------------------|
| Skewed features | Log transform or robust models |
| Outliers | Robust scaling, winsorizing, or tree models |
| Multicollinearity | Regularization or feature selection |
| Nonlinear relationships | Polynomial features or nonlinear models |
| Class imbalance | Resampling, weighted loss, threshold tuning |
| Missing patterns | Imputation strategy or indicator features |

### 4.3 When EDA Misleads

**Dangers**:

1. **Overfitting to training data patterns**
   - Pattern in training may not generalize
   - Always validate on holdout

2. **Simpson's paradox**
   - Aggregate trend reverses within subgroups
   - Always stratify by key variables

3. **Confounders**
   - Correlation driven by third variable
   - Cannot conclude causation from EDA

4. **Visualization lies**
   - Truncated axes exaggerate differences
   - Cherry-picked time windows

5. **Multiple testing**
   - With enough comparisons, spurious patterns appear
   - Correct for multiple hypotheses

---

## Part V: Feature Engineering

### 5.1 Encoding Strategies

**Categorical Encoding**:

| Method | When to Use | Pitfall |
|--------|-------------|---------|
| One-hot | Low cardinality (<20) | Explodes with many categories |
| Label encoding | Ordinal data | Implies false order for nominal |
| Target encoding | High cardinality | Leakage if not done carefully |
| Frequency encoding | When frequency matters | Ties between categories |
| Hash encoding | Very high cardinality | Collisions |

**Target Encoding (with regularization)**:
```python
def target_encode(train, test, col, target, smoothing=10):
    global_mean = train[target].mean()
    stats = train.groupby(col)[target].agg(['mean', 'count'])
    
    # Smoothed mean
    stats['encoded'] = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
    
    train[f'{col}_encoded'] = train[col].map(stats['encoded'])
    test[f'{col}_encoded'] = test[col].map(stats['encoded']).fillna(global_mean)
    return train, test
```

### 5.2 Feature Leakage

**Types**:
1. **Target leakage**: Feature has target information
2. **Temporal leakage**: Using future data
3. **Preprocessing leakage**: Fitting on test data

**Detection**:
- Suspiciously high performance on validation
- Feature importance shows unexpected variable
- Performance drops dramatically in production

### 5.3 Interaction Terms

When $f(x_1, x_2) \neq f(x_1) + f(x_2)$.

**Example**: Click rate by device AND time of day:
- Mobile + morning: High
- Mobile + evening: Low
- Desktop + morning: Low
- Desktop + evening: High

Linear model without interaction cannot capture this!

```python
# Create interactions
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_interact = poly.fit_transform(X)
```

### 5.4 Domain-Driven Features

**Better than automated feature engineering**:

- Finance: Rolling volatility, Sharpe ratio
- Healthcare: Time since last visit, medication adherence
- E-commerce: Recency, frequency, monetary (RFM)
- NLP: Sentiment, readability, named entities

---

## Part VI: Modeling

### 6.1 Baseline Models

**Always start simple**:

| Problem | Baseline |
|---------|----------|
| Regression | Predict mean, linear regression |
| Classification | Predict mode, logistic regression |
| Time series | Predict last value, seasonal naive |
| Ranking | Random, popularity |

**Purpose of baseline**:
1. Sanity check (can we beat random?)
2. Upper bound on improvement
3. Simple model may be sufficient
4. Debugging complex models

### 6.2 Model Complexity Control

**Complexity vs Performance**:
```
Performance
    ↑
    |        ____
    |      /      \
    |     /        \
    |    /          \
    |___/____________\___→ Complexity
       Simple    Overfit
```

**Control Mechanisms**:
- Regularization (L1, L2)
- Early stopping
- Dropout
- Pruning (trees)
- Hyperparameter tuning

### 6.3 Assumption Checking

| Model | Key Assumptions | How to Check |
|-------|-----------------|--------------|
| Linear Regression | Linearity, normality, homoscedasticity | Residual plots |
| Logistic Regression | Linear log-odds | ROC, calibration |
| Tree models | None (but can overfit) | Learning curves |
| Neural networks | Smooth, differentiable | Loss curves |

---

## Part VII: Evaluation

### 7.1 Metric Selection

**Classification**:
| Metric | When to Use |
|--------|-------------|
| Accuracy | Balanced classes only |
| Precision | Cost of FP high (spam) |
| Recall | Cost of FN high (fraud, disease) |
| F1 | Balance precision/recall |
| AUC-ROC | Ranking quality |
| AUC-PR | Imbalanced data |
| Log-loss | Probability calibration matters |

**Regression**:
| Metric | When to Use |
|--------|-------------|
| MSE/RMSE | Penalize large errors |
| MAE | Robust to outliers |
| MAPE | Percentage errors matter |
| R² | Explain variance |

### 7.2 Why Accuracy Is Misleading

With 99% negative class:
- Predicting all negative → 99% accuracy!
- But completely useless for finding positives

**Better metrics for imbalance**:
- Precision-Recall curve
- F1 score
- Matthews Correlation Coefficient
- Cost-weighted accuracy

### 7.3 Cross-Validation Pitfalls

| Pitfall | Description | Solution |
|---------|-------------|----------|
| Data leakage | Preprocessing before split | Pipeline inside CV |
| Temporal leakage | Shuffled time series | Time-based splits |
| Group leakage | Same user in train/test | GroupKFold |
| Small data variance | High CV variance | Repeated CV |

```python
# CORRECT: Pipeline inside CV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

scores = cross_val_score(pipe, X, y, cv=5)  # Scaler fit inside each fold
```

---

## Part VIII: Deployment & Monitoring

### 8.1 Concept Drift

**Definition**: The relationship between features and target changes.

**Types**:
- Sudden: COVID-19 pandemic
- Gradual: User behavior evolution
- Recurring: Seasonality
- Incremental: Slow change over time

**Detection**:
```python
def detect_drift(train_dist, prod_dist, threshold=0.1):
    from scipy.stats import ks_2samp
    stat, pvalue = ks_2samp(train_dist, prod_dist)
    if pvalue < threshold:
        print(f"DRIFT DETECTED: KS stat = {stat}, p = {pvalue}")
    return stat, pvalue
```

### 8.2 Data Drift

**Definition**: Input feature distribution changes.

**Monitoring**:
- Track feature statistics (mean, std, percentiles)
- Track missing patterns
- Track categorical distribution

**Response**:
- Alert when drift exceeds threshold
- Retrain on recent data
- Investigate root cause

### 8.3 Feedback Loops

**Danger**: Model predictions influence future data.

**Example**: 
- Fraud model flags transactions as risky
- Flagged transactions get additional review
- Fewer false negatives for flagged group
- Model learns flagged = higher fraud (self-fulfilling)

**Mitigation**:
- Random holdout (no model intervention)
- Counterfactual evaluation
- Causal inference methods

### 8.4 Model Decay

**Reality**: All models decay without maintenance.

**Decay Causes**:
- Data drift
- Concept drift
- Feature pipeline changes
- Dependency updates
- Population changes

**Monitoring Dashboard**:
```
+------------------+------------------+
| Metric           | Current | Target |
+------------------+------------------+
| AUC-ROC          | 0.82    | 0.85   |
| Feature drift    | 0.05    | <0.10  |
| Prediction drift | 0.08    | <0.15  |
| Latency p99      | 45ms    | <50ms  |
| Error rate       | 0.1%    | <1%    |
+------------------+------------------+
```

---

## Real-World Case Study: Churn Prediction

### Stage 1: Problem Framing
- Business question: Reduce churn by 15%
- ML question: Identify at-risk customers for intervention
- Target: Customer doesn't purchase in next 30 days
- Actionable: Marketing team can send retention offers

### Stage 2: Data Collection
- Transaction history (3 years)
- Customer demographics
- Support tickets
- Website behavior

### Stage 3: Data Understanding
- 15% of customers churned
- Missing demographics for 30%
- Strange spike in transactions during promotion period

### Stage 4: EDA
- Churn correlated with days since last purchase
- High support ticket frequency → churn
- Certain product categories → lower churn

### Stage 5: Feature Engineering
- Recency, frequency, monetary (RFM) scores
- Rolling average of purchases
- Sentiment from support tickets

### Stage 6: Modeling
- Baseline: Logistic regression (AUC 0.72)
- Final: XGBoost (AUC 0.84)

### Stage 7: Evaluation
- Precision at 30% recall: 0.65
- Business value: $2M recovered revenue projected

### Stage 8: Deployment
- Weekly batch predictions
- Alert when feature drift >10%
- Retrain monthly

---

## References

- Géron - *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*
- Kuhn & Johnson - *Feature Engineering and Selection*
- Sculley et al. - *Hidden Technical Debt in ML Systems*
- Breck et al. - *Data Validation for Machine Learning*
