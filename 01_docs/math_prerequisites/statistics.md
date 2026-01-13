# Statistics for Machine Learning

Statistics provides methods for data analysis, inference, and model evaluation.

---

## 1. Descriptive Statistics

### Measures of Central Tendency

**Mean (Average):**
$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

**Median:** Middle value when sorted (robust to outliers)

**Mode:** Most frequent value

### Measures of Spread

**Variance:**
$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2$$

*(Note: n-1 for unbiased sample variance)*

**Standard Deviation:** $s = \sqrt{s^2}$

**Interquartile Range (IQR):** $Q_3 - Q_1$ (robust to outliers)

---

## 2. Estimation Theory

### Point Estimators

An estimator $\hat{\theta}$ estimates the true parameter $\theta$.

**Bias:**
$$\text{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta$$

**Variance:**
$$\text{Var}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \mathbb{E}[\hat{\theta}])^2]$$

**Mean Squared Error:**
$$\text{MSE}(\hat{\theta}) = \text{Bias}^2 + \text{Variance}$$

### Properties of Good Estimators
- **Unbiased:** $\mathbb{E}[\hat{\theta}] = \theta$
- **Consistent:** $\hat{\theta} \to \theta$ as $n \to \infty$
- **Efficient:** Minimum variance among unbiased estimators

---

## 3. Confidence Intervals

A confidence interval provides a range likely containing the true parameter.

For mean with known variance:
$$\bar{x} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$$

For mean with unknown variance (t-distribution):
$$\bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}$$

**Interpretation:** A 95% CI means: if we repeated the experiment many times, 95% of the intervals would contain the true parameter.

---

## 4. Hypothesis Testing

### Framework

1. **Null hypothesis** $H_0$: Status quo (typically no effect)
2. **Alternative hypothesis** $H_1$: What we want to test
3. **Test statistic**: Computed from data
4. **p-value**: Probability of seeing data at least as extreme under $H_0$
5. **Decision**: Reject $H_0$ if p-value < $\alpha$ (significance level)

### Types of Errors

| | $H_0$ True | $H_0$ False |
|---|---|---|
| **Reject $H_0$** | Type I (α) | Correct |
| **Fail to reject** | Correct | Type II (β) |

**Power:** $1 - \beta$ = probability of correctly rejecting false $H_0$

### Common Tests

**t-test (comparing means):**
$$t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}$$

**Chi-squared test (categorical data):**
$$\chi^2 = \sum \frac{(O - E)^2}{E}$$

---

## 5. Correlation

### Pearson Correlation

Measures linear relationship:
$$r = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}}$$

- $r = 1$: Perfect positive correlation
- $r = -1$: Perfect negative correlation
- $r = 0$: No linear correlation

### Spearman Correlation

Measures monotonic relationship (uses ranks).

**Caution:** Correlation ≠ Causation

---

## 6. Central Limit Theorem

If $X_1, \ldots, X_n$ are i.i.d. with mean $\mu$ and variance $\sigma^2$:

$$\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1) \text{ as } n \to \infty$$

**Why it matters:** 
- Justifies using normal distribution for sample means
- Foundation for many statistical tests

---

## 7. Bootstrapping

Non-parametric method for estimating sampling distributions.

**Algorithm:**
1. Draw B samples with replacement from data
2. Compute statistic for each resample
3. Use distribution of resampled statistics

```python
import numpy as np

def bootstrap_ci(data, statistic, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval."""
    n = len(data)
    stats = []
    for _ in range(n_bootstrap):
        resample = np.random.choice(data, size=n, replace=True)
        stats.append(statistic(resample))
    
    alpha = 1 - ci
    lower = np.percentile(stats, 100 * alpha/2)
    upper = np.percentile(stats, 100 * (1 - alpha/2))
    return lower, upper
```

---

## 8. Statistical Learning Concepts

### Training, Validation, Test Split

- **Training set:** Model fitting
- **Validation set:** Hyperparameter tuning
- **Test set:** Final evaluation (never used for model selection)

### Cross-Validation

**K-Fold Cross-Validation:**
1. Split data into K folds
2. Train on K-1 folds, validate on remaining fold
3. Repeat K times, average results

Reduces variance of performance estimate.

### Sampling Bias

- **Selection bias:** Non-representative sample
- **Survivorship bias:** Only observing "survivors"
- **Confounding:** Hidden variable affects outcome

---

## 9. Key Applications in ML

| Concept | ML Application |
|---------|----------------|
| Mean/Variance | Feature normalization |
| Confidence intervals | Model uncertainty |
| Hypothesis testing | A/B testing |
| Correlation | Feature selection |
| CLT | Batch normalization justification |
| Bootstrap | Bagging, Random Forest |
| Cross-validation | Model selection |

---

## Python Examples

```python
import numpy as np
from scipy import stats

# Descriptive statistics
data = np.random.normal(100, 15, 1000)
mean = np.mean(data)
std = np.std(data, ddof=1)  # Sample std

# Confidence interval
ci = stats.t.interval(
    0.95, 
    df=len(data)-1, 
    loc=mean, 
    scale=stats.sem(data)
)

# t-test
sample1 = np.random.normal(100, 15, 50)
sample2 = np.random.normal(105, 15, 50)
t_stat, p_value = stats.ttest_ind(sample1, sample2)

# Correlation
r, p = stats.pearsonr(sample1[:50], sample2[:50])
```

---

## Further Reading

- "All of Statistics" by Larry Wasserman
- "Statistical Inference" by Casella and Berger
- "The Elements of Statistical Learning" (Chapters 7-8)
