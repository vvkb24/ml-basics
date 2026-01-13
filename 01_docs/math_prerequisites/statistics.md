# Statistics for Machine Learning: Complete Guide

Statistics provides the tools for data analysis, inference, and model evaluation—essential for understanding and validating ML systems.

---

## Why Statistics Matters in ML

| Statistical Concept | ML Application |
|--------------------|----------------|
| Descriptive stats | Data exploration, feature engineering |
| Estimation theory | Model training, parameter learning |
| Hypothesis testing | A/B testing, model comparison |
| Confidence intervals | Uncertainty quantification |
| Sampling | Train/test splits, bootstrapping |

---

## 1. Descriptive Statistics

### Measures of Central Tendency

![Central Tendency](https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/Visualisation_mode_median_mean.svg/450px-Visualisation_mode_median_mean.svg.png)
*Comparison of mean, median, and mode — Source: [Wikipedia](https://en.wikipedia.org/wiki/Central_tendency)*

**Mean (Arithmetic Average):**
$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

- Sensitive to outliers
- Used for: Feature centering, normalization

**Median:** Middle value when sorted
- Robust to outliers
- Used for: Skewed data, robust statistics

**Mode:** Most frequent value
- Used for: Categorical data, multimodal distributions

### Measures of Spread

**Variance:**
$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2$$

*Note: $n-1$ (Bessel's correction) for unbiased sample variance*

**Standard Deviation:** $s = \sqrt{s^2}$
- Same units as data
- ~68% of data within 1σ of mean (for Gaussian)

**Interquartile Range (IQR):** $Q_3 - Q_1$
- Robust to outliers
- Used for: Box plots, outlier detection ($<Q_1 - 1.5 \cdot IQR$ or $>Q_3 + 1.5 \cdot IQR$)

**Coefficient of Variation:** $CV = \frac{s}{\bar{x}}$
- Relative variability (unitless)
- Used for: Comparing variability across different scales

### Shape Measures

**Skewness:** Asymmetry of distribution
- Positive: Right tail longer
- Negative: Left tail longer
- Zero: Symmetric

**Kurtosis:** "Tailedness" of distribution
- High: Heavy tails (more outliers)
- Low: Light tails

---

## 2. Estimation Theory

### Point Estimators

An estimator $\hat{\theta}$ approximates the true parameter $\theta$.

**Properties of Estimators:**

| Property | Definition | Implication |
|----------|------------|-------------|
| **Unbiased** | $\mathbb{E}[\hat{\theta}] = \theta$ | On average, correct |
| **Consistent** | $\hat{\theta} \xrightarrow{p} \theta$ as $n \to \infty$ | Converges to truth |
| **Efficient** | Achieves Cramér-Rao lower bound | Minimum variance |

**Bias-Variance Decomposition (for estimators):**
$$\text{MSE}(\hat{\theta}) = \text{Bias}^2(\hat{\theta}) + \text{Var}(\hat{\theta})$$

### Maximum Likelihood Estimation (MLE)

Find $\theta$ that maximizes:
$$\mathcal{L}(\theta) = \prod_{i=1}^{n} p(x_i \mid \theta)$$

**Properties of MLE:**
- Asymptotically unbiased
- Asymptotically efficient
- Asymptotically normal

### Method of Moments

Equate sample moments to population moments:
$$\frac{1}{n}\sum_{i=1}^{n} x_i^k = \mathbb{E}[X^k]$$

Solve for parameters. Simpler but less efficient than MLE.

---

## 3. Confidence Intervals

A confidence interval provides a range likely containing the true parameter.

![Confidence Interval](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Confidence-interval.svg/500px-Confidence-interval.svg.png)
*95% confidence intervals — Source: [Wikipedia](https://en.wikipedia.org/wiki/Confidence_interval)*

### For Mean (Known Variance)

$$\bar{x} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$$

Where $z_{\alpha/2} = 1.96$ for 95% confidence.

### For Mean (Unknown Variance)

$$\bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}$$

Uses t-distribution with $n-1$ degrees of freedom.

### Correct Interpretation

**What 95% CI means:**
- If we repeated the experiment many times, 95% of the intervals would contain the true parameter.

**What it does NOT mean:**
- ❌ 95% probability that θ is in this interval
- ❌ 95% of the data falls in this interval

### Width of Confidence Interval

$$\text{Width} \propto \frac{\sigma}{\sqrt{n}}$$

- More data → narrower interval
- More variability → wider interval
- Higher confidence → wider interval

---

## 4. Hypothesis Testing

### The Framework

```
1. State null hypothesis H₀ (typically: no effect)
2. State alternative H₁ (what we want to show)
3. Choose significance level α (typically 0.05)
4. Compute test statistic from data
5. Calculate p-value
6. Decision: Reject H₀ if p-value < α
```

### Types of Errors

|  | H₀ True | H₀ False |
|---|---------|----------|
| **Reject H₀** | Type I Error (α) | ✓ Correct (Power) |
| **Fail to reject** | ✓ Correct | Type II Error (β) |

- **α (significance level)**: P(Type I error) = P(reject true H₀)
- **β**: P(Type II error) = P(fail to reject false H₀)
- **Power = 1 - β**: P(correctly reject false H₀)

### Common Statistical Tests

| Test | Use Case | Test Statistic |
|------|----------|----------------|
| **One-sample t-test** | Compare mean to known value | $t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}$ |
| **Two-sample t-test** | Compare means of two groups | $t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_p^2(1/n_1 + 1/n_2)}}$ |
| **Paired t-test** | Before/after comparisons | t-test on differences |
| **Chi-squared test** | Categorical independence | $\chi^2 = \sum \frac{(O - E)^2}{E}$ |
| **ANOVA** | Compare 3+ group means | F-statistic |

### p-Value: What It Really Means

**Definition:** Probability of observing data at least as extreme as ours, IF H₀ were true.

**p-value is NOT:**
- ❌ P(H₀ is true)
- ❌ Probability of making an error
- ❌ Effect size or importance

**p < 0.05 is arbitrary!** Consider effect size and practical significance.

### Multiple Testing Problem

Testing n hypotheses at α = 0.05:
- P(at least one false positive) = $1 - (1 - \alpha)^n \approx n\alpha$ for small α

**Corrections:**
- **Bonferroni:** Use $\alpha/n$
- **Benjamini-Hochberg:** Controls false discovery rate (FDR)

---

## 5. Correlation and Association

### Pearson Correlation

Measures **linear** relationship:
$$r = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$$

| r | Interpretation |
|---|----------------|
| 1.0 | Perfect positive linear |
| 0.7-0.9 | Strong positive |
| 0.4-0.6 | Moderate positive |
| 0.0 | No linear relationship |
| -1.0 | Perfect negative linear |

### Spearman Rank Correlation

Measures **monotonic** relationship (uses ranks instead of values).
- Robust to outliers
- Works for ordinal data

### Correlation ≠ Causation

![Spurious Correlation](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Correlation_examples2.svg/600px-Correlation_examples2.svg.png)
*Different types of correlations — Source: [Wikipedia](https://en.wikipedia.org/wiki/Correlation)*

**Why correlation doesn't imply causation:**
1. **Confounding variables**: Z causes both X and Y
2. **Reverse causation**: Y causes X
3. **Coincidence**: Spurious correlation

**To establish causation:** Randomized controlled experiments or causal inference methods.

---

## 6. Central Limit Theorem (CLT)

**The most important theorem in statistics!**

If $X_1, \ldots, X_n$ are i.i.d. with mean $\mu$ and variance $\sigma^2$:

$$\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1) \text{ as } n \to \infty$$

Or equivalently: $\bar{X} \sim \mathcal{N}(\mu, \sigma^2/n)$ approximately.

### Why CLT Matters

1. **Justifies normal approximations** for sample means
2. **Foundation for statistical tests** (t-test, z-test)
3. **Works regardless of original distribution** (with finite variance)
4. **Batch normalization in neural networks** relies on CLT-like effects

### Rule of Thumb

- n ≥ 30 usually sufficient for CLT
- For very skewed data, may need n > 100

---

## 7. Sampling and Resampling

### Random Sampling

Every element has known probability of selection.

**Bias sources:**
- **Selection bias:** Non-random selection
- **Survivorship bias:** Only observing "survivors"
- **Response bias:** Non-response related to variable of interest

### Stratified Sampling

Divide population into strata, sample from each.
- Ensures representation of subgroups
- Reduces variance

### Bootstrap

**Non-parametric method for estimating sampling distributions.**

**Algorithm:**
1. Draw B samples **with replacement** from data
2. Compute statistic for each resample
3. Use distribution of resampled statistics

**Applications:**
- Confidence intervals
- Standard errors
- **Random Forest = Bagging = Bootstrap Aggregating**

```python
import numpy as np

def bootstrap_ci(data, statistic=np.mean, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval."""
    n = len(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        resample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(resample))
    
    alpha = 1 - ci
    lower = np.percentile(bootstrap_stats, 100 * alpha/2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha/2))
    
    return lower, upper

# Example
data = np.random.exponential(2, 100)
lower, upper = bootstrap_ci(data)
print(f"95% CI for mean: [{lower:.2f}, {upper:.2f}]")
```

---

## 8. A/B Testing

### Framework for ML

1. **Define metric**: What are you measuring? (CTR, conversion, revenue)
2. **Determine sample size**: Power analysis
3. **Randomize**: Assign users randomly to control/treatment
4. **Run experiment**: Collect data
5. **Analyze**: Hypothesis test with pre-registered α
6. **Decision**: Deploy if significant AND practically meaningful

### Sample Size Calculation

For comparing two proportions:
$$n = \frac{(z_{\alpha/2} + z_{\beta})^2 (p_1(1-p_1) + p_2(1-p_2))}{(p_1 - p_2)^2}$$

**Rule of thumb:** To detect 5% relative lift with 80% power, need ~16,000 users per group.

### Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Peeking | Inflates false positive rate | Pre-specify sample size |
| Multiple testing | False discoveries | Correction (Bonferroni/FDR) |
| Selection bias | Biased groups | True randomization |
| Novelty effect | Temporary changes | Longer experiments |

---

## 9. Statistical Learning Concepts

### Train/Validation/Test Split

| Set | Purpose | Typical Size |
|-----|---------|--------------|
| **Training** | Model fitting | 60-80% |
| **Validation** | Hyperparameter tuning | 10-20% |
| **Test** | Final evaluation | 10-20% |

**Critical:** Never use test set for model selection!

### Cross-Validation

**K-Fold Cross-Validation:**
1. Split data into K folds
2. Train on K-1 folds, validate on remaining fold
3. Repeat K times, average results

**Benefits:**
- Uses all data for both training and validation
- More robust estimate of performance
- Standard: K=5 or K=10

### Bias-Variance from Statistical Perspective

For an estimator $\hat{f}$:
$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2[\hat{f}] + \text{Var}[\hat{f}] + \sigma^2$$

- **High bias:** Underfitting (model too simple)
- **High variance:** Overfitting (model too complex)

---

## Python Implementation Examples

```python
import numpy as np
from scipy import stats

# === Descriptive Statistics ===
data = np.random.normal(100, 15, 1000)

mean = np.mean(data)
median = np.median(data)
std = np.std(data, ddof=1)  # Sample std (n-1)
iqr = np.percentile(data, 75) - np.percentile(data, 25)

print(f"Mean: {mean:.2f}, Median: {median:.2f}")
print(f"Std: {std:.2f}, IQR: {iqr:.2f}")

# === Confidence Interval ===
n = len(data)
se = stats.sem(data)
ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
print(f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")

# === Two-Sample t-test ===
group_a = np.random.normal(100, 15, 50)
group_b = np.random.normal(105, 15, 50)

t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")

# === Effect Size (Cohen's d) ===
pooled_std = np.sqrt((np.var(group_a, ddof=1) + np.var(group_b, ddof=1)) / 2)
cohens_d = (np.mean(group_b) - np.mean(group_a)) / pooled_std
print(f"Cohen's d: {cohens_d:.3f}")

# === Chi-Squared Test ===
observed = np.array([[50, 30], [20, 40]])
chi2, p, dof, expected = stats.chi2_contingency(observed)
print(f"Chi-squared: {chi2:.2f}, p-value: {p:.4f}")

# === Correlation ===
x = np.random.normal(0, 1, 100)
y = 0.7 * x + np.random.normal(0, 0.5, 100)

pearson_r, pearson_p = stats.pearsonr(x, y)
spearman_r, spearman_p = stats.spearmanr(x, y)
print(f"Pearson r: {pearson_r:.3f}, Spearman ρ: {spearman_r:.3f}")

# === Power Analysis ===
from statsmodels.stats.power import TTestIndPower

power_analysis = TTestIndPower()
sample_size = power_analysis.solve_power(
    effect_size=0.5,  # Cohen's d
    alpha=0.05,
    power=0.8,
    alternative='two-sided'
)
print(f"Required sample size per group: {sample_size:.0f}")
```

---

## Key ML Applications Summary

| Statistic Concept | ML Application |
|-------------------|----------------|
| Mean/Variance | Feature normalization, batch norm |
| Confidence intervals | Model uncertainty bounds |
| Hypothesis testing | A/B testing, model comparison |
| Correlation | Feature selection, EDA |
| Bootstrap | Bagging, Random Forest |
| Cross-validation | Model selection, hyperparameter tuning |
| CLT | Justifies many approximations |
| Effect size | Practical significance |

---

## Further Reading

### Textbooks
- Wasserman - *All of Statistics*
- Casella & Berger - *Statistical Inference*
- Hastie, Tibshirani, Friedman - *Elements of Statistical Learning* (Ch. 7-8)

### Online Resources
- [Seeing Theory: Visual Statistics](https://seeing-theory.brown.edu/)
- [StatQuest YouTube Channel](https://www.youtube.com/c/joshstarmer)
- [Khan Academy Statistics](https://www.khanacademy.org/math/statistics-probability)

---

## Image Sources

- Central tendency comparison: [Wikipedia - Central Tendency](https://en.wikipedia.org/wiki/Central_tendency)
- Confidence interval illustration: [Wikipedia - Confidence Interval](https://en.wikipedia.org/wiki/Confidence_interval)
- Correlation examples: [Wikipedia - Correlation](https://en.wikipedia.org/wiki/Correlation)
