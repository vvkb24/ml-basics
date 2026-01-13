# Probability for Machine Learning: Complete Guide

Probability theory provides the mathematical framework for reasoning under uncertainty—the foundation of all machine learning.

---

## Why Probability Matters in ML

Every ML model makes assumptions about data. Probability lets us:
- **Quantify uncertainty** in predictions
- **Learn from noisy data** by separating signal from noise
- **Make optimal decisions** under incomplete information
- **Generalize** from samples to populations

---

## Notation Reference

| Symbol | Meaning | Example |
|--------|---------|---------|
| $P(A)$ | Probability of event A | $P(\text{rain}) = 0.3$ |
| $P(A \mid B)$ | Conditional probability of A given B | $P(\text{rain} \mid \text{clouds})$ |
| $p(x)$ | Probability density (continuous) or mass (discrete) | $p(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}$ |
| $\mathbb{E}[X]$ | Expected value of X | $\mathbb{E}[\text{dice}] = 3.5$ |
| $\text{Var}(X)$ | Variance of X | Spread of distribution |
| $X \sim P$ | X is distributed according to P | $X \sim \mathcal{N}(0, 1)$ |

---

## 1. Foundations of Probability

### Kolmogorov Axioms

All of probability theory builds on three axioms:

1. **Non-negativity**: $P(A) \geq 0$ for any event A
2. **Normalization**: $P(\Omega) = 1$ (sample space has probability 1)
3. **Additivity**: $P(A \cup B) = P(A) + P(B)$ for mutually exclusive events

### Joint and Marginal Probability

**Joint probability** — probability of A AND B:
$$P(A, B) = P(A \cap B)$$

**Marginal probability** — obtained by summing/integrating out other variables:
$$P(A) = \sum_{b} P(A, B = b) \quad \text{(discrete)}$$
$$p(x) = \int p(x, y) \, dy \quad \text{(continuous)}$$

**ML Connection**: In graphical models (Bayesian networks), we marginalize over latent variables to compute observable probabilities.

### Independence

Events A and B are **independent** if:
$$P(A, B) = P(A) \cdot P(B)$$

**Conditional independence**: A ⊥ B | C if $P(A, B \mid C) = P(A \mid C) \cdot P(B \mid C)$

**ML Connection**: Naive Bayes assumes features are conditionally independent given the class label—often wrong but surprisingly effective.

---

## 2. Conditional Probability

The probability of A **given** that B occurred:

$$P(A \mid B) = \frac{P(A, B)}{P(B)}, \quad P(B) > 0$$

### Chain Rule (Product Rule)

$$P(A, B) = P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$$

For multiple variables:
$$P(X_1, X_2, \ldots, X_n) = P(X_1) \prod_{i=2}^{n} P(X_i \mid X_1, \ldots, X_{i-1})$$

**ML Connection**: Autoregressive models (GPT, LSTMs) generate sequences by modeling $P(x_t \mid x_1, \ldots, x_{t-1})$.

### Law of Total Probability

$$P(A) = \sum_i P(A \mid B_i) P(B_i)$$

where $\{B_i\}$ partitions the sample space.

---

## 3. Bayes' Theorem

The foundation of probabilistic inference:

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

### ML Terminology

$$\underbrace{P(\theta \mid \mathcal{D})}_{\text{posterior}} = \frac{\overbrace{P(\mathcal{D} \mid \theta)}^{\text{likelihood}} \cdot \overbrace{P(\theta)}^{\text{prior}}}{\underbrace{P(\mathcal{D})}_{\text{evidence}}}$$

| Term | What It Represents | Role |
|------|-------------------|------|
| **Prior** $P(\theta)$ | Belief before seeing data | Regularization |
| **Likelihood** $P(\mathcal{D}\mid\theta)$ | How well θ explains data | Data fitting |
| **Posterior** $P(\theta\mid\mathcal{D})$ | Updated belief after data | Final model |
| **Evidence** $P(\mathcal{D})$ | Normalizing constant | Model comparison |

### Why Bayes Matters for ML

1. **Regularization is a prior**: L2 regularization = Gaussian prior on weights
2. **Uncertainty quantification**: Posterior gives confidence, not just predictions
3. **Bayesian neural networks**: Model uncertainty in deep learning
4. **Naive Bayes classifier**: Fast, interpretable baseline

### Example: Medical Diagnosis

- Disease prevalence (prior): $P(\text{disease}) = 0.01$
- Test sensitivity: $P(\text{positive} \mid \text{disease}) = 0.99$
- Test specificity: $P(\text{negative} \mid \text{no disease}) = 0.95$

What's $P(\text{disease} \mid \text{positive})$?

$$P(\text{disease} \mid +) = \frac{0.99 \times 0.01}{0.99 \times 0.01 + 0.05 \times 0.99} = 0.167$$

**Only 16.7%!** The base rate (prior) dominates.

---

## 4. Expectation and Variance

### Expected Value (Mean)

**Discrete:**
$$\mathbb{E}[X] = \sum_{x} x \cdot P(X = x)$$

**Continuous:**
$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot p(x) \, dx$$

### Properties of Expectation

| Property | Formula | Use Case |
|----------|---------|----------|
| Linearity | $\mathbb{E}[aX + b] = a\mathbb{E}[X] + b$ | Scaling predictions |
| Additivity | $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$ | Always true! |
| Product (independent) | $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ | Only if independent |

### Variance

Measures spread around the mean:
$$\text{Var}(X) = \mathbb{E}[(X - \mu)^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

**Standard deviation**: $\sigma = \sqrt{\text{Var}(X)}$

### Covariance and Correlation

**Covariance** — measures linear relationship:
$$\text{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$$

**Correlation** — normalized covariance:
$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]$$

**ML Connection**: 
- Feature covariance matrix is central to PCA
- Correlation ≠ Causation!

---

## 5. Common Probability Distributions

### Discrete Distributions

![Probability Distributions](https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/Binomial_distribution_pmf.svg/600px-Binomial_distribution_pmf.svg.png)
*Binomial distribution for different parameters — Source: [Wikipedia](https://en.wikipedia.org/wiki/Binomial_distribution)*

#### Bernoulli Distribution

Single binary trial:
$$P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}$$

- Mean: $p$
- Variance: $p(1-p)$
- **ML Use**: Binary classification output

#### Binomial Distribution

Number of successes in n independent trials:
$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

- Mean: $np$
- Variance: $np(1-p)$

#### Poisson Distribution

Count data (events per interval):
$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

- Mean = Variance = $\lambda$
- **ML Use**: Count regression, rare event modeling

#### Categorical/Multinomial

K categories with probabilities $\pi_1, \ldots, \pi_K$ summing to 1.
- **ML Use**: Multi-class classification output (softmax)

### Continuous Distributions

![Normal Distribution](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/600px-Normal_Distribution_PDF.svg.png)
*Gaussian (normal) distribution for different σ — Source: [Wikipedia](https://en.wikipedia.org/wiki/Normal_distribution)*

#### Gaussian (Normal) Distribution

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

Notation: $X \sim \mathcal{N}(\mu, \sigma^2)$

**Why it's everywhere**:
1. Central Limit Theorem: Sums converge to Gaussian
2. Maximum entropy distribution for given mean/variance
3. Closed under linear operations

#### Multivariate Gaussian

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

- **ML Uses**: Gaussian Naive Bayes, GMM, linear regression likelihood, VAE

#### Exponential Distribution

Time until event:
$$p(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

- Mean: $1/\lambda$
- **ML Use**: Survival analysis, time-to-event

#### Beta Distribution

Probability of probability (prior for Bernoulli):
$$p(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)}$$

- **ML Use**: Bayesian inference for proportions

---

## 6. Maximum Likelihood Estimation (MLE)

Given data $\mathcal{D} = \{x_1, \ldots, x_n\}$, find parameters maximizing:

$$\mathcal{L}(\theta) = \prod_{i=1}^{n} p(x_i \mid \theta)$$

In practice, maximize **log-likelihood** (avoids numerical underflow):
$$\ell(\theta) = \sum_{i=1}^{n} \log p(x_i \mid \theta)$$

### Example: Gaussian MLE

Given samples from $\mathcal{N}(\mu, \sigma^2)$:
$$\hat{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^{n} x_i$$
$$\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^{n} (x_i - \hat{\mu})^2$$

Note: MLE variance is biased; unbiased uses $n-1$.

### MLE vs MAP

| Aspect | MLE | MAP |
|--------|-----|-----|
| Objective | Maximize likelihood | Maximize posterior |
| Prior | None (flat) | Included |
| Formula | $\arg\max_\theta P(D\mid\theta)$ | $\arg\max_\theta P(D\mid\theta)P(\theta)$ |
| Regularization | None | Implicit |

**ML Connection**: Training neural networks with cross-entropy loss = MLE with categorical likelihood.

---

## 7. Information Theory

### Entropy

Measures uncertainty/information content:
$$H(X) = -\sum_{x} p(x) \log p(x) = \mathbb{E}[-\log p(X)]$$

- **Low entropy**: Predictable (peaked distribution)
- **High entropy**: Uncertain (flat distribution)

![Entropy](https://upload.wikimedia.org/wikipedia/commons/thumb/2/22/Binary_entropy_plot.svg/400px-Binary_entropy_plot.svg.png)
*Binary entropy function — Source: [Wikipedia](https://en.wikipedia.org/wiki/Binary_entropy_function)*

### Cross-Entropy

Entropy using wrong distribution q instead of true p:
$$H(p, q) = -\sum_{x} p(x) \log q(x)$$

**ML Connection**: Cross-entropy loss for classification:
$$\mathcal{L} = -\sum_{i} y_i \log \hat{y}_i$$

### KL Divergence

Measures "distance" between distributions:
$$D_{KL}(p \| q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)$$

Properties:
- $D_{KL} \geq 0$ (Gibbs' inequality)
- $D_{KL} = 0 \iff p = q$
- **Not symmetric**: $D_{KL}(p\|q) \neq D_{KL}(q\|p)$

**ML Connection**: 
- VAE loss includes KL term: $D_{KL}(q(z|x) \| p(z))$
- Minimizing cross-entropy = minimizing KL from true distribution

---

## 8. Key ML Applications Summary

| Probability Concept | ML Application |
|---------------------|----------------|
| Bayes' theorem | Bayesian inference, Naive Bayes |
| Conditional probability | Graphical models, language models |
| MLE | Parameter estimation, neural network training |
| Gaussian distribution | Linear regression, GMM, VAE |
| Bernoulli/Categorical | Classification outputs |
| Cross-entropy | Classification loss function |
| KL divergence | VAE, regularization, model comparison |
| Entropy | Decision trees (information gain) |

---

## Python Implementation Examples

```python
import numpy as np
from scipy import stats

# === Gaussian Distribution ===
mu, sigma = 0, 1
samples = np.random.normal(mu, sigma, 1000)
x = np.linspace(-4, 4, 100)
pdf = stats.norm.pdf(x, mu, sigma)

# === MLE for Gaussian ===
mu_hat = np.mean(samples)
sigma_hat = np.std(samples, ddof=0)  # MLE uses n, not n-1

# === Bayes Example: Medical Test ===
prior_disease = 0.01
sensitivity = 0.99  # P(+|disease)
specificity = 0.95  # P(-|no disease)

p_positive = sensitivity * prior_disease + (1 - specificity) * (1 - prior_disease)
posterior = (sensitivity * prior_disease) / p_positive
print(f"P(disease|positive) = {posterior:.3f}")  # 0.167

# === Entropy ===
probs = np.array([0.5, 0.3, 0.2])
entropy = -np.sum(probs * np.log2(probs))
print(f"Entropy = {entropy:.3f} bits")

# === KL Divergence ===
p = np.array([0.4, 0.6])
q = np.array([0.5, 0.5])
kl_div = np.sum(p * np.log(p / q))
print(f"KL(p||q) = {kl_div:.4f}")
```

---

## Further Reading

### Textbooks
- Bishop - *Pattern Recognition and Machine Learning* (Chapters 1-2)
- Murphy - *Probabilistic Machine Learning: An Introduction*
- MacKay - *Information Theory, Inference, and Learning Algorithms*

### Online Resources
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf)
- [3Blue1Brown: Bayes' Theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM)
- [Seeing Theory: Visual Probability](https://seeing-theory.brown.edu/)

---

## Image Sources

- Binomial distribution: [Wikipedia - Binomial Distribution](https://en.wikipedia.org/wiki/Binomial_distribution)
- Normal distribution: [Wikipedia - Normal Distribution](https://en.wikipedia.org/wiki/Normal_distribution)
- Binary entropy: [Wikipedia - Binary Entropy Function](https://en.wikipedia.org/wiki/Binary_entropy_function)
