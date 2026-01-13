# Probability for Machine Learning

Probability theory provides the mathematical framework for reasoning under uncertainty.

## Notation

| Symbol | Meaning |
|--------|---------|
| $P(A)$ | Probability of event A |
| $P(A \mid B)$ | Conditional probability of A given B |
| $p(x)$ | Probability density (continuous) or mass (discrete) |
| $\mathbb{E}[X]$ | Expected value of X |
| $\text{Var}(X)$ | Variance of X |

---

## 1. Basic Probability

### Axioms of Probability

1. $0 \leq P(A) \leq 1$ for any event A
2. $P(\Omega) = 1$ (probability of sample space)
3. $P(A \cup B) = P(A) + P(B)$ if A and B are mutually exclusive

### Joint and Marginal Probability

**Joint probability:** $P(A, B) = P(A \cap B)$

**Marginalization (Sum Rule):**
$$P(A) = \sum_{b} P(A, B = b)$$

$$p(x) = \int p(x, y) \, dy$$

---

## 2. Conditional Probability

The probability of A given that B has occurred:

$$P(A \mid B) = \frac{P(A, B)}{P(B)}$$

### Product Rule (Chain Rule)

$$P(A, B) = P(A \mid B) P(B) = P(B \mid A) P(A)$$

For multiple variables:
$$P(X_1, X_2, \ldots, X_n) = P(X_1) \prod_{i=2}^{n} P(X_i \mid X_1, \ldots, X_{i-1})$$

---

## 3. Bayes' Theorem

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$$

In ML terminology:
$$\underbrace{P(\theta \mid \mathcal{D})}_{\text{posterior}} = \frac{\overbrace{P(\mathcal{D} \mid \theta)}^{\text{likelihood}} \cdot \overbrace{P(\theta)}^{\text{prior}}}{\underbrace{P(\mathcal{D})}_{\text{evidence}}}$$

**Why it matters:**
- Foundation of Bayesian inference
- Naive Bayes classifier
- Probabilistic modeling

---

## 4. Expectation and Variance

### Expected Value

**Discrete:**
$$\mathbb{E}[X] = \sum_{x} x \cdot P(X = x)$$

**Continuous:**
$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot p(x) \, dx$$

### Properties
- Linearity: $\mathbb{E}[aX + b] = a\mathbb{E}[X] + b$
- $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$
- For independent X, Y: $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$

### Variance

$$\text{Var}(X) = \mathbb{E}[(X - \mu)^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

**Standard Deviation:** $\sigma = \sqrt{\text{Var}(X)}$

### Covariance

$$\text{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$$

**Independence:** If X and Y are independent, $\text{Cov}(X, Y) = 0$

---

## 5. Common Distributions

### Bernoulli Distribution

For binary outcome:
$$P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}$$

- Mean: $p$
- Variance: $p(1-p)$

### Binomial Distribution

Number of successes in n trials:
$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

- Mean: $np$
- Variance: $np(1-p)$

### Gaussian (Normal) Distribution

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

Notation: $X \sim \mathcal{N}(\mu, \sigma^2)$

**Standard Normal:** $Z \sim \mathcal{N}(0, 1)$

**Multivariate Gaussian:**
$$p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

**Why it matters:**
- Central Limit Theorem
- Maximum Likelihood for regression
- Gaussian Mixture Models

### Categorical/Multinomial

For k categories:
$$P(X = k) = \pi_k, \quad \sum_{k=1}^{K} \pi_k = 1$$

### Poisson Distribution

For count data:
$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

---

## 6. Maximum Likelihood Estimation

Given data $\mathcal{D} = \{x_1, \ldots, x_n\}$, find parameters that maximize:

$$\mathcal{L}(\theta) = \prod_{i=1}^{n} p(x_i \mid \theta)$$

In practice, maximize log-likelihood:
$$\ell(\theta) = \sum_{i=1}^{n} \log p(x_i \mid \theta)$$

**Example: Gaussian MLE**

Given samples from $\mathcal{N}(\mu, \sigma^2)$:
- $\hat{\mu} = \frac{1}{n}\sum_{i=1}^{n} x_i$ (sample mean)
- $\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n} (x_i - \hat{\mu})^2$ (sample variance)

**Why it matters:** Foundation for training probabilistic models.

---

## 7. Information Theory

### Entropy

Measures uncertainty in a distribution:
$$H(X) = -\sum_{x} p(x) \log p(x)$$

For continuous: $h(X) = -\int p(x) \log p(x) \, dx$

### Cross-Entropy

$$H(p, q) = -\sum_{x} p(x) \log q(x)$$

**Why it matters:** Loss function for classification.

### KL Divergence

Measures difference between distributions:
$$D_{KL}(p \| q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}$$

Properties:
- $D_{KL} \geq 0$
- $D_{KL} = 0$ iff $p = q$
- Not symmetric

---

## 8. Key Applications in ML

| Concept | ML Application |
|---------|----------------|
| Bayes' theorem | Naive Bayes, Bayesian inference |
| MLE | Parameter estimation |
| Gaussian | Linear regression noise model |
| Bernoulli | Binary classification |
| Cross-entropy | Classification loss |
| KL divergence | VAE, regularization |

---

## Python Examples

```python
import numpy as np
from scipy import stats

# Gaussian distribution
mu, sigma = 0, 1
samples = np.random.normal(mu, sigma, 1000)

# Probability density
x = np.linspace(-3, 3, 100)
pdf = stats.norm.pdf(x, mu, sigma)

# MLE for Gaussian
mu_hat = np.mean(samples)
sigma_hat = np.std(samples)

# Computing entropy
probs = np.array([0.5, 0.3, 0.2])
entropy = -np.sum(probs * np.log(probs))
```

---

## Further Reading

- "Pattern Recognition and Machine Learning" by Bishop (Chapters 1-2)
- Stanford CS229 Probability Review
- "Information Theory, Inference, and Learning Algorithms" by MacKay
