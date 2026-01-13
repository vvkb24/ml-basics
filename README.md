# Machine Learning: Mathematical Foundations and Applications

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"/>
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"/>
  <img src="https://img.shields.io/badge/Contributions-welcome-orange.svg" alt="Contributions Welcome"/>
</p>

<p align="center">
  <strong>A comprehensive, mathematically rigorous guide to machine learning — from first principles to production.</strong>
  <strong>If you feel something is missing in this, dont wait addddddd</strong>
</p>

---

## 🎯 Mission

This repository bridges the gap between mathematical theory and practical implementation in machine learning. Every algorithm includes:

- **Complete mathematical derivations** — no steps skipped
- **From-scratch implementations** — understand every line of code
- **Framework implementations** — leverage scikit-learn, PyTorch, and TensorFlow
- **Interactive experiments** — visualize and explore concepts

## 📚 Who Is This For?

| Level | Description |
|-------|-------------|
| 🌱 **Beginners** | Start with [foundations/](./foundations/) and the [beginner learning path](./docs/learning_paths/beginner.md) |
| 📈 **Intermediate** | Dive into [algorithms/](./algorithms/) and [neural_networks/](./neural_networks/) |
| 🚀 **Advanced** | Explore [transformers/](./transformers/) and [deep_learning/](./deep_learning/) |
| 🔬 **Researchers** | Use as a reference for mathematical derivations |

## 🗂️ Repository Structure

```
ml-math-and-applications/
│
├── docs/                      # Learning paths, prerequisites, theory
├── foundations/               # NumPy, Matplotlib, Python refresher
├── algorithms/
│   ├── supervised/            # Linear regression, logistic regression, SVM, etc.
│   ├── unsupervised/          # K-means, GMM, hierarchical clustering
│   └── ensemble_methods/      # Random forest, gradient boosting, XGBoost
├── dimensionality_reduction/  # PCA, LDA, t-SNE, UMAP
├── neural_networks/           # Perceptron, MLP, backpropagation
├── deep_learning/             # CNN, RNN, LSTM, attention
├── transformers/              # Self-attention, transformers from scratch
├── frameworks/                # scikit-learn, PyTorch, TensorFlow guides
├── applications/              # Real-world projects
├── experiments/               # Ablation studies, benchmarking
├── utils/                     # Helper functions
├── tests/                     # Unit tests
└── references/                # Books, papers, resources
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-math-and-applications.git
cd ml-math-and-applications

# Option 1: Using pip
pip install -r requirements.txt

# Option 2: Using conda
conda env create -f environment.yml
conda activate ml-math
```

### Your First Algorithm

```python
# From-scratch linear regression
from algorithms.supervised.linear_regression.scratch import LinearRegressionScratch
import numpy as np

# Generate sample data
X = np.random.randn(100, 2)
y = 3 * X[:, 0] + 2 * X[:, 1] + 1 + np.random.randn(100) * 0.1

# Train model
model = LinearRegressionScratch(method='normal_equation')
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
print(f"R² Score: {model.score(X, y):.4f}")
```

## 📖 Algorithm Standard

Each algorithm module follows a consistent structure:

```
algorithm_name/
├── README.md           # Overview and quick start
├── theory.md           # Mathematical foundations
├── scratch.py          # NumPy implementation
├── sklearn_impl.py     # Framework implementation
└── experiments.ipynb   # Interactive exploration
```

**Mathematical documentation includes:**
1. Problem definition
2. Assumptions and prerequisites
3. Loss function formulation
4. Optimization derivation
5. Regularization variants
6. Computational complexity

## 🧮 Mathematical Prerequisites

Before diving into algorithms, ensure familiarity with:

| Topic | Key Concepts | Resource |
|-------|--------------|----------|
| Linear Algebra | Vectors, matrices, eigenvalues | [docs/math_prerequisites/linear_algebra.md](./docs/math_prerequisites/linear_algebra.md) |
| Probability | Distributions, Bayes' theorem | [docs/math_prerequisites/probability.md](./docs/math_prerequisites/probability.md) |
| Calculus | Gradients, chain rule | [docs/math_prerequisites/calculus.md](./docs/math_prerequisites/calculus.md) |
| Optimization | Gradient descent, convexity | [docs/math_prerequisites/optimization.md](./docs/math_prerequisites/optimization.md) |

## 🛠️ Technologies

| Category | Tools |
|----------|-------|
| **Core** | Python 3.8+, NumPy, SciPy |
| **ML Frameworks** | scikit-learn, PyTorch, TensorFlow |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Notebooks** | Jupyter Lab |
| **Testing** | pytest |

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

**Ways to contribute:**
- 📝 Fix typos or improve explanations
- 🧮 Add mathematical derivations
- 💻 Implement new algorithms
- 🧪 Add experiments and visualizations
- 📚 Improve documentation

## 📜 License

This project is licensed under the MIT License — see [LICENSE](./LICENSE) for details.

## 🙏 Acknowledgments

This project draws inspiration from:
- Stanford CS229/CS231n courses
- "Pattern Recognition and Machine Learning" by Bishop
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
- The open-source ML community

---

<p align="center">
  <em>Star ⭐ this repo if you find it helpful!</em>
</p>
