# Contributing to ML Math and Applications

Thank you for your interest in contributing! This document provides guidelines for contributions.

## 🎯 Contribution Types

### 1. Content Contributions
- **New Algorithms**: Follow the [algorithm standard](#algorithm-standard)
- **Mathematical Derivations**: Must be complete with all steps shown
- **Experiments**: Add Jupyter notebooks with visualizations
- **Applications**: Real-world use cases with datasets

### 2. Documentation
- Fix typos and grammatical errors
- Improve explanations for clarity
- Add examples and diagrams
- Translate content (create `docs/translations/`)

### 3. Code Improvements
- Bug fixes
- Performance optimizations
- Test coverage improvements
- Code style consistency

## 📋 Algorithm Standard

Every new algorithm must include:

```
algorithm_name/
├── README.md           # Overview, equations summary, usage example
├── theory.md           # Complete mathematical treatment
├── scratch.py          # From-scratch NumPy implementation
├── sklearn_impl.py     # scikit-learn/framework implementation
└── experiments.ipynb   # Visualizations and experiments
```

### theory.md Requirements

1. **Problem Definition**: What problem does this algorithm solve?
2. **Assumptions**: What assumptions does the algorithm make?
3. **Mathematical Formulation**: Define all notation
4. **Loss Function**: Derive the loss function
5. **Optimization**: Show the optimization derivation
6. **Regularization**: Discuss regularization variants
7. **Complexity**: Time and space complexity

### scratch.py Requirements

```python
"""
Algorithm Name - From Scratch Implementation

Mathematical basis: [brief description]
"""

import numpy as np
from typing import Optional

class AlgorithmName:
    """
    Algorithm implementation with detailed docstrings.
    
    Parameters
    ----------
    param1 : type
        Description with mathematical context
        
    Attributes
    ----------
    weights_ : np.ndarray
        Learned parameters after fitting
        
    Examples
    --------
    >>> model = AlgorithmName(param1=value)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X_test)
    """
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AlgorithmName':
        """
        Fit the model to training data.
        
        The optimization follows:
        [brief math description]
        """
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
```

## 🔄 Pull Request Process

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/algorithm-name`
3. **Make changes** following the standards above
4. **Test**: Run `pytest tests/` and ensure all pass
5. **Lint**: Ensure code follows PEP 8
6. **Commit**: Use clear commit messages
7. **Push** and create a **Pull Request**

### Commit Message Format

```
[type]: brief description

- Detail 1
- Detail 2

Closes #issue_number
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## 🧪 Testing Requirements

- All implementations must have corresponding tests
- Tests should verify mathematical correctness
- Compare against known solutions or framework implementations

```python
# Example test
def test_linear_regression_normal_equation():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    
    model = LinearRegressionScratch(method='normal_equation')
    model.fit(X, y)
    
    np.testing.assert_array_almost_equal(
        model.weights_, 
        np.array([1, 2]), 
        decimal=5
    )
```

## 📐 Mathematical Notation Standards

Use consistent notation throughout:

| Symbol | Meaning |
|--------|---------|
| $X$ | Feature matrix $(n \times d)$ |
| $y$ | Target vector $(n \times 1)$ |
| $\theta$ or $w$ | Weight vector |
| $\hat{y}$ | Predictions |
| $\mathcal{L}$ | Loss function |
| $\lambda$ | Regularization parameter |
| $\eta$ | Learning rate |

## 💬 Questions?

Open an issue with the `question` label or start a discussion.

---

Thank you for helping make machine learning education more accessible! 🙏
