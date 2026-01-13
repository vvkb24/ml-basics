# Python Refresher for Machine Learning

Essential Python concepts for ML practitioners.

---

## Data Types

### Lists
```python
# Creation and manipulation
numbers = [1, 2, 3, 4, 5]
numbers.append(6)
numbers.extend([7, 8])

# List comprehension
squares = [x**2 for x in range(10)]
evens = [x for x in numbers if x % 2 == 0]
```

### Dictionaries
```python
# Key-value pairs
model_config = {
    'learning_rate': 0.01,
    'epochs': 100,
    'batch_size': 32
}

# Dictionary comprehension
squared_dict = {x: x**2 for x in range(5)}
```

### Tuples (Immutable)
```python
point = (3, 4)
x, y = point  # Unpacking
```

---

## Functions

```python
# Basic function
def compute_mse(y_true, y_pred):
    """Compute Mean Squared Error."""
    return sum((t - p)**2 for t, p in zip(y_true, y_pred)) / len(y_true)

# Default arguments
def train(epochs=100, lr=0.01):
    pass

# *args and **kwargs
def flexible_func(*args, **kwargs):
    print(f"Args: {args}")
    print(f"Kwargs: {kwargs}")
```

### Lambda Functions
```python
# Anonymous functions
square = lambda x: x**2
sorted_by_second = sorted(pairs, key=lambda x: x[1])
```

---

## Classes

```python
class LinearModel:
    """Simple linear model class."""
    
    def __init__(self, n_features):
        self.weights = [0.0] * n_features
        self.bias = 0.0
    
    def predict(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
    
    def __repr__(self):
        return f"LinearModel(weights={self.weights}, bias={self.bias})"
```

### Inheritance
```python
class RegularizedModel(LinearModel):
    def __init__(self, n_features, reg_strength=0.01):
        super().__init__(n_features)
        self.reg_strength = reg_strength
```

---

## Control Flow

### Conditionals
```python
if condition:
    pass
elif another_condition:
    pass
else:
    pass
```

### Loops
```python
# For loop with enumerate
for i, item in enumerate(items):
    print(f"{i}: {item}")

# While loop
while condition:
    pass
    if break_condition:
        break
```

---

## File I/O

```python
# Reading
with open('data.txt', 'r') as f:
    content = f.read()
    # or lines = f.readlines()

# Writing
with open('output.txt', 'w') as f:
    f.write("Hello, World!")

# CSV with pandas (common in ML)
import pandas as pd
df = pd.read_csv('data.csv')
df.to_csv('output.csv', index=False)
```

---

## Exception Handling

```python
try:
    result = risky_operation()
except ValueError as e:
    print(f"Value error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    cleanup()
```

---

## Common Libraries

### NumPy
```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
matrix = np.zeros((3, 4))
random = np.random.randn(100, 10)
```

### Pandas
```python
import pandas as pd

df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
df['c'] = df['a'] + df['b']
df_filtered = df[df['a'] > 1]
```

### Matplotlib
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='data')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.savefig('plot.png')
```

---

## Type Hints

```python
from typing import List, Dict, Optional, Tuple

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 100,
    lr: float = 0.01
) -> Tuple[np.ndarray, float]:
    """Train and return weights and final loss."""
    pass
```

---

## Useful Patterns

### Context Managers
```python
from contextlib import contextmanager

@contextmanager
def timer():
    import time
    start = time.time()
    yield
    print(f"Elapsed: {time.time() - start:.2f}s")

with timer():
    train_model()
```

### Generators
```python
def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

for batch in batch_generator(training_data, 32):
    process(batch)
```

---

## Virtual Environments

```bash
# Create environment
python -m venv ml-env

# Activate (Windows)
ml-env\Scripts\activate

# Activate (Unix)
source ml-env/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

## Further Reading

- [Official Python Tutorial](https://docs.python.org/3/tutorial/)
- [Real Python](https://realpython.com/)
- NumPy documentation
- Pandas documentation
