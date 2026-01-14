# PyTorch: The Language of Differentiable Computing

A concept-first, failure-aware guide to PyTorch for serious practitioners.

---

## 1. Why PyTorch Exists: Conceptual and Historical Context

### The Problem PyTorch Solves

Before PyTorch (2016):
- **TensorFlow 1.x**: Static computation graphs (define-then-run)
- **Theano**: Slow compilation, hard debugging
- **Pain point**: Can't use Python debugger on neural networks

PyTorch introduced:
1. **Dynamic graphs** (define-by-run): Build graph while executing
2. **Pythonic API**: Feels like NumPy with gradients
3. **Eager execution**: See results immediately
4. **Easy debugging**: Standard Python tools work

### The Fundamental Innovation

**Static graphs** (TensorFlow 1.x):
```python
# Define graph
x = tf.placeholder(tf.float32)
y = x * 2
# Execute later
sess.run(y, feed_dict={x: 5})  # Only now does computation happen
```

**Dynamic graphs** (PyTorch):
```python
x = torch.tensor(5.0, requires_grad=True)
y = x * 2  # Computation happens NOW
print(y)  # tensor(10., grad_fn=<MulBackward0>)
```

**Why this matters**: Debugging is normal Python. Loops, conditionals work naturally.

### Historical Context

- **2002**: Torch (Lua-based, NYU)
- **2016**: PyTorch released by Facebook AI Research (FAIR)
- **2018**: PyTorch 1.0 (production-ready)
- **2022**: PyTorch joins Linux Foundation (neutral governance)
- **Today**: Dominant in research, growing in production

**Research DNA**: PyTorch was designed for researchers who iterate fast. Production features came later.

---

## 2. Mathematical Abstractions PyTorch Encodes

### Tensors as the Universal Representation

A tensor is a multi-dimensional array with:
- **Shape**: Dimensions $(d_1, d_2, \ldots, d_n)$
- **Dtype**: Data type (float32, int64, etc.)
- **Device**: CPU or GPU
- **Gradient tracking**: For automatic differentiation

$$\text{Tensor} = (\text{data}, \text{shape}, \text{dtype}, \text{device}, \text{requires\_grad})$$

### Automatic Differentiation (Autograd)

PyTorch encodes the **chain rule** automatically:

For $f = g \circ h$, the derivative is:
$$\frac{\partial f}{\partial x} = \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial x}$$

**In PyTorch**:
```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2  # Records operation
z = y * 3   # Records operation

z.backward()  # Computes dz/dx via chain rule
print(x.grad)  # tensor(12.) because d(3x²)/dx = 6x = 12
```

### The Computation Graph

Every operation builds a directed acyclic graph (DAG):

```
x (leaf) ---> [square] ---> y ---> [multiply by 3] ---> z
                                                           |
                                                     z.backward()
```

`backward()` traverses this graph in reverse to compute gradients.

### The Training Loop Pattern

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass: compute predictions
        predictions = model(batch.inputs)
        
        # Compute loss
        loss = loss_fn(predictions, batch.targets)
        
        # Backward pass: compute gradients
        optimizer.zero_grad()  # Clear old gradients!
        loss.backward()        # Compute new gradients
        
        # Update parameters
        optimizer.step()       # Apply gradients to weights
```

This encodes gradient descent:
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}$$

---

## 3. Assumptions Hidden in Common Functions

### `tensor.backward()` Assumptions

```python
loss.backward()  # What does this assume?
```

**Hidden assumptions:**
1. **Scalar output**: `backward()` with no arguments only works on scalars
2. **Gradients accumulate**: Must call `optimizer.zero_grad()` first!
3. **Graph is retained once**: Use `retain_graph=True` for multiple backwards

**The accumulation trap:**
```python
for batch in data:
    loss = compute_loss(batch)
    loss.backward()  # Gradients ACCUMULATE!

# After 10 batches, gradients are 10x what you expect!
# Fix: Always zero_grad() before backward()
```

### `model.train()` vs `model.eval()`

```python
model.train()  # Training mode
model.eval()   # Evaluation mode
```

**What changes:**
| Component | train() | eval() |
|-----------|---------|--------|
| Dropout | Active (random drops) | Disabled (deterministic) |
| BatchNorm | Uses batch statistics | Uses running statistics |
| Gradients | Enabled | Still enabled (use no_grad) |

**Common mistake:**
```python
model.eval()  # Only changes behavior of some layers
predictions = model(test_data)  # Still computing gradients!

# Correct:
model.eval()
with torch.no_grad():  # Actually disables gradient computation
    predictions = model(test_data)
```

### `nn.CrossEntropyLoss()` Assumptions

```python
loss = nn.CrossEntropyLoss()(logits, targets)
```

**Hidden assumptions:**
1. **Logits, not probabilities**: Do NOT apply softmax before!
2. **Targets are class indices**: Not one-hot encoded!
3. **Reduction is mean**: Averages over batch by default

**The double-softmax trap:**
```python
# WRONG: softmax applied twice
logits = model(x)
probs = F.softmax(logits, dim=1)  # Softmax #1
loss = nn.CrossEntropyLoss()(probs, y)  # Contains Softmax #2!

# CORRECT: pass raw logits
logits = model(x)
loss = nn.CrossEntropyLoss()(logits, y)
```

### `DataLoader` Shuffling

```python
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

**Hidden assumptions:**
1. **Shuffling per epoch**: Different order each traversal
2. **Drop last batch if incomplete**: Use `drop_last=True` if needed for BatchNorm
3. **Memory pinning**: `pin_memory=True` speeds up GPU transfer

---

## 4. What PyTorch Does NOT Protect Against

### 4.1 Silent Broadcasting Errors

```python
a = torch.randn(10, 5)
b = torch.randn(5, 10)  # Wrong shape!

c = a + b  # No error due to broadcasting, but likely a bug!
print(c.shape)  # torch.Size([10, 10]) - probably not what you wanted
```

**Protection**: Always check tensor shapes during debugging.

### 4.2 GPU-CPU Mismatch

```python
model = Model().cuda()  # Model on GPU
x = torch.randn(32, 10)  # Data on CPU

y = model(x)  # RuntimeError: Input and weight tensors must be on same device
```

**Solution**: 
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
x = x.to(device)
```

### 4.3 In-Place Operations Break Gradient

```python
x = torch.randn(5, requires_grad=True)
x += 1  # In-place modification

loss = x.sum()
loss.backward()  # RuntimeError: in-place operation modified a leaf variable
```

**Solution**: Avoid in-place operations on tensors requiring grad:
```python
x = x + 1  # Creates new tensor, safe
```

### 4.4 Memory Leaks from Storing Tensors

```python
losses = []
for batch in data:
    loss = compute_loss(batch)
    losses.append(loss)  # Stores entire computation graph!

# GPU memory explodes because each loss holds its graph
```

**Solution**: Detach or convert to Python scalar:
```python
losses.append(loss.item())  # Extracts Python float
# or
losses.append(loss.detach())  # Removes from graph
```

---

## 5. Failure Modes with Concrete Examples

### Failure Mode 1: Forgetting zero_grad()

```python
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    loss = compute_loss()
    loss.backward()
    optimizer.step()  # Gradients accumulate!
    
# After epoch 1: gradients = g
# After epoch 2: gradients = g + g = 2g
# After epoch 100: gradients = 100g  (exploding!)
```

**Symptom**: Loss oscillates wildly or explodes.

**Fix**: Always `optimizer.zero_grad()` before `backward()`.

### Failure Mode 2: Model Not in Training Mode

```python
model = MyModel()
# Forgot model.train()

for batch in train_loader:
    loss = criterion(model(batch.x), batch.y)
    loss.backward()
    optimizer.step()

# If model has Dropout:
# - Dropout is DISABLED (using eval mode default)
# - Model trains without regularization
# - Overfits badly
```

**Symptom**: Model overfits despite dropout in architecture.

### Failure Mode 3: Learning Rate Too High

```python
optimizer = optim.SGD(model.parameters(), lr=1.0)  # Way too high

# Training:
# Epoch 1: loss = 2.3
# Epoch 2: loss = 14.7
# Epoch 3: loss = nan
```

**Symptom**: Loss increases or becomes NaN.

**Debug**: Start with lr=1e-3, decrease if unstable.

### Failure Mode 4: BatchNorm with Small Batches

```python
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.BatchNorm1d(50),  # Problem!
    nn.ReLU()
)

# With batch_size=1:
x = torch.randn(1, 100)
model(x)  # Error or incorrect normalization
```

**Problem**: BatchNorm computes mean/std over batch. With 1 sample, std=0.

**Solutions**:
- Use `batch_size >= 16`
- Use GroupNorm or LayerNorm for small batches
- Use `nn.BatchNorm1d(50, track_running_stats=False)` in training

### Failure Mode 5: Data Not Shuffled

```python
# Sorted by class: all cats, then all dogs
dataset = sorted_cats_dogs_dataset

loader = DataLoader(dataset, batch_size=32, shuffle=False)  # Bug!

# First 50 batches: all cats
# Last 50 batches: all dogs
# Model learns to predict "cat" then "dog" based on batch order
```

**Symptom**: Good training loss, terrible generalization.

---

## 6. Performance and Scaling Trade-offs

### When PyTorch is Fast

| Situation | Speed | Why |
|-----------|-------|-----|
| Batched operations | ⚡ Fast | GPU parallelism |
| Matrix multiply | ⚡ Very fast | cuBLAS optimized |
| Conv2d | ⚡ Fast | cuDNN convolution |
| Modern GPUs | ⚡ Fast | Tensor cores (A100, V100) |

### When PyTorch is Slow

| Situation | Speed | Why | Solution |
|-----------|-------|-----|----------|
| Small batches | 🐌 Slow | GPU underutilized | Increase batch size |
| Many small ops | 🐌 Slow | Python overhead | Use TorchScript |
| CPU training | 🐌 Slow | No GPU acceleration | Use GPU |
| Data loading | 🐌 Bottleneck | Disk I/O | `num_workers > 0` |

### Memory Optimization

```python
# FP16 training (half memory, faster on modern GPUs)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in data:
    with autocast():
        loss = model(batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Gradient checkpointing (trade compute for memory)
from torch.utils.checkpoint import checkpoint
output = checkpoint(expensive_layer, input)
```

### Multi-GPU Strategies

| Strategy | Use Case | Complexity |
|----------|----------|------------|
| `DataParallel` | Single machine, multiple GPUs | Easy |
| `DistributedDataParallel` | Best performance | Medium |
| `FSDP` | Very large models | Complex |

---

## 7. When NOT to Use PyTorch

### Use JAX Instead When:
- You need **functional programming** style
- You want **composable transformations** (vmap, pmap)
- You're doing **research on novel architectures**

### Use TensorFlow Instead When:
- You need **TensorFlow Serving** for production
- You're using **TensorFlow-specific tools** (TFX, TFLite)
- You need **TPU support** (though PyTorch/XLA exists)

### Use scikit-learn Instead When:
- You're doing **classical ML** (trees, linear models)
- Data is **small** (< 100k samples)
- You don't need **custom gradients**

### Use NumPy Instead When:
- You don't need **gradients**
- You don't need **GPU**
- You're doing **simple numerical operations**

---

## 8. Real-World Anti-Patterns

### Anti-Pattern 1: Logic in Forward Pass Without Scripting

```python
# Works in Python, fails in TorchScript
def forward(self, x):
    if x.shape[0] > 10:  # Dynamic condition
        return self.big_branch(x)
    return self.small_branch(x)
```

**Problem**: TorchScript needs to trace/script this; dynamic shapes are tricky.

**Solution**: Use `torch.jit.script` or restructure for static shapes.

### Anti-Pattern 2: Storing Entire Dataset in GPU

```python
# WRONG: Moves all data to GPU at once
train_data = train_data.cuda()  # 100GB dataset = OOM

# CORRECT: Move batch by batch
for batch in loader:
    batch = batch.cuda()
    # process...
```

### Anti-Pattern 3: Not Using num_workers

```python
# SLOW: Main process loads data
loader = DataLoader(dataset, batch_size=32, num_workers=0)

# FAST: Parallel data loading
loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
```

### Anti-Pattern 4: Random Seeds Not Set

```python
# Non-reproducible training
model = train(data)  # Different results each run

# Reproducible:
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Anti-Pattern 5: Evaluating Without no_grad

```python
# SLOW: Computing gradients during evaluation
model.eval()
for batch in test_loader:
    predictions = model(batch)  # Still tracking gradients!

# FAST: Disable gradient computation
model.eval()
with torch.no_grad():
    for batch in test_loader:
        predictions = model(batch)
```

---

## Summary: PyTorch Decision Framework

| Question | Answer | Action |
|----------|--------|--------|
| Need gradients? | Yes | Use PyTorch |
| Need GPU? | Yes | Use PyTorch |
| Classical ML? | Yes | Use scikit-learn |
| Just linear algebra? | Yes | Use NumPy |
| Production serving? | Yes | Export to ONNX or TorchScript |
| Debugging? | Yes | PyTorch eager mode |

---

## Essential Imports Cheatsheet

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Common layers
nn.Linear, nn.Conv2d, nn.BatchNorm1d, nn.Dropout
nn.ReLU, nn.Sigmoid, nn.Softmax

# Loss functions
nn.CrossEntropyLoss()  # Classification (logits, not probs!)
nn.MSELoss()           # Regression
nn.BCEWithLogitsLoss() # Binary classification

# Optimizers
optim.SGD(params, lr=0.01, momentum=0.9)
optim.Adam(params, lr=0.001)
optim.AdamW(params, lr=0.001, weight_decay=0.01)
```

---

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- ["Deep Learning with PyTorch"](https://www.manning.com/books/deep-learning-with-pytorch) - Eli Stevens
- [fast.ai Course](https://course.fast.ai/) - Jeremy Howard
- ["Programming PyTorch for Deep Learning"](https://www.oreilly.com/library/view/programming-pytorch-for/9781492045342/) - Ian Pointer
