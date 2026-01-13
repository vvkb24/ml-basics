# Frequently Asked Questions

Common questions about machine learning concepts.

---

## Getting Started

### Q: What math do I need to know for ML?

**A:** Essential topics:
- **Linear algebra**: Vectors, matrices, eigenvalues
- **Calculus**: Derivatives, gradients, chain rule
- **Probability**: Distributions, Bayes' theorem
- **Statistics**: Estimation, hypothesis testing

See [Math Prerequisites](./math_prerequisites/) for detailed coverage.

### Q: Do I need to know how to code algorithms from scratch?

**A:** While you can use libraries like scikit-learn, understanding implementation helps you:
- Debug issues
- Choose appropriate algorithms
- Optimize for specific use cases
- Understand limitations

---

## Algorithms

### Q: When should I use linear regression vs. neural networks?

**A:**

| Use Linear Regression | Use Neural Networks |
|----------------------|---------------------|
| Small dataset | Large dataset |
| Interpretability needed | Complex patterns |
| Linear relationships | Non-linear relationships |
| Limited compute | Sufficient compute |

### Q: What's the difference between classification and regression?

**A:**
- **Regression**: Predicting continuous values (price, temperature)
- **Classification**: Predicting discrete categories (spam/not spam, cat/dog)

### Q: How do I choose between Random Forest and Gradient Boosting?

**A:**
- **Random Forest**: Parallelizable, robust, good baseline
- **Gradient Boosting**: Often higher accuracy, sequential, more tuning needed

---

## Training

### Q: How much data do I need?

**A:** Depends on:
- Model complexity (more parameters → more data)
- Problem difficulty
- Feature quality

**Rules of thumb:**
- Linear models: 10× features
- Neural networks: 1000× parameters (varies widely)
- Start with what you have, collect more if needed

### Q: How do I prevent overfitting?

**A:**
1. Get more training data
2. Use regularization (L1, L2, dropout)
3. Reduce model complexity
4. Early stopping
5. Data augmentation
6. Cross-validation for model selection

### Q: How do I handle imbalanced classes?

**A:**
1. Collect more minority class data
2. Resample (oversample minority / undersample majority)
3. Use class weights
4. Use appropriate metrics (F1, AUC, not accuracy)
5. Generate synthetic samples (SMOTE)

---

## Evaluation

### Q: Why is my test accuracy lower than training accuracy?

**A:** This is normal (training data was seen during learning). Large gaps indicate overfitting. Solutions:
- Regularization
- More training data
- Simpler model

### Q: Should I use accuracy as my metric?

**A:** Only for balanced classes. For imbalanced data, prefer:
- Precision/Recall
- F1 Score
- AUC-ROC
- Precision-Recall AUC

### Q: What's a good R² score?

**A:** Depends on the domain:
- Physical sciences: > 0.9 often expected
- Social sciences: 0.3-0.5 may be good
- Finance: Any positive R² can be valuable

---

## Deep Learning

### Q: GPU vs CPU for training?

**A:**
- **CPU**: Small models, debugging, inference
- **GPU**: Training neural networks (10-100× faster for matrix operations)

### Q: How do I choose architecture depth and width?

**A:**
- Start with proven architectures for your domain
- Deeper: Learn more abstract features
- Wider: Learn more features per layer
- Use validation set to compare architectures

### Q: What is transfer learning?

**A:** Using a pre-trained model (trained on large dataset) and adapting it to new task:
1. Freeze base layers
2. Replace final layer(s)
3. Fine-tune on new data

---

## Practical Tips

### Q: How do I structure an ML project?

**A:**
1. Define problem and success metrics
2. Collect and explore data
3. Prepare data (clean, split, preprocess)
4. Start with simple baseline
5. Iteratively improve model
6. Evaluate on test set (once!)
7. Deploy and monitor

### Q: How do I debug a model that's not learning?

**A:** Check:
1. Data preprocessing (normalization, encoding)
2. Learning rate (try 10× smaller)
3. Model capacity (too simple?)
4. Loss function (appropriate for task?)
5. Gradient flow (vanishing/exploding?)
6. Training long enough (more epochs?)
7. Random seed (reproducibility)

---

## Resources

### Q: What resources do you recommend?

**A:**

**Books:**
- Bishop: Pattern Recognition and Machine Learning
- Hastie et al: Elements of Statistical Learning
- Goodfellow et al: Deep Learning

**Courses:**
- Stanford CS229 (ML)
- Stanford CS231n (Deep Learning/CV)
- Fast.ai (Practical Deep Learning)

**Practice:**
- Kaggle competitions
- Build projects
- Read papers and implement
