# Evaluation Metrics

Choosing the right metric is crucial for model assessment.

---

## Regression Metrics

### Mean Squared Error (MSE)

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

- Penalizes large errors more heavily
- In original units squared

### Root Mean Squared Error (RMSE)

$$\text{RMSE} = \sqrt{\text{MSE}}$$

- Same units as target
- More interpretable than MSE

### Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

- Robust to outliers
- All errors weighted equally

### R² Score (Coefficient of Determination)

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

- Range: $(-\infty, 1]$
- $R^2 = 1$: Perfect predictions
- $R^2 = 0$: Same as predicting mean
- $R^2 < 0$: Worse than mean

### Mean Absolute Percentage Error (MAPE)

$$\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

- Scale-independent
- Undefined when $y_i = 0$

---

## Classification Metrics

### Confusion Matrix

|  | Predicted + | Predicted - |
|---|---|---|
| **Actual +** | TP (True Positive) | FN (False Negative) |
| **Actual -** | FP (False Positive) | TN (True Negative) |

### Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Limitation:** Misleading with imbalanced classes.

### Precision

$$\text{Precision} = \frac{TP}{TP + FP}$$

Of predicted positives, how many are correct?

**High precision:** Few false positives (spam filter).

### Recall (Sensitivity, TPR)

$$\text{Recall} = \frac{TP}{TP + FN}$$

Of actual positives, how many did we catch?

**High recall:** Few false negatives (disease detection).

### F1 Score

Harmonic mean of precision and recall:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### F-beta Score

Generalization with tunable weight:

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

- $\beta < 1$: Emphasize precision
- $\beta > 1$: Emphasize recall

### Specificity (TNR)

$$\text{Specificity} = \frac{TN}{TN + FP}$$

---

## Threshold-Independent Metrics

### ROC Curve

Plot **True Positive Rate** vs. **False Positive Rate** at various thresholds.

![ROC Curve](https://scikit-learn.org/stable/_images/sphx_glr_plot_roc_001.png)
*Source: [scikit-learn documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)*

**ROC Curve Interpretation:**
- **X-axis**: False Positive Rate (FPR) from 0 to 1
- **Y-axis**: True Positive Rate (TPR) from 0 to 1
- **Diagonal line** (from (0,0) to (1,1)): Random classifier
- **Best point**: Top-left corner (0, 1) = perfect classifier
- **Curve above diagonal**: Better than random

### AUC-ROC

**Area Under ROC Curve**

- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random guessing
- AUC < 0.5: Worse than random

**Interpretation:** Probability that a randomly chosen positive is ranked higher than a randomly chosen negative.

### Precision-Recall Curve

Plot precision vs. recall at various thresholds.

**Better for imbalanced datasets** (focuses on positive class).

### Average Precision (AP)

Area under precision-recall curve.

---

## Multi-Class Metrics

### Macro Average

Average metric across classes:
$$\text{Macro-F1} = \frac{1}{K}\sum_{k=1}^{K} F_{1,k}$$

Treats all classes equally.

### Weighted Average

Weight by class frequency:
$$\text{Weighted-F1} = \sum_{k=1}^{K} \frac{n_k}{N} F_{1,k}$$

### Micro Average

Aggregate TP, FP, FN across classes, then compute metric.

---

## Probabilistic Metrics

### Log Loss (Cross-Entropy)

$$\text{LogLoss} = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)]$$

Penalizes confident wrong predictions heavily.

### Brier Score

$$\text{Brier} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{p}_i)^2$$

Mean squared error for probabilities.

---

## Ranking Metrics

### Mean Reciprocal Rank (MRR)

$$\text{MRR} = \frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{\text{rank}_i}$$

### Normalized Discounted Cumulative Gain (NDCG)

$$\text{DCG} = \sum_{i=1}^{k}\frac{2^{rel_i} - 1}{\log_2(i+1)}$$

$$\text{NDCG} = \frac{\text{DCG}}{\text{IDCG}}$$

---

## Choosing the Right Metric

| Scenario | Recommended Metric |
|----------|-------------------|
| Balanced classification | Accuracy, F1 |
| Imbalanced classification | F1, AUC-ROC, PR-AUC |
| Cost-sensitive | Custom weighted metric |
| Probabilistic output | Log loss, Brier score |
| Regression | RMSE, MAE, R² |
| Outlier-robust regression | MAE |
| Ranking | NDCG, MRR |

---

## Python Examples

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Regression
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# Classification
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)

# Detailed report
print(classification_report(y_true, y_pred))
```

---

## Key Takeaways

1. **Accuracy is not enough** for imbalanced data
2. **Precision/Recall tradeoff** depends on use case
3. **AUC-ROC** provides threshold-independent evaluation
4. **Choose metrics aligned with business goals**
5. **Report multiple metrics** for complete picture
