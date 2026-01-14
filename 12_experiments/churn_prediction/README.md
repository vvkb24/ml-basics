# Customer Churn Prediction: A Complete Case Study

A real-world tabular data case study with explicit decision checkpoints, failure modes, and scientific reasoning.

---

## Problem Framing

### Business Question
**"Which customers are likely to churn, and why?"**

### ML Translation
- **Task**: Binary classification (churn = 1, stay = 0)
- **Target Variable**: Customer churned within next 30 days
- **Unit of Prediction**: Individual customer
- **Actionable Outcome**: Prioritize retention interventions

### Why This Problem is Non-Trivial

| Challenge | Manifestation |
|-----------|---------------|
| Class imbalance | Typically 15-25% churn rate |
| Temporal leakage | Target depends on future behavior |
| Feature validity | Some features not available at prediction time |
| Business constraints | False positives have different costs than FN |

---

## Data Characteristics

### Dataset: Telco Customer Churn

**Source**: [Kaggle Telco Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)

**Size**: ~7,000 customers

**Features**:
- Demographics (gender, senior citizen, partner, dependents)
- Account info (tenure, contract type, payment method)
- Services (internet, phone, streaming, security)
- Charges (monthly, total)

**Target**: `Churn` (Yes/No)

### Data Quality Issues (Pre-discovered)

| Issue | Column | Impact |
|-------|--------|--------|
| Numeric as string | TotalCharges | Contains " " for zero tenure |
| Categorical encoding | Multiple columns | Yes/No needs to be 0/1 |
| Missing values | TotalCharges (11 rows) | Need imputation strategy |
| Multicollinearity | MonthlyCharges × Tenure ≈ TotalCharges | Redundancy |

---

## Case Study Highlights

This case study explicitly demonstrates:

### 1. Statistical Paradox: Simpson's Paradox

**Observation**: Month-to-month contracts have higher churn overall (45% vs 11%).

**But wait**: Within each tenure group, contract type effect is REVERSED for some segments!

```
Low tenure (0-12 months):
- Month-to-month: 50% churn
- 1-year contract: 15% churn

High tenure (>48 months):
- Month-to-month: 20% churn
- 1-year contract: 5% churn
```

The aggregate masks that tenure is the true driver.

### 2. Data Leakage Example

**The trap**: Using `TotalCharges` as a feature.

**Why it leaks**: 
$$\text{TotalCharges} = \text{MonthlyCharges} \times \text{Tenure}$$

If you're predicting "will this customer churn in the NEXT month", `TotalCharges` already includes past behavior up to today—which is fine. But if `Tenure` includes the churn period itself, that's leakage.

**Proper approach**: Use only features available at prediction time (snapshot date).

### 3. Metric Misuse Example

**The trap**: Optimizing for accuracy on imbalanced data.

```
Baseline: Predict all "No Churn"
Accuracy: 73% (but 0% recall on churners!)

Our model:
Accuracy: 78%
Precision: 65%
Recall: 55%
```

**Which is better?** Depends on business cost matrix:
- Cost of missing a churner (false negative) = $500 retention cost
- Cost of unnecessary intervention (false positive) = $50

### 4. Real-World Failure Scenario

**Scenario**: Model deployed, but churn rate changes from 26% to 35% in Q4.

**Cause**: Holiday promotions attract bargain hunters who churn after discounts end.

**Failure**: Model was trained on non-promotional data, doesn't generalize.

**Solution**: 
1. Monitor feature drift (average tenure of predictions)
2. Retrain quarterly with recent data
3. Add "acquired during promotion" feature

---

## File Structure

```
12_experiments/churn_prediction/
├── README.md              # This file
├── data/
│   └── telco_churn.csv    # Downloaded dataset
├── 01_eda.ipynb           # Exploratory analysis
├── 02_modeling.ipynb      # Model training and selection
└── 03_evaluation.ipynb    # Metrics and deployment considerations
```

---

## Decision Checkpoints

Throughout the notebooks, look for **DECISION CHECKPOINT** markers:

> **DECISION CHECKPOINT 1**: Why use a bar chart instead of pie chart for contract distribution?
> 
> - Pie charts are bad for comparing categories
> - Bar charts allow easy height comparison
> - We decided to use horizontal bars because contract names are long

> **DECISION CHECKPOINT 2**: Why not use accuracy as primary metric?
> 
> - Class imbalance (26% churn) makes accuracy misleading
> - Business cares more about catching churners (recall)
> - We chose F1 as compromise, with threshold tuned for recall ≥ 60%

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Download data (or get from Kaggle link above)
# Place telco_churn.csv in data/ folder

# Run notebooks in order:
# 1. 01_eda.ipynb
# 2. 02_modeling.ipynb  
# 3. 03_evaluation.ipynb
```

---

## Key Learnings Preview

1. **EDA revealed**: Tenure is the #1 predictor, not contract type
2. **Feature engineering**: Created "tenure_group" buckets improved interpretability
3. **Model choice**: Random Forest > Logistic Regression (non-linear interactions)
4. **Threshold tuning**: 0.35 instead of 0.5 to increase recall
5. **Business impact**: $2.3M annual savings from targeted retention

---

## References

- [Kaggle Competition](https://www.kaggle.com/blastchar/telco-customer-churn)
- ["Building Machine Learning Powered Applications"](https://www.oreilly.com/library/view/building-machine-learning/9781492045106/) - Emmanuel Ameisen
- [scikit-learn Imbalanced Classification](https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel)
