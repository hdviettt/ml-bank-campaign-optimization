# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a bank marketing campaign prediction project using machine learning classification models (Decision Tree and Logistic Regression) to optimize direct marketing campaigns by minimizing false positives (unnecessary calls) and false negatives (missed customers).

The project uses the Bank Marketing Dataset from UCI Machine Learning Repository containing 41,188 records from Portuguese banking telemarketing campaigns. The target variable predicts whether a customer subscribes to a term deposit product.

## Repository Structure

```
bankml/
├── input/           # Source data and reference materials
│   ├── 1-assignment-brief.pdf          # Project specification
│   ├── 2-assignment-template.md        # Assignment report template
│   ├── 3-example-google-colab.ipynb    # Main notebook with full ML pipeline
│   └── 4-data.csv                      # Bank marketing dataset (41K+ records)
└── output/          # Results, models, visualizations
```

## Data Pipeline Architecture

The notebook follows this workflow:

1. **Data Loading & EDA**: Load CSV, analyze 20 input features (10 categorical, 8 numerical, 2 macroeconomic)
   - Highly imbalanced: 11.3% positive class ("yes"), 88.7% negative
   - Key features: `nr.employed`, `cons.conf.idx`, `euribor3m`, `pdays`, `campaign`

2. **Data Preprocessing**:
   - Replace 'unknown' → NaN
   - Impute numerics with mean, categoricals with mode
   - Drop `duration` (data leakage)
   - Feature engineering: `was_contacted_before` (binary), `campaign_log`, `previous_log`
   - StandardScaler for numerics, OneHotEncoder for categoricals
   - Stratified 75/25 train-test split
   - Result: 59 features after encoding

3. **Modeling**:
   - Decision Tree (entropy criterion, `class_weight='balanced'`)
   - Logistic Regression (`class_weight='balanced'`)
   - GridSearchCV with StratifiedKFold for hyperparameter tuning
   - Metrics: accuracy, precision, recall, F1, ROC-AUC

4. **Cost-Sensitive Optimization**:
   - Custom cost matrix: FP=+1.5, FN=+20.0, TP=-5.0, TN=0.0
   - Threshold sweep (0.01–0.99) to minimize expected cost
   - Optimal thresholds: DT=0.35, LR=0.38

## Key Functions in Notebook

- `lump_rare(series, min_freq)` - Group infrequent categorical values
- `eval_model(y_true, y_pred, y_proba, name)` - Compute classification metrics
- `expected_cost(y_true, y_proba, threshold)` - Calculate campaign cost using custom matrix
- `sweep_thresholds(y_true, y_proba, name)` - Find optimal classification threshold
- `metrics_at(y_true, y_proba, th, name)` - Metrics at specific threshold
- `sweep_threshold_cost(y_true, y_proba, th_grid, ...)` - Comprehensive threshold analysis with visualization

## Common Development Commands

This project uses Jupyter notebooks. To work with the main analysis:

```bash
# Launch Jupyter (if working locally)
jupyter notebook input/3-example-google-colab.ipynb

# Or open in Google Colab (recommended, as indicated by filename)
```

## Important Implementation Details

**Class Imbalance Handling**:
- Always use `stratify=y` in train_test_split
- Set `class_weight='balanced'` in models
- Prioritize recall over accuracy for minority class detection

**GridSearchCV Configuration**:
- Decision Tree params: `max_depth`, `min_samples_leaf`, `ccp_alpha`, `class_weight`
- Logistic Regression params: `C`, `penalty`, `solver`, `class_weight`
- Use `StratifiedKFold` (cv=5) with `scoring='roc_auc'`

**Feature Engineering Rules**:
- `pdays=999` indicates "not previously contacted" → encode as binary flag
- Log-transform right-skewed features (`campaign`, `previous`)
- **Never use `duration`** in training (only known after call ends)

**Cost Matrix Business Logic**:
- Missing a potential customer (FN) is much more expensive than unnecessary call (FP)
- Threshold optimization is critical for real-world deployment
- Logistic Regression provides better calibrated probabilities for threshold tuning

## Dependencies

Standard data science stack:
- pandas, numpy - Data manipulation
- matplotlib, seaborn - Visualization
- scikit-learn - ML models, preprocessing, evaluation
  - `DecisionTreeClassifier`, `LogisticRegression`
  - `GridSearchCV`, `StratifiedKFold`
  - `StandardScaler`, `OneHotEncoder`, `SimpleImputer`
  - Metrics: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`, `RocCurveDisplay`
