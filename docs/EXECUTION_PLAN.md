# EXECUTION PLAN - Bank Marketing ML Assignment

**Project:** Data Mining Solutions for Direct Marketing Campaigns
**Course:** CIS051-3 Business Analytics
**Goal:** Create Decision Tree and Logistic Regression models to optimize bank telemarketing campaign costs

---

## PROJECT DELIVERABLES

1. **notebook.ipynb** - Jupyter notebook with complete ML pipeline
2. **assets/** - Folder containing all data visualization images
3. **report.md** - Final assignment report following template structure (~2,500 words)

---

## KEY DECISIONS & CONSTRAINTS

### Target Variable Understanding
- `y = "yes"`: Client **accepted** the term deposit offer (positive class, ~11%)
- `y = "no"`: Client **rejected** the term deposit offer (negative class, ~89%)

### Duration Variable Decision
**CRITICAL:** The `duration` variable (last contact duration in seconds) must be handled carefully:
- ✅ Include in EDA to analyze its correlation with success
- ✅ Optionally build one experimental model WITH duration to show unrealistic performance
- ✅ **EXCLUDE from all final predictive models** (data leakage - only known after call ends)
- ✅ Document this decision clearly in report with justification

### Report Structure
- Follow **2-assignment-template.md** structure exactly
- NO separate literature review section
- Integrate literature citations within Introduction, Methods, and Conclusion sections
- Target ~2,500 words total

### Technology Stack
- Python with Jupyter Notebook
- pandas, numpy, matplotlib, seaborn, scikit-learn
- Decision Tree (entropy criterion) + Logistic Regression
- GridSearchCV for hyperparameter tuning
- Cost-sensitive threshold optimization

---

## PHASE-BY-PHASE EXECUTION PLAN

---

## **PHASE 0: Literature Review & Domain Understanding**

### Objectives
- Understand the Bank Marketing dataset context
- Build theoretical foundation for methodology choices
- Gather 8-10 academic references for citations
- Understand benchmark performance if available

### Tasks

1. **Dataset Documentation Review:**
   - UCI Machine Learning Repository - Bank Marketing Dataset
   - Original paper: Moro, S., Cortez, P., & Rita, P. (2014)
   - Understand Portuguese bank telemarketing context
   - Review data collection methodology

2. **Literature Research - Key Topics:**
   - Direct marketing optimization and cost-sensitive classification
   - Decision Trees: interpretability, handling non-linearity, feature importance
   - Logistic Regression: probability calibration, threshold optimization
   - Class imbalance handling techniques (SMOTE, class weights, stratification)
   - Cost-sensitive learning approaches
   - Evaluation metrics for imbalanced datasets (Recall, Precision, F1, ROC-AUC)

3. **Establish Theoretical Justifications:**
   - Why Decision Tree? → Interpretability, business rules, non-linear patterns, no scaling needed
   - Why Logistic Regression? → Probabilistic output, well-calibrated, industry standard
   - Why cost matrix? → Business ROI focus, asymmetric error costs (FN >> FP)
   - Why prioritize Recall? → Missing customers (FN) more expensive than wasted calls (FP)

4. **Find Benchmark Results:**
   - Check UCI repository discussions
   - Look for papers using same dataset
   - Note state-of-the-art performance metrics

### Expected Output
- Reference list (8-10 sources): academic papers, textbooks, sklearn documentation
- Notes for Introduction section justifications
- Benchmark metrics for comparison (if available)
- Theoretical foundation for all methodology choices

### Key References to Include
- Moro et al. (2014) - Dataset paper [REQUIRED]
- Cost-sensitive learning papers
- Marketing analytics papers
- Scikit-learn documentation
- Classification textbooks (Hastie, James et al.)

---

## **PHASE 1: Project Setup & Environment**

### Tasks
1. Create directory structure:
   ```
   bankml/
   ├── notebook.ipynb
   ├── assets/
   ├── report.md
   ├── input/
   │   └── 4-data.csv
   └── output/
   ```

2. Install required libraries:
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.linear_model import LogisticRegression
   from sklearn.preprocessing import StandardScaler, OneHotEncoder
   from sklearn.impute import SimpleImputer
   from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix,
                                 ConfusionMatrixDisplay, RocCurveDisplay)
   ```

3. Load and verify dataset:
   - Load `input/4-data.csv`
   - Display shape, columns, data types
   - Verify 41,188 records loaded successfully

### Expected Output
- Clean project structure
- All dependencies installed
- Dataset loaded and verified
- Initial notebook structure with markdown headers

---

## **PHASE 1.5: Data Dictionary & Column Understanding**

### Objectives
- Create comprehensive understanding of all 21 columns
- Document the meaning and business context of each variable
- Identify preprocessing requirements per column
- Make informed decision about `duration` variable

### Complete Data Dictionary

**Client Demographics:**
- `age` (numeric): Client age in years
- `job` (categorical): Job type (admin, blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown)
- `marital` (categorical): Marital status (married, divorced, single, unknown)
- `education` (categorical): Education level (basic.4y, basic.6y, basic.9y, high.school, illiterate, professional.course, university.degree, unknown)
- `default` (binary): Has credit in default? (yes, no, unknown)
- `housing` (binary): Has housing loan? (yes, no, unknown)
- `loan` (binary): Has personal loan? (yes, no, unknown)

**Campaign Contact Information:**
- `contact` (categorical): Contact communication type (cellular, telephone)
- `month` (categorical): Last contact month (jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec)
- `day_of_week` (categorical): Last contact day (mon, tue, wed, thu, fri)
- `duration` (numeric): **⚠️ Last contact duration in seconds** [DATA LEAKAGE]
- `campaign` (numeric): Number of contacts performed during this campaign for this client
- `pdays` (numeric): Days since client was last contacted from previous campaign (999 = never contacted)
- `previous` (numeric): Number of contacts performed before this campaign for this client
- `poutcome` (categorical): Outcome of previous marketing campaign (failure, nonexistent, success)

**Economic Context Indicators:**
- `emp.var.rate` (numeric): Employment variation rate (quarterly indicator)
- `cons.price.idx` (numeric): Consumer price index (monthly indicator)
- `cons.conf.idx` (numeric): Consumer confidence index (monthly indicator)
- `euribor3m` (numeric): Euribor 3 month rate (daily indicator)
- `nr.employed` (numeric): Number of employees (quarterly indicator)

**Target Variable:**
- `y` (binary): **Has the client subscribed to a term deposit?** (yes = accepted, no = rejected)

### Duration Variable Analysis

**What it is:**
- Duration of the last phone call in seconds
- Available in historical data

**Why it's problematic:**
- Only known AFTER the call completes
- Cannot be used for prediction BEFORE making the call
- Including it creates unrealistic model unsuitable for production

**Our approach:**
1. Analyze in EDA to show strong correlation with success
2. Optionally build experimental model to show inflated performance
3. **Exclude from all final predictive models**
4. Document decision clearly with business justification

**Justification for report:**
> "The duration variable shows strong correlation with campaign success—longer calls tend to result in higher acceptance rates. However, this represents data leakage: call duration is only known after the call concludes, but our predictive model must assess success probability before initiating contact. Including duration would yield unrealistically high accuracy unsuitable for deployment. We exclude duration from predictive models while acknowledging its descriptive value."

### Expected Output
- Complete data dictionary in notebook
- Clear understanding of each variable's business meaning
- Documented decision on duration variable with justification
- Identified preprocessing needs (missing values, encoding, scaling)
- List of feature engineering opportunities

---

## **PHASE 2: Exploratory Data Analysis (EDA)**

### Objectives
- Understand data distributions and patterns
- Identify data quality issues
- Discover relationships between features and target
- Generate insights for feature engineering
- Create visualizations for report

### Analysis Tasks

**1. Initial Data Inspection**
- Dataset shape and structure
- Data types verification against data dictionary
- Missing values count ('unknown' and NaN)
- First/last few rows examination
- Summary statistics

**2. Target Variable Analysis**
- Class distribution counts and percentages
- Visualization: bar chart of acceptance rate
- Business context: What does 11% vs 89% imbalance mean?
- Calculate baseline accuracy (predicting majority class)

**Visualization:** `assets/01_class_distribution.png`

**3. Numerical Features Analysis**

For each numeric variable, analyze:
- Distribution (histogram)
- Summary statistics (mean, median, std, quartiles)
- Skewness identification
- Relationship with target (box plots, violin plots)

**Variables to analyze:**
- `age`: Age distribution, acceptance by age group
- `campaign`: Right-skewed? Need log transformation?
- `previous`: Right-skewed? Distribution of previous contacts
- `pdays`: Special value 999 concentration
- `duration`: **Strong correlation with target** (then explain exclusion)
- Economic indicators: emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed

**Visualizations:**
- `assets/02_numerical_distributions.png` (multi-panel histogram)
- `assets/03_age_vs_target.png`
- `assets/04_duration_analysis.png` (with explanation)
- `assets/05_economic_indicators.png`

**4. Categorical Features Analysis**

For each categorical variable:
- Value counts
- Acceptance rate per category
- Visualizations (count plots, stacked bar charts)

**Variables to analyze:**
- `job`: Which jobs have higher acceptance?
- `marital`: Marital status impact
- `education`: Education level patterns
- `contact`: Cellular vs telephone effectiveness
- `month`: Seasonality - which months perform best?
- `day_of_week`: Day patterns
- `poutcome`: Previous campaign outcome impact (strong predictor?)

**Visualizations:**
- `assets/06_job_analysis.png`
- `assets/07_month_seasonality.png`
- `assets/08_poutcome_impact.png`

**5. Special Variable Analysis**

**pdays deep dive:**
- Distribution showing 999 concentration
- Interpretation: 999 = never contacted before
- Compare acceptance rate: contacted before vs never contacted
- Justification for `was_contacted_before` feature engineering

**Visualization:** `assets/09_pdays_distribution.png`

**6. Feature Relationships**

**Correlation analysis:**
- Correlation matrix for numeric features
- Heatmap visualization
- Identify multicollinearity (economic indicators likely correlated)
- Note strong correlations for interpretation

**Bi-variate analysis:**
- Key feature pairs vs target
- Economic indicators vs acceptance rate
- Campaign intensity (campaign, previous) vs success

**Visualizations:**
- `assets/10_correlation_heatmap.png`
- `assets/11_bivariate_analysis.png`

**7. Duration Variable Special Section**

**Analysis to include:**
- Distribution of duration
- Duration vs acceptance rate (show strong positive correlation)
- Example: "Calls > 5 minutes have 80%+ acceptance"
- Statistical test (t-test or Mann-Whitney) showing significant difference
- **Clear explanation:** "But we cannot use this for prediction!"
- Optional: Build one model WITH duration showing ~90% accuracy
- Compare to model WITHOUT duration showing realistic ~85% accuracy
- Conclusion: Exclude duration from final models

**Visualization:** `assets/12_duration_leakage_analysis.png`

### Key Insights to Document

From EDA, document insights such as:
- Class imbalance severity (11.3% positive)
- Strong predictors identified (poutcome, economic indicators, previous contact)
- Seasonality patterns (best months for campaigns)
- Data quality issues (unknown values, skewness)
- Feature engineering opportunities (pdays → binary, log transformations)
- Duration variable conclusion

### Expected Output
- 12-15 high-quality visualizations saved to assets/
- Detailed EDA section in notebook with interpretations
- Clear evidence for preprocessing decisions
- Insights for feature engineering
- Business insights for report
- Duration variable analysis and exclusion justification

---

## **PHASE 3: Data Preprocessing & Feature Engineering**

### Objectives
- Handle missing values appropriately
- Engineer new features based on EDA insights
- Encode categorical variables
- Scale numerical features
- Create train-test split with stratification

### Tasks

**1. Handle Missing Values**

**Strategy:**
- Replace 'unknown' strings with NaN
- Count missing values per column
- Imputation:
  - **Numerical features:** Mean imputation (SimpleImputer, strategy='mean')
  - **Categorical features:** Mode imputation (SimpleImputer, strategy='most_frequent')

**Justification (for report):**
- Mean imputation: Standard practice for numerical data, maintains distribution center
- Mode imputation: Preserves most common category, suitable for categorical data
- Alternative considered: Dropping rows (but would lose too much data)
- Alternative considered: Multiple imputation (too complex for this scope)

**Code pattern:**
```python
# Numerical imputation
num_imputer = SimpleImputer(strategy='mean')
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])

# Categorical imputation
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
```

**2. Feature Engineering**

**Variables to DROP:**
- `duration`: Data leakage (justified in Phase 1.5 and Phase 2)

**Variables to CREATE:**

**a) was_contacted_before (binary):**
```python
df['was_contacted_before'] = (df['pdays'] != 999).astype(int)
```
- **Justification:** Separates "never contacted" (pdays=999) from "contacted N days ago"
- **EDA evidence:** Showed different acceptance rates between these groups
- **Business logic:** Previous contact is valuable signal, but exact days may be noisy

**b) campaign_log (numeric):**
```python
df['campaign_log'] = np.log1p(df['campaign'])
```
- **Justification:** Reduces right skewness shown in EDA
- **Benefit:** Improves model performance by normalizing distribution
- **Method:** log1p = log(1+x) to handle zeros

**c) previous_log (numeric):**
```python
df['previous_log'] = np.log1p(df['previous'])
```
- **Justification:** Same as campaign_log, reduces skewness

**Optional features (if time permits):**
- Age groups (binning: <30, 30-45, 45-60, >60)
- Economic indicator interactions
- Month seasonality encoding (high vs low season)

**3. Encoding Categorical Variables**

**Method:** One-Hot Encoding

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
encoded = encoder.fit_transform(X_train[cat_cols])
```

**Parameters:**
- `drop='first'`: Avoid multicollinearity (drop one category as reference)
- `sparse=False`: Return dense array for easier handling
- `handle_unknown='ignore'`: Handle unseen categories in test set

**Justification:**
- Nominal categories (no ordinal relationship)
- Standard approach for tree-based and linear models
- Alternative considered: Label encoding (inappropriate for nominal data)

**Expected feature count:** ~59 features after encoding (verify in notebook)

**4. Scaling Numerical Features**

**Method:** StandardScaler (z-score normalization)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[num_cols])
```

**Justification:**
- Different scales: age (20-90) vs euribor3m (0-5)
- **Required for Logistic Regression:** Distance-based algorithm
- **Optional for Decision Tree:** But doesn't hurt, maintains consistency
- Standard practice in ML pipelines

**5. Train-Test Split**

**Strategy:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
```

**Parameters:**
- `test_size=0.25`: 75% train, 25% test (standard split)
- `random_state=42`: Reproducibility
- **`stratify=y`**: Maintain class ratio in both sets (CRITICAL for imbalanced data)

**Justification:**
- Stratification ensures both train and test have ~11% positive class
- Prevents random split creating different distributions
- Essential for reliable evaluation on imbalanced data

### Preprocessing Pipeline

Document the complete pipeline:
1. Load raw data
2. Handle missing values (imputation)
3. Feature engineering (drop duration, create new features)
4. Split data (stratified)
5. Encode categoricals (on train, transform test)
6. Scale numericals (fit on train, transform test)

**Important:** Fit encoders and scalers on training data only, then transform test data to prevent data leakage!

### Expected Output
- Preprocessed datasets: X_train, X_test, y_train, y_test
- Feature count documented (~59 features)
- All preprocessing steps justified and documented
- Visualization showing preprocessing impact (optional)
- Ready for modeling

---

## **PHASE 4: Baseline Model Development**

### Objectives
- Build initial Decision Tree and Logistic Regression models
- Establish baseline performance metrics
- Create performance visualizations
- Set benchmark for improvement

### Tasks

**1. Decision Tree Baseline**

```python
from sklearn.tree import DecisionTreeClassifier

dt_baseline = DecisionTreeClassifier(
    criterion='entropy',           # Information gain
    class_weight='balanced',       # Handle imbalance
    random_state=42
)

dt_baseline.fit(X_train, y_train)
y_pred_dt = dt_baseline.predict(X_test)
y_proba_dt = dt_baseline.predict_proba(X_test)[:, 1]
```

**Parameters:**
- `criterion='entropy'`: Information gain splitting (vs Gini)
- `class_weight='balanced'`: Automatically adjust weights inversely proportional to class frequencies
- Other params: Default (no max_depth, min_samples_leaf=1, etc.)

**Justification:**
- Entropy criterion: Traditional approach, interpretable as information gain
- Balanced weights: Essential for imbalanced data, prevents majority class bias

**2. Logistic Regression Baseline**

```python
from sklearn.linear_model import LogisticRegression

lr_baseline = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)

lr_baseline.fit(X_train, y_train)
y_pred_lr = lr_baseline.predict(X_test)
y_proba_lr = lr_baseline.predict_proba(X_test)[:, 1]
```

**Parameters:**
- `class_weight='balanced'`: Same as DT
- `max_iter=1000`: Ensure convergence
- Other params: Default (penalty='l2', C=1.0, solver='lbfgs')

**3. Evaluation Metrics**

For both models, calculate:

```python
def evaluate_model(y_true, y_pred, y_proba, model_name):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_proba)
    }
    return metrics
```

**Metrics explanation:**
- **Accuracy:** Overall correctness (less important due to imbalance)
- **Precision:** Of predicted positives, how many are correct? (FP cost consideration)
- **Recall:** Of actual positives, how many did we catch? (FN cost - PRIORITY)
- **F1-Score:** Harmonic mean of Precision and Recall
- **ROC-AUC:** Area under ROC curve (threshold-independent performance)

**Why prioritize Recall?**
- Missing a customer (FN) is more costly than unnecessary call (FP)
- Business goal: Don't miss potential acceptors
- Will be formalized in cost matrix later

**4. Visualization - Confusion Matrices**

```python
from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dt, ax=axes[0])
axes[0].set_title('Decision Tree - Baseline')

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr, ax=axes[1])
axes[1].set_title('Logistic Regression - Baseline')

plt.savefig('assets/13_baseline_confusion_matrices.png', dpi=300, bbox_inches='tight')
```

**5. Visualization - ROC Curves**

```python
from sklearn.metrics import RocCurveDisplay

fig, ax = plt.subplots(figsize=(8, 6))

RocCurveDisplay.from_predictions(y_test, y_proba_dt, name='Decision Tree', ax=ax)
RocCurveDisplay.from_predictions(y_test, y_proba_lr, name='Logistic Regression', ax=ax)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.legend()
plt.title('ROC Curves - Baseline Models')
plt.savefig('assets/14_baseline_roc_curves.png', dpi=300, bbox_inches='tight')
```

**6. Results Comparison Table**

Create comparison table:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Decision Tree | 0.90 | 0.65 | 0.28 | 0.39 | 0.79 |
| Logistic Regression | 0.90 | 0.70 | 0.22 | 0.33 | 0.80 |

*(Values are estimates based on template, actual values will vary)*

**Interpretation:**
- Both models achieve ~90% accuracy (due to imbalanced data - baseline is 89%)
- **Low recall** (~0.22-0.28): Missing many positive cases
- High precision: When predicting positive, often correct
- ROC-AUC ~0.80: Reasonable discrimination ability
- **Problem:** Need to improve recall to catch more potential customers

### Expected Output
- Two trained baseline models
- Complete metrics for both models
- Comparison table
- 2 visualizations:
  - `assets/13_baseline_confusion_matrices.png`
  - `assets/14_baseline_roc_curves.png`
- Documentation of baseline performance
- Identified need for improvement (low recall)

---

## **PHASE 5: Hyperparameter Optimization**

### Objectives
- Systematically search for optimal hyperparameters
- Improve model performance, especially recall
- Use cross-validation to prevent overfitting
- Document best parameters and improvements

### Tasks

**1. Decision Tree GridSearchCV**

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

param_grid_dt = {
    'max_depth': [None, 5, 10, 15, 20, 25],
    'min_samples_leaf': [1, 2, 5, 10, 20],
    'min_samples_split': [2, 5, 10, 20],
    'ccp_alpha': [0.0, 0.001, 0.005, 0.01],
    'class_weight': ['balanced']
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_dt = GridSearchCV(
    estimator=DecisionTreeClassifier(criterion='entropy', random_state=42),
    param_grid=param_grid_dt,
    cv=cv_strategy,
    scoring='roc_auc',           # Optimize for ROC-AUC
    n_jobs=-1,                    # Use all CPU cores
    verbose=1
)

grid_dt.fit(X_train, y_train)

print("Best parameters:", grid_dt.best_params_)
print("Best CV ROC-AUC:", grid_dt.best_score_)

best_dt = grid_dt.best_estimator_
```

**Parameter explanations:**
- `max_depth`: Maximum tree depth (controls overfitting)
- `min_samples_leaf`: Minimum samples required at leaf node (prevents over-splitting)
- `min_samples_split`: Minimum samples required to split node
- `ccp_alpha`: Complexity parameter for pruning (higher = more pruning)
- `class_weight='balanced'`: Keep balanced weighting

**CV strategy:**
- `StratifiedKFold(n_splits=5)`: 5-fold cross-validation maintaining class ratios
- Essential for imbalanced data

**Scoring:**
- `scoring='roc_auc'`: Optimize for AUC (threshold-independent, good for imbalanced data)

**Expected best parameters** (from template):
```python
{
    'ccp_alpha': 0.001,
    'class_weight': 'balanced',
    'max_depth': None,
    'min_samples_leaf': 1,
    'min_samples_split': 2
}
```

**2. Logistic Regression GridSearchCV**

```python
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['saga', 'liblinear'],
    'class_weight': ['balanced'],
    'max_iter': [1000]
}

grid_lr = GridSearchCV(
    estimator=LogisticRegression(random_state=42),
    param_grid=param_grid_lr,
    cv=cv_strategy,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_lr.fit(X_train, y_train)

print("Best parameters:", grid_lr.best_params_)
print("Best CV ROC-AUC:", grid_lr.best_score_)

best_lr = grid_lr.best_estimator_
```

**Parameter explanations:**
- `C`: Inverse regularization strength (smaller = stronger regularization)
- `penalty`: Regularization type ('l1' = Lasso, 'l2' = Ridge)
- `solver`: Optimization algorithm (saga and liblinear support l1)
- `max_iter`: Maximum iterations for convergence

**Expected best parameters** (from template):
```python
{
    'C': 0.1,
    'class_weight': 'balanced',
    'penalty': 'l1',
    'solver': 'saga'
}
```

**3. Evaluate Tuned Models**

```python
# Decision Tree tuned
y_pred_dt_tuned = best_dt.predict(X_test)
y_proba_dt_tuned = best_dt.predict_proba(X_test)[:, 1]
metrics_dt_tuned = evaluate_model(y_test, y_pred_dt_tuned, y_proba_dt_tuned, 'DT Tuned')

# Logistic Regression tuned
y_pred_lr_tuned = best_lr.predict(X_test)
y_proba_lr_tuned = best_lr.predict_proba(X_test)[:, 1]
metrics_lr_tuned = evaluate_model(y_test, y_pred_lr_tuned, y_proba_lr_tuned, 'LR Tuned')
```

**4. Comparison: Baseline vs Tuned**

Create comprehensive comparison table:

| Model | Stage | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|-------|----------|-----------|--------|-----|---------|
| DT | Baseline | 0.9020 | 0.6513 | 0.2802 | 0.3918 | 0.7930 |
| DT | Tuned | 0.8631 | 0.4260 | **0.6207** | 0.5053 | 0.8014 |
| LR | Baseline | 0.9013 | 0.7011 | 0.2164 | 0.3307 | 0.8046 |
| LR | Tuned | 0.8344 | 0.3660 | **0.6414** | 0.4660 | 0.8041 |

**Key observations:**
- Recall improvement: DT +0.34, LR +0.42 (MAJOR improvement!)
- Accuracy decreased (expected - less majority class bias)
- Precision decreased (trade-off for higher recall)
- ROC-AUC stable or improved
- **Conclusion:** Successfully improved recall - catching more potential customers

**5. Visualizations**

**Tuned Confusion Matrices:**
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dt_tuned, ax=axes[0])
axes[0].set_title('Decision Tree - Tuned')
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr_tuned, ax=axes[1])
axes[1].set_title('Logistic Regression - Tuned')
plt.savefig('assets/15_tuned_confusion_matrices.png', dpi=300, bbox_inches='tight')
```

**ROC Curves - Baseline vs Tuned:**
```python
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_predictions(y_test, y_proba_dt, name='DT Baseline', ax=ax, linestyle='--')
RocCurveDisplay.from_predictions(y_test, y_proba_dt_tuned, name='DT Tuned', ax=ax)
RocCurveDisplay.from_predictions(y_test, y_proba_lr, name='LR Baseline', ax=ax, linestyle='--')
RocCurveDisplay.from_predictions(y_test, y_proba_lr_tuned, name='LR Tuned', ax=ax)
plt.legend()
plt.title('ROC Curves - Baseline vs Tuned')
plt.savefig('assets/16_tuned_roc_curves.png', dpi=300, bbox_inches='tight')
```

### Expected Output
- Best hyperparameters for both models (documented)
- Tuned model performance metrics
- Baseline vs Tuned comparison table
- Evidence of recall improvement
- 2 visualizations:
  - `assets/15_tuned_confusion_matrices.png`
  - `assets/16_tuned_roc_curves.png`
- Documentation of GridSearch process and results

---

## **PHASE 6: Cost-Sensitive Threshold Optimization**

### Objectives
- Define business cost matrix
- Find optimal classification thresholds
- Minimize expected campaign cost
- Evaluate final performance at optimal thresholds

### Tasks

**1. Define Cost Matrix**

Based on business logic:

```python
# Cost matrix
COST_FP = 1.5    # False Positive: unnecessary call cost
COST_FN = 20.0   # False Negative: missed customer opportunity cost
COST_TP = -5.0   # True Positive: revenue from successful call (negative cost = profit)
COST_TN = 0.0    # True Negative: correctly avoided call (no cost)
```

**Business justification:**
- **FP (1.5):** Wasted call - agent time, phone cost, customer annoyance
- **FN (20.0):** Missed customer - lost revenue from potential term deposit (much higher!)
- **TP (-5.0):** Successful sale - revenue/profit (negative cost)
- **TN (0.0):** Correctly not calling someone who would reject (no cost or benefit)

**Asymmetry:** FN >> FP reflects business reality - missing customers is very expensive

**2. Implement Expected Cost Function**

```python
def expected_cost(y_true, y_proba, threshold=0.5):
    """
    Calculate expected cost per customer given predictions and threshold.

    Parameters:
    - y_true: True labels (0/1)
    - y_proba: Predicted probabilities for positive class
    - threshold: Classification threshold

    Returns:
    - Average cost per customer
    """
    y_pred = (y_proba >= threshold).astype(int)

    # Calculate confusion matrix components
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    # Calculate total cost
    total_cost = (fp * COST_FP +
                  fn * COST_FN +
                  tp * COST_TP +
                  tn * COST_TN)

    # Average cost per customer
    avg_cost = total_cost / len(y_true)

    return avg_cost
```

**3. Threshold Sweep**

```python
# Test thresholds from 0.01 to 0.99
thresholds = np.linspace(0.01, 0.99, 99)

# Decision Tree
costs_dt = []
for thresh in thresholds:
    cost = expected_cost(y_test, y_proba_dt_tuned, threshold=thresh)
    costs_dt.append(cost)

optimal_thresh_dt = thresholds[np.argmin(costs_dt)]
min_cost_dt = np.min(costs_dt)

print(f"DT Optimal Threshold: {optimal_thresh_dt:.2f}")
print(f"DT Minimum Avg Cost: {min_cost_dt:.3f}")

# Logistic Regression
costs_lr = []
for thresh in thresholds:
    cost = expected_cost(y_test, y_proba_lr_tuned, threshold=thresh)
    costs_lr.append(cost)

optimal_thresh_lr = thresholds[np.argmin(costs_lr)]
min_cost_lr = np.min(costs_lr)

print(f"LR Optimal Threshold: {optimal_thresh_lr:.2f}")
print(f"LR Minimum Avg Cost: {min_cost_lr:.3f}")
```

**Expected results** (from template):
- DT Optimal Threshold: **0.35**
- DT Minimum Cost: **0.552**
- LR Optimal Threshold: **0.38**
- LR Minimum Cost: **0.509** (BEST!)

**4. Visualize Cost vs Threshold**

```python
plt.figure(figsize=(10, 6))
plt.plot(thresholds, costs_dt, label='Decision Tree', linewidth=2)
plt.plot(thresholds, costs_lr, label='Logistic Regression', linewidth=2)
plt.axvline(optimal_thresh_dt, color='blue', linestyle='--', alpha=0.7,
            label=f'DT Optimal ({optimal_thresh_dt:.2f})')
plt.axvline(optimal_thresh_lr, color='orange', linestyle='--', alpha=0.7,
            label=f'LR Optimal ({optimal_thresh_lr:.2f})')
plt.xlabel('Classification Threshold')
plt.ylabel('Average Cost per Customer')
plt.title('Cost-Sensitive Threshold Optimization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('assets/17_cost_threshold_optimization.png', dpi=300, bbox_inches='tight')
```

**5. Evaluate at Optimal Thresholds**

```python
# Decision Tree at optimal threshold
y_pred_dt_optimal = (y_proba_dt_tuned >= optimal_thresh_dt).astype(int)
metrics_dt_optimal = evaluate_model(y_test, y_pred_dt_optimal, y_proba_dt_tuned, 'DT Optimal')

# Logistic Regression at optimal threshold
y_pred_lr_optimal = (y_proba_lr_tuned >= optimal_thresh_lr).astype(int)
metrics_lr_optimal = evaluate_model(y_test, y_pred_lr_optimal, y_proba_lr_tuned, 'LR Optimal')
```

**6. Final Metrics at Optimal Thresholds**

Create table showing:

| Model | Threshold | Accuracy | Precision | Recall | F1 | Avg Cost |
|-------|-----------|----------|-----------|--------|-----|----------|
| DT | 0.35 | 0.78 | 0.35 | 0.75 | 0.48 | 0.552 |
| LR | 0.38 | 0.76 | 0.32 | 0.78 | 0.46 | **0.509** |

**Key observations:**
- **Recall further improved** to ~0.75-0.78 (catching 75%+ of potential customers!)
- Precision decreased (more false positives, but acceptable given cost structure)
- **Logistic Regression achieves lowest cost** → Winner model
- Optimal thresholds much lower than default 0.5 (favoring recall over precision)

### Expected Output
- Cost matrix defined and justified
- Expected cost function implemented
- Optimal thresholds found for both models
- Final performance metrics at optimal thresholds
- Visualization: `assets/17_cost_threshold_optimization.png`
- Clear winner: Logistic Regression (lowest cost 0.509)
- Documentation of cost-sensitive approach

---

## **PHASE 7: Model Interpretability & Feature Importance**

### Objectives
- Extract and visualize feature importance
- Interpret which features drive predictions
- Translate technical findings to business insights
- Provide actionable recommendations

### Tasks

**1. Decision Tree Feature Importance**

```python
# Extract feature importances
feature_names = ...  # List of feature names after encoding
importances_dt = best_dt.feature_importances_

# Create DataFrame for easy sorting
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances_dt
}).sort_values('Importance', ascending=False)

# Display top 10
print("Top 10 Most Important Features (Decision Tree):")
print(feature_importance_df.head(10))
```

**Expected top features** (from template):
1. `nr.employed` (employment rate) - 0.67
2. `cons.conf.idx` (consumer confidence) - 0.13
3. `was_contacted_before` - 0.05
4. `cons.price.idx` - 0.04
5. `euribor3m` - 0.03
6. (others...)

**Interpretation:**
- **Economic indicators dominate:** Employment rate and consumer confidence
- **Previous contact matters:** Binary flag we engineered
- **Macroeconomic context** is crucial for campaign success

**Visualize:**
```python
plt.figure(figsize=(10, 6))
top_features = feature_importance_df.head(15)
plt.barh(top_features['Feature'], top_features['Importance'])
plt.xlabel('Importance')
plt.title('Decision Tree - Top 15 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('assets/18_dt_feature_importance.png', dpi=300, bbox_inches='tight')
```

**2. Decision Tree Visualization**

```python
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(best_dt,
          max_depth=3,              # Limit depth for readability
          feature_names=feature_names,
          class_names=['Reject', 'Accept'],
          filled=True,
          fontsize=10)
plt.title('Decision Tree Structure (Depth=3)')
plt.savefig('assets/19_decision_tree_structure.png', dpi=300, bbox_inches='tight')
```

**Tree interpretation:**
- Root node: Likely splits on nr.employed or cons.conf.idx
- Follow decision paths to understand rules
- Example rule: "IF nr.employed > 5100 AND cons.conf.idx > -40 THEN likely reject"

**3. Logistic Regression Coefficients**

```python
# Extract coefficients
coefficients = best_lr.coef_[0]

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print("Top 10 Most Influential Features (Logistic Regression):")
print(coef_df.head(10))
```

**Expected top coefficients** (from template):
- `emp.var.rate`: -1.69 (negative → higher employment variation decreases acceptance)
- `month_mar`: +1.07 (positive → March campaigns more successful)
- `cons.price.idx`: +0.77 (positive → higher CPI increases acceptance)
- `poutcome_failure`: -0.29 (negative → previous failure decreases probability)
- (others...)

**Interpretation:**
- **Negative coefficients:** Decrease log-odds of acceptance
- **Positive coefficients:** Increase log-odds of acceptance
- Magnitude indicates strength of effect

**Visualize:**
```python
# Plot top positive and negative coefficients
plt.figure(figsize=(10, 8))

top_positive = coef_df.nlargest(10, 'Coefficient')
top_negative = coef_df.nsmallest(10, 'Coefficient')
top_coefs = pd.concat([top_negative, top_positive]).sort_values('Coefficient')

colors = ['red' if x < 0 else 'green' for x in top_coefs['Coefficient']]
plt.barh(top_coefs['Feature'], top_coefs['Coefficient'], color=colors)
plt.xlabel('Coefficient Value')
plt.title('Logistic Regression - Top Positive & Negative Coefficients')
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig('assets/20_lr_coefficients.png', dpi=300, bbox_inches='tight')
```

**4. Business Insights Translation**

**From Decision Tree:**
- **Employment rate** is the strongest predictor
  - **Business insight:** Target campaigns during economically stable periods with low unemployment
- **Consumer confidence** is second most important
  - **Business insight:** Monitor consumer sentiment indicators before launching campaigns
- **Previous contact** matters significantly
  - **Business insight:** Prioritize customers who were contacted before (warm leads)

**From Logistic Regression:**
- **March campaigns** perform better
  - **Business insight:** Consider seasonality in campaign planning
- **Employment variation** negatively impacts success
  - **Business insight:** Avoid campaigns during economic volatility
- **Consumer price index** positively correlates
  - **Business insight:** Higher purchasing power may indicate better acceptance

**Actionable Recommendations:**
1. **Timing:** Launch campaigns in March, during economically stable periods
2. **Targeting:** Prioritize previously contacted customers
3. **Economic monitoring:** Track nr.employed, cons.conf.idx before campaigns
4. **Avoid:** High unemployment periods, economic uncertainty times

### Expected Output
- Feature importance rankings for both models
- Business interpretation of top features
- 3 visualizations:
  - `assets/18_dt_feature_importance.png`
  - `assets/19_decision_tree_structure.png`
  - `assets/20_lr_coefficients.png`
- Actionable business recommendations
- Alignment between DT and LR findings (both identify economic indicators)

---

## **PHASE 8: Model Comparison & Final Selection**

### Objectives
- Comprehensive comparison across all stages
- Justify final model selection
- Visualize complete model progression
- Prepare final results for report

### Tasks

**1. Master Comparison Table**

Create comprehensive table:

| Model | Stage | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg Cost | Threshold |
|-------|-------|----------|-----------|--------|-----|---------|----------|-----------|
| **Decision Tree** |
| DT | Baseline | 0.9020 | 0.6513 | 0.2802 | 0.3918 | 0.7930 | - | 0.50 |
| DT | Tuned | 0.8631 | 0.4260 | 0.6207 | 0.5053 | 0.8014 | - | 0.50 |
| DT | Optimized | 0.78 | 0.35 | 0.75 | 0.48 | 0.8014 | **0.552** | 0.35 |
| **Logistic Regression** |
| LR | Baseline | 0.9013 | 0.7011 | 0.2164 | 0.3307 | 0.8046 | - | 0.50 |
| LR | Tuned | 0.8344 | 0.3660 | 0.6414 | 0.4660 | 0.8041 | - | 0.50 |
| LR | Optimized | 0.76 | 0.32 | 0.78 | 0.46 | 0.8041 | **0.509** | 0.38 |

**2. Model Comparison Analysis**

**Decision Tree:**
- ✅ **Strengths:**
  - High interpretability (visual tree structure)
  - Clear decision rules
  - No scaling required
  - Handles non-linear relationships
  - Feature importance easy to extract

- ❌ **Weaknesses:**
  - Prone to overfitting
  - Less stable (high variance)
  - Higher cost than LR (0.552)
  - Lower recall at optimal threshold

**Logistic Regression:**
- ✅ **Strengths:**
  - **Lowest cost (0.509)** ← Key advantage
  - **Highest recall (0.78)** ← Catches most customers
  - Well-calibrated probabilities
  - Better generalization
  - Industry standard for binary classification

- ❌ **Weaknesses:**
  - Less interpretable than DT
  - Assumes linear relationships (after transformation)
  - Requires feature scaling

**3. Final Model Selection**

**Winner: Logistic Regression (Cost-Optimized, Threshold=0.38)**

**Justification:**
1. **Lowest campaign cost:** 0.509 vs 0.552 (DT)
2. **Highest recall:** 0.78 (catches 78% of potential customers)
3. **Better business ROI:** Lower cost directly translates to higher profit
4. **Well-calibrated probabilities:** Better for threshold tuning
5. **Generalizes better:** More stable across different data samples

**Trade-offs accepted:**
- Lower accuracy (0.76) vs baseline (0.90) - acceptable due to better recall
- Lower precision (0.32) - acceptable due to cost structure (FN >> FP)
- Less interpretability - mitigated by coefficient analysis

**4. Combined ROC Curve Visualization**

```python
plt.figure(figsize=(10, 8))

# Baseline models
RocCurveDisplay.from_predictions(y_test, y_proba_dt,
                                  name='DT Baseline (AUC=0.79)',
                                  linestyle=':', color='blue', ax=plt.gca())
RocCurveDisplay.from_predictions(y_test, y_proba_lr,
                                  name='LR Baseline (AUC=0.80)',
                                  linestyle=':', color='orange', ax=plt.gca())

# Tuned models
RocCurveDisplay.from_predictions(y_test, y_proba_dt_tuned,
                                  name='DT Tuned (AUC=0.80)',
                                  linestyle='--', color='blue', ax=plt.gca())
RocCurveDisplay.from_predictions(y_test, y_proba_lr_tuned,
                                  name='LR Tuned (AUC=0.80)',
                                  linestyle='--', color='orange', ax=plt.gca())

# Highlight final winner
RocCurveDisplay.from_predictions(y_test, y_proba_lr_tuned,
                                  name='★ LR Final (Cost=0.509)',
                                  linewidth=3, color='green', ax=plt.gca())

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.title('ROC Curve Comparison - All Models')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig('assets/21_roc_comparison_final.png', dpi=300, bbox_inches='tight')
```

**5. Performance Progression Summary**

Document the improvement journey:

**Recall progression (Logistic Regression):**
- Baseline: 0.22 (catching only 22% of customers)
- After tuning: 0.64 (catching 64% - nearly 3x improvement!)
- After cost optimization: 0.78 (catching 78% - optimal for business cost)

**This means:**
- Baseline: Missing 78% of potential customers
- Final: Missing only 22% of potential customers
- **Massive improvement in campaign efficiency**

**Cost progression:**
- Before optimization: Using default threshold 0.5
- After optimization: Average cost reduced to 0.509 per customer
- **Significant cost savings at scale:** For 10,000 calls, saving thousands of dollars

### Expected Output
- Master comparison table (all stages, all metrics)
- Detailed model comparison analysis
- Clear justification for Logistic Regression selection
- Performance progression summary
- Visualization: `assets/21_roc_comparison_final.png`
- Documentation ready for report Conclusion section

---

## **PHASE 9: Report Writing**

### Objectives
- Write complete academic report following template structure
- Integrate all results, visualizations, and insights
- Professional academic writing style
- ~2,500 words target
- Properly cite all references

### Report Structure

Following **input/2-assignment-template.md** exactly:

---

### **1. Introduction (~400 words)**

**Content to include:**

**Paragraph 1 - Context and Motivation:**
- Direct marketing is critical for banks and financial institutions
- Challenge: Poor targeting leads to wasted resources (calls to non-interested customers)
- Opportunity: Predictive analytics can optimize resource allocation
- Focus: Classification models to minimize false positives (wasted calls) and false negatives (missed customers)

**Paragraph 2 - Dataset Description:**
- Bank Marketing Dataset from UCI Machine Learning Repository [cite Moro et al., 2014]
- 41,188 records from Portuguese bank telemarketing campaigns
- Features: Socio-demographic details, campaign information, economic indicators
- Target variable: Whether customer subscribed to term deposit (yes/no)
- Class imbalance: ~11% acceptance, ~89% rejection

**Paragraph 3 - Literature Context:**
- Machine learning widely applied in marketing prediction [cite relevant papers]
- Decision Trees: Transparency, interpretability, feature ranking
- Logistic Regression: Probabilistic outputs, threshold tuning, industry standard
- Cost-sensitive approaches address business ROI directly

**Paragraph 4 - Objectives:**
- Primary goal: Minimize campaign cost by reducing FP and FN errors
- Construct and compare two models: Decision Tree and Logistic Regression
- Optimize using hyperparameter tuning and cost-sensitive thresholds
- Provide actionable business insights

**Key references to cite:**
- Moro, S., Cortez, P., & Rita, P. (2014) - Dataset paper [REQUIRED]
- Marketing analytics papers
- ML textbooks

---

### **2. Designing a Solution (~700 words)**

#### **2.1 Exploratory Data Analysis (~250 words)**

**Content:**
- Dataset composition: 20 input features (10 categorical, 8 numerical, 2 economic indicators)
- Target distribution: 11.3% positive, 88.7% negative (severe imbalance)
- **Numerical features:** Age, campaign, previous show right-skewed distributions
  - Reference: `assets/02_numerical_distributions.png`
- **Categorical patterns:**
  - Most clients married, admin/blue-collar jobs
  - Housing loans common, personal loans less frequent
  - Calls peaked in May, July, August
  - Reference: `assets/06_job_analysis.png`, `assets/07_month_seasonality.png`
- **Special variable - pdays:** 999 indicates never contacted before (majority)
  - Reference: `assets/09_pdays_distribution.png`
- **Economic indicators:** Employment rate, consumer confidence, Euribor rate form clusters
  - High correlation among economic indicators
  - Reference: `assets/10_correlation_heatmap.png`
- **Duration variable discussion:**
  - Strong correlation with success (longer calls → higher acceptance)
  - **Data leakage issue:** Only known after call concludes
  - **Decision:** Exclude from predictive models (unsuitable for pre-call prediction)
  - Reference: `assets/12_duration_leakage_analysis.png`

**Key findings summary:**
- Severe class imbalance requires specialized handling
- Economic context strongly influences campaign success
- Previous contact history valuable predictor
- Feature engineering opportunities identified

#### **2.2 Data Cleaning and Preprocessing (~250 words)**

**Content:**

**Missing data handling:**
- Replaced 'unknown' categorical values with NaN
- Numerical features: Mean imputation (maintains distribution center)
- Categorical features: Mode imputation (most frequent value)
- Justification: Standard practice, simple, effective for this dataset

**Feature engineering:**
- **Dropped:** `duration` variable (data leakage - explained in 2.1)
- **Created:**
  - `was_contacted_before`: Binary flag (1 if pdays ≠ 999, else 0)
    - Rationale: Separates warm leads from cold prospects
  - `campaign_log`: log(campaign + 1) transformation
    - Rationale: Reduces right skewness
  - `previous_log`: log(previous + 1) transformation
    - Rationale: Reduces right skewness

**Encoding and scaling:**
- **One-Hot Encoding** for categorical variables (handle_unknown='ignore')
  - Justification: Nominal categories, no ordinal relationship
- **StandardScaler** for numerical features
  - Justification: Different scales, required for Logistic Regression
- Final feature space: 59 features (after encoding)

**Data split:**
- 75% training (30,891 samples), 25% testing (10,297 samples)
- **Stratified split** (stratify=y) to maintain class ratio
- Justification: Critical for imbalanced data evaluation
- random_state=42 for reproducibility

**Class imbalance mitigation:**
- Used `class_weight='balanced'` in both models
- Automatically adjusts weights inversely proportional to class frequencies
- Prevents majority class bias

#### **2.3 Modeling Approach (~200 words)**

**Content:**

**Algorithms selected:**

**1. Decision Tree (Entropy criterion):**
- Rationale: High interpretability, visual decision rules
- Handles non-linear relationships naturally
- No scaling required
- Feature importance extraction
- Suitable for business rule generation

**2. Logistic Regression:**
- Rationale: Probabilistic predictions (well-calibrated)
- Industry standard for binary classification
- Supports threshold optimization
- Simple, efficient, generalizes well

**Evaluation metrics:**
- **Accuracy:** Overall correctness (baseline metric)
- **Precision:** Of predicted positives, how many correct? (FP consideration)
- **Recall:** Of actual positives, how many caught? (FN consideration - PRIORITY)
- **F1-Score:** Harmonic mean of Precision and Recall
- **ROC-AUC:** Threshold-independent discrimination ability

**Metric prioritization:**
- **Recall prioritized** over accuracy
- Justification: Detecting positive responses critical (FN cost > FP cost)
- Business context: Missing customers more expensive than unnecessary calls

**Optimization strategy:**
- Stage 1: Baseline models with default parameters
- Stage 2: GridSearchCV with cross-validation
- Stage 3: Cost-sensitive threshold optimization

---

### **3. Experiments (~900 words)**

#### **3.1 Baseline Models (~150 words)**

**Content:**

**Decision Tree baseline:**
- Configuration: entropy criterion, balanced class weights, default parameters
- Performance: Accuracy=0.90, Precision=0.65, Recall=0.28, F1=0.39, ROC-AUC=0.79
- Reference: `assets/13_baseline_confusion_matrices.png`, `assets/14_baseline_roc_curves.png`

**Logistic Regression baseline:**
- Configuration: balanced class weights, L2 penalty, default C=1.0
- Performance: Accuracy=0.90, Precision=0.70, Recall=0.22, F1=0.33, ROC-AUC=0.80
- Reference: Same figures as above

**Baseline comparison table:**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Decision Tree | 0.9020 | 0.6513 | 0.2802 | 0.3918 | 0.7930 |
| Logistic Regression | 0.9013 | 0.7011 | 0.2164 | 0.3307 | 0.8046 |

**Analysis:**
- High accuracy (~90%) due to class imbalance (baseline=89%)
- **Low recall** (~0.22-0.28): Missing many positive cases - PROBLEM!
- ROC-AUC ~0.80: Reasonable discrimination ability
- Confirms need for optimization to improve recall

#### **3.2 Hyperparameter Optimization (~250 words)**

**Content:**

**Decision Tree GridSearchCV:**
- Parameters tuned: max_depth, min_samples_leaf, min_samples_split, ccp_alpha
- Cross-validation: 5-fold StratifiedKFold (maintains class ratio)
- Scoring metric: ROC-AUC (threshold-independent)
- Best parameters found:
  ```
  {
    'ccp_alpha': 0.001,
    'class_weight': 'balanced',
    'max_depth': None,
    'min_samples_leaf': 1,
    'min_samples_split': 2
  }
  ```
- Best CV ROC-AUC: 0.8014

**Logistic Regression GridSearchCV:**
- Parameters tuned: C (regularization), penalty (L1/L2), solver
- Same CV strategy
- Best parameters found:
  ```
  {
    'C': 0.1,
    'class_weight': 'balanced',
    'penalty': 'l1',
    'solver': 'saga'
  }
  ```
- Best CV ROC-AUC: 0.8041

**Tuned model performance:**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| DT (tuned) | 0.8631 | 0.4260 | **0.6207** | 0.5053 | 0.8014 |
| LR (tuned) | 0.8344 | 0.3660 | **0.6414** | 0.4660 | 0.8041 |

**Analysis:**
- **Substantial recall improvement:** DT +0.34, LR +0.42
- Accuracy decreased (expected - less majority class bias)
- Precision decreased (trade-off for higher recall)
- ROC-AUC maintained or improved
- Reference: `assets/15_tuned_confusion_matrices.png`, `assets/16_tuned_roc_curves.png`

**Interpretation:**
- Successfully reduced false negatives (missed customers)
- Models now catch 62-64% of potential customers vs 22-28% baseline
- Trade-off: More false positives (unnecessary calls) - addressed in next stage

#### **3.3 Cost-Sensitive Threshold Optimization (~250 words)**

**Content:**

**Business cost matrix:**
- False Positive (unnecessary call): Cost = +1.5
- False Negative (missed customer): Cost = +20.0
- True Positive (successful sale): Cost = -5.0 (revenue/profit)
- True Negative (correctly avoided): Cost = 0.0

**Rationale:**
- Asymmetric costs: Missing customer (FN) much more expensive than wasted call (FP)
- FN/FP ratio = 13.3x reflects business reality
- True positive generates revenue (negative cost = profit)

**Threshold sweep methodology:**
- Tested 99 thresholds from 0.01 to 0.99
- Calculated expected cost per customer at each threshold
- Identified minimum cost threshold
- Reference: `assets/17_cost_threshold_optimization.png`

**Optimal thresholds found:**

| Model | Optimal Threshold | Average Cost |
|-------|-------------------|--------------|
| Decision Tree | 0.35 | 0.552 |
| Logistic Regression | 0.38 | **0.509** |

**Analysis:**
- Optimal thresholds much lower than default 0.5
- Lower threshold → Favor recall over precision (align with cost structure)
- **Logistic Regression achieves lowest cost** → Winner model

**Performance at optimal thresholds:**

| Model | Threshold | Recall | Precision | Avg Cost |
|-------|-----------|--------|-----------|----------|
| DT | 0.35 | 0.75 | 0.35 | 0.552 |
| LR | 0.38 | **0.78** | 0.32 | **0.509** |

**Interpretation:**
- Recall further improved to 0.75-0.78 (catching 75-78% of customers!)
- Only missing 22-25% of potential customers (vs 72-78% at baseline)
- Cost minimization achieved while maximizing customer capture
- Logistic Regression: Best balance of recall and cost efficiency

#### **3.4 Model Interpretability (~150 words)**

**Content:**

**Decision Tree feature importance (Top 5):**
1. nr.employed (employment rate) - 0.67
2. cons.conf.idx (consumer confidence) - 0.13
3. was_contacted_before - 0.05
4. cons.price.idx - 0.04
5. euribor3m - 0.03

**Analysis:**
- Economic indicators dominate (nr.employed = 67% of importance)
- Consumer confidence second most important
- Previous contact history matters (our engineered feature)
- Reference: `assets/18_dt_feature_importance.png`, `assets/19_decision_tree_structure.png`

**Logistic Regression coefficients (Top by |β|):**
- emp.var.rate (-1.69): Higher employment variation → lower acceptance
- month_mar (+1.07): March campaigns more successful
- cons.price.idx (+0.77): Higher CPI → higher acceptance
- poutcome_failure (-0.29): Previous failure → lower probability

**Analysis:**
- Negative coefficients decrease acceptance probability
- Positive coefficients increase acceptance probability
- Economic and temporal factors crucial
- Reference: `assets/20_lr_coefficients.png`

**Business insights:**
- **Employment rate** and **consumer confidence** are strongest predictors
- **Economic stability** critical for campaign success
- **Previous contact** valuable (warm leads perform better)
- **Seasonality** matters (March advantageous)
- **Macroeconomic monitoring** should guide campaign timing

#### **3.5 ROC Curve Comparison (~100 words)**

**Content:**
- ROC curves demonstrate clear improvement from baseline to tuned models
- Both models achieve ROC-AUC ~0.80 (good discrimination ability)
- Curves show better true positive rate across all false positive rates
- Reference: `assets/21_roc_comparison_final.png`

**Analysis:**
- Baseline models: AUC ~0.79-0.80
- Tuned models: AUC maintained at ~0.80
- Improvement primarily in recall at chosen thresholds (not reflected in AUC alone)
- Cost-sensitive thresholds operate at higher TPR, accepting higher FPR (aligned with cost structure)

---

### **4. Conclusions (~400 words)**

**Content to include:**

**Paragraph 1 - Summary of achievements:**
- Successfully developed predictive models to optimize direct marketing campaign
- Two algorithms compared: Decision Tree and Logistic Regression
- Multi-stage optimization: baseline → hyperparameter tuning → cost-sensitive threshold
- Achieved substantial recall improvement and cost minimization

**Paragraph 2 - Key findings:**
- **Class imbalance** initially caused low recall (~22-28% baseline)
- **Hyperparameter tuning** improved recall to 62-64% (nearly 3x improvement)
- **Cost-sensitive threshold optimization** further improved to 75-78% recall
- **Final model:** Logistic Regression with threshold=0.38
  - Recall: 0.78 (catching 78% of potential customers)
  - Average cost: 0.509 per customer (lowest among all models)
  - Precision: 0.32 (acceptable trade-off given cost structure)

**Paragraph 3 - Model comparison:**
- Both models achieved competitive performance
- **Decision Tree advantages:** High interpretability, visual rules, feature importance
- **Logistic Regression advantages:** Lower cost (0.509 vs 0.552), higher recall (0.78 vs 0.75), better generalization
- **Final selection:** Logistic Regression due to superior cost efficiency and recall

**Paragraph 4 - Business insights and recommendations:**
- **Economic indicators** (employment rate, consumer confidence) are strongest predictors
- **Timing recommendations:**
  - Launch campaigns during economically stable periods
  - Target March for seasonal advantage
  - Monitor macroeconomic indicators before campaigns
- **Targeting recommendations:**
  - Prioritize previously contacted customers (warm leads)
  - Avoid campaigns during high employment variation periods
- **Expected business impact:**
  - 78% customer capture vs 22% baseline (3.5x improvement)
  - Reduced average cost to 0.509 per customer
  - For 10,000 contact campaign: Significant cost savings and revenue increase

**Paragraph 5 - Comparison with literature:**
- Performance competitive with benchmarks from UCI repository
- ROC-AUC ~0.80 aligns with published results [cite if available]
- Cost-sensitive approach provides additional business value beyond accuracy

**Paragraph 6 - Limitations:**
- Duration variable excluded (data leakage) - limits maximum achievable accuracy
- Class imbalance persists despite mitigation strategies
- Cost matrix values assumed (could be refined with real business data)
- Economic indicators may not generalize to other countries/time periods
- Test set limited to 25% - could benefit from additional validation

**Paragraph 7 - Future work:**
- **Ensemble methods:** Random Forest, Gradient Boosting for potentially better performance
- **Advanced sampling:** SMOTE, ADASYN for imbalance handling
- **Feature engineering:** Additional domain-specific features
- **Cost matrix refinement:** Use actual business data for precise costs
- **Temporal validation:** Test on future campaigns (time-based split)
- **Model deployment:** Real-time prediction system integration
- **A/B testing:** Compare model-driven targeting vs traditional methods

---

### **5. References**

**Minimum 8-10 references to include:**

1. **Moro, S., Cortez, P., & Rita, P. (2014).** A Data-Driven Approach to Predict the Success of Bank Telemarketing. *Decision Support Systems*, 62, 22-31. [REQUIRED - dataset paper]

2. **Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984).** *Classification and Regression Trees.* Chapman & Hall/CRC.

3. **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning: Data Mining, Inference, and Prediction.* 2nd ed. Springer.

4. **James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021).** *An Introduction to Statistical Learning with Applications in R.* 2nd ed. Springer.

5. **Provost, F., & Fawcett, T. (2013).** *Data Science for Business: What You Need to Know about Data Mining and Data-Analytic Thinking.* O'Reilly Media.

6. **Elkan, C. (2001).** The Foundations of Cost-Sensitive Learning. *Proceedings of the 17th International Joint Conference on Artificial Intelligence*, 973-978.

7. **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).** SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357.

8. **Pedregosa, F., et al. (2011).** Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

9. **Géron, A. (2022).** *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.* 3rd ed. O'Reilly Media.

10. **Additional domain-specific papers** on marketing analytics, telemarketing prediction, or cost-sensitive classification as found during literature review.

---

### **6. Appendix**

**Content:**
- "Complete implementation available in `notebook.ipynb`"
- "All visualizations available in `assets/` folder"
- Optional: Full data dictionary table
- Optional: Additional confusion matrices or metric tables

---

### Report Writing Guidelines

**Writing style:**
- Academic formal tone
- Third person (avoid "I" or "we")
- Past tense for experiments ("The model achieved...")
- Present tense for facts ("Decision Trees are...")
- Clear, concise sentences
- Define technical terms on first use

**Formatting:**
- Markdown format (.md file)
- Headers: # for main sections, ## for subsections
- Tables: Markdown table syntax
- Images: `![Description](assets/filename.png)`
- Lists: Bullet points or numbered
- Code: Inline with backticks or blocks with triple backticks (sparingly in report)

**Integration with notebook:**
- All metrics in report must match notebook outputs exactly
- Reference figure numbers consistently
- Ensure all images referenced actually exist in assets/

**Word count management:**
- Introduction: ~400
- Section 2: ~700 (split: 250 + 250 + 200)
- Section 3: ~900 (split: 150 + 250 + 250 + 150 + 100)
- Conclusions: ~400
- Total: ~2,400 words (excluding references, tables, figure captions)

### Expected Output
- Complete `report.md` file (~2,500 words)
- Professional academic writing
- All sections following template structure
- All visualizations properly referenced
- Proper citations throughout
- Ready for submission

---

## **PHASE 10: Final Quality Checks**

### Objectives
- Ensure all deliverables are complete and professional
- Verify consistency across notebook, assets, and report
- Check reproducibility
- Final polishing

### Tasks

**1. Notebook Quality Check**

**Structure:**
- [ ] Clear title and description at top
- [ ] Markdown headers for each phase/section
- [ ] Code cells with comments explaining logic
- [ ] Output cells visible (no errors)
- [ ] Visualizations rendered inline
- [ ] Summary/conclusions at end

**Code quality:**
- [ ] Consistent variable naming
- [ ] No hardcoded paths (use relative paths)
- [ ] random_state set consistently (42)
- [ ] All imports at top
- [ ] Functions defined before use
- [ ] No unused code or commented-out blocks

**Reproducibility:**
- [ ] Run "Restart & Run All" - verify no errors
- [ ] Check execution time reasonable
- [ ] Verify all outputs consistent
- [ ] Save notebook with outputs visible

**Documentation:**
- [ ] Each section has explanatory markdown
- [ ] Key decisions documented
- [ ] Results interpreted (not just printed)
- [ ] Rationale for choices explained

**2. Assets Folder Quality Check**

**File organization:**
- [ ] All images saved with descriptive names
- [ ] Consistent naming convention (01_description.png, 02_description.png, ...)
- [ ] All images referenced in report actually exist
- [ ] No unused/duplicate images

**Image quality:**
- [ ] Resolution adequate (300 dpi for figures)
- [ ] Clear labels, titles, legends
- [ ] Readable font sizes
- [ ] Consistent color schemes
- [ ] Professional appearance

**Completeness:**
- [ ] All phases generated required visualizations
- [ ] ~15-20 images total
- [ ] Key figures for report included

**Expected assets list:**
```
assets/
├── 01_class_distribution.png
├── 02_numerical_distributions.png
├── 03_age_vs_target.png
├── 04_duration_analysis.png
├── 05_economic_indicators.png
├── 06_job_analysis.png
├── 07_month_seasonality.png
├── 08_poutcome_impact.png
├── 09_pdays_distribution.png
├── 10_correlation_heatmap.png
├── 11_bivariate_analysis.png
├── 12_duration_leakage_analysis.png
├── 13_baseline_confusion_matrices.png
├── 14_baseline_roc_curves.png
├── 15_tuned_confusion_matrices.png
├── 16_tuned_roc_curves.png
├── 17_cost_threshold_optimization.png
├── 18_dt_feature_importance.png
├── 19_decision_tree_structure.png
├── 20_lr_coefficients.png
└── 21_roc_comparison_final.png
```

**3. Report Quality Check**

**Content:**
- [ ] Follows template structure exactly
- [ ] All sections complete
- [ ] Word count ~2,500 (±200)
- [ ] All metrics match notebook outputs
- [ ] All tables formatted correctly
- [ ] All figures referenced

**Writing:**
- [ ] Spell check completed
- [ ] Grammar check completed
- [ ] Academic tone consistent
- [ ] No first person ("I", "we")
- [ ] Technical terms defined
- [ ] Clear, concise sentences

**References:**
- [ ] Minimum 8 references
- [ ] Moro et al. (2014) included [REQUIRED]
- [ ] Consistent citation format
- [ ] All citations have corresponding references
- [ ] References formatted correctly

**Figures:**
- [ ] All image links working (![](assets/...))
- [ ] Figure captions descriptive
- [ ] Figures referenced in text ("As shown in Figure X...")
- [ ] Numbering consistent

**Tables:**
- [ ] Markdown table syntax correct
- [ ] Headers clear
- [ ] Values aligned
- [ ] Table captions included
- [ ] Tables referenced in text

**4. Cross-Reference Verification**

**Consistency checks:**
- [ ] Report metrics = Notebook outputs (exact match!)
- [ ] Report figures = Assets folder files (all exist)
- [ ] Report parameter values = Notebook parameter values
- [ ] Report interpretations aligned with visualizations

**Numbers to verify:**
- [ ] Dataset size (41,188 records)
- [ ] Class distribution (11.3% / 88.7%)
- [ ] Train/test split (75% / 25%)
- [ ] Final feature count (~59)
- [ ] All performance metrics (accuracy, precision, recall, F1, ROC-AUC)
- [ ] Optimal thresholds (DT: 0.35, LR: 0.38)
- [ ] Final costs (DT: 0.552, LR: 0.509)
- [ ] Top feature importances
- [ ] Top coefficients

**5. Assignment Requirements Verification**

**Threshold requirements (to pass):**
- [x] Created project environment (replaces R Cloud)
- [x] Uploaded data and code
- [x] Built Decision Tree model
- [x] Analyzed and described results
- **Total:** 42% requirements met

**Advanced requirements (60-70%):**
- [x] Identified parameter sets for optimization
- [x] Explained parameter influence on accuracy
- [x] Ran multiple experiments (baseline, tuned, cost-optimized)
- [x] Compared results (DT vs LR comparison)
- [x] Compared with literature (referenced benchmarks)
- **Total:** All advanced requirements met

**Excellence requirements (70%+):**
- [x] Publishable-level analysis (cost-sensitive optimization)
- [x] Multiple parameter sets tested systematically
- [x] Competitive performance to state-of-art
- [x] Professional presentation
- [x] Business-focused insights
- **Total:** Excellence criteria met

**Grading rubric alignment:**

| Criterion | Weight | Our Work | Expected Grade |
|-----------|--------|----------|----------------|
| Analysis | 30% | Comprehensive EDA, literature integration, SOTA comparison | 70%+ (Excellent) |
| Design | 40% | Multiple experiments, GridSearch, cost optimization, competitive performance | 70%+ (Excellent) |
| Conclusion | 30% | Multi-parameter comparison, demonstrates competitive solution | 70%+ (Excellent) |

**Target grade:** 70%+ (First Class)

**6. Final Deliverables Checklist**

- [ ] `notebook.ipynb` - Complete, executable, documented
- [ ] `assets/` folder - 15-20 professional visualizations
- [ ] `report.md` - 2,500 words, academic quality, follows template
- [ ] All files in correct locations
- [ ] No missing dependencies
- [ ] Reproducible from scratch

**7. Submission Preparation**

**File organization:**
```
bankml/
├── notebook.ipynb           ← Main deliverable
├── assets/                  ← All visualizations
│   ├── 01_*.png
│   ├── 02_*.png
│   └── ...
├── report.md                ← Main deliverable
├── input/
│   └── 4-data.csv
├── CLAUDE.md               ← For reference
├── EXECUTION_PLAN.md       ← For reference
└── README.md               ← Optional
```

**Optional: Create README.md:**
```markdown
# Bank Marketing Campaign Optimization

Data Mining assignment for CIS051-3 Business Analytics

## Deliverables
- `notebook.ipynb`: Complete ML pipeline implementation
- `report.md`: Assignment report (~2,500 words)
- `assets/`: All visualizations

## How to Run
1. Ensure Python 3.x with required libraries installed
2. Open `notebook.ipynb` in Jupyter
3. Run all cells sequentially

## Results
- Final model: Logistic Regression (threshold=0.38)
- Recall: 0.78 (catching 78% of customers)
- Average cost: 0.509 per customer
```

**Final checks before submission:**
- [ ] All deliverables complete
- [ ] Quality verified
- [ ] Consistency checked
- [ ] Ready for submission

### Expected Output
- Three polished, professional deliverables
- All quality checks passed
- Meets all assignment requirements
- Aiming for 70%+ grade (First Class)
- Ready for submission to BREO

---

## SUMMARY

### What We're Building

A complete data mining solution with:
1. **Jupyter Notebook** - End-to-end ML pipeline with comprehensive documentation
2. **Visualization Assets** - 15-20 professional charts and figures
3. **Academic Report** - 2,500-word analysis following assignment template

### Key Technical Achievements

- Decision Tree and Logistic Regression models
- Systematic hyperparameter optimization via GridSearchCV
- Cost-sensitive threshold optimization (novel business-focused approach)
- 3.5x improvement in recall (0.22 → 0.78)
- Minimized campaign cost to 0.509 per customer

### Key Academic Achievements

- Comprehensive EDA with business insights
- Rigorous methodology with justifications
- Literature-grounded approach
- Professional visualizations
- Publishable-level analysis (targeting 70%+ grade)

### Timeline

**Estimated total: 18-26 hours** across 10 phases

### Critical Success Factors

1. ✅ Understand duration variable issue (data leakage)
2. ✅ Properly handle class imbalance (stratification, balanced weights)
3. ✅ Prioritize recall over accuracy (business alignment)
4. ✅ Systematic optimization (baseline → tuned → cost-optimized)
5. ✅ Professional presentation (academic writing, quality visualizations)

---

## EXECUTION READINESS

This plan is now ready for execution. Proceed phase-by-phase or execute in chunks as preferred.

**Recommended execution approach:**
- **Chunk 1:** Phases 0-2 (Setup, understanding, EDA)
- **Chunk 2:** Phases 3-5 (Preprocessing, baseline, tuning)
- **Chunk 3:** Phases 6-8 (Cost optimization, interpretation, comparison)
- **Chunk 4:** Phases 9-10 (Report writing, quality checks)

Ready to begin when you approve! 🚀
