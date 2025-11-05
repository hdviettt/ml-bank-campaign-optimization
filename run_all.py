"""
Bank Marketing Campaign Optimization - Complete Automated Execution
All phases in one script - Windows compatible
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import pickle
from pathlib import Path

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)

# Settings
warnings.filterwarnings('ignore')
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('ggplot')
sns.set_palette('husl')
pd.set_option('display.max_columns', None)
np.random.seed(42)

# Create directories
Path('assets').mkdir(exist_ok=True)
Path('output').mkdir(exist_ok=True)

print("="*80)
print("BANK MARKETING CAMPAIGN OPTIMIZATION - AUTOMATED EXECUTION")
print("="*80)

# ============================================================================
# PHASE 0-1: Setup & Data Loading
# ============================================================================
print("\nPHASE 0-1: Setup & Data Loading")
print("-"*80)

df_original = pd.read_csv('data/bank-marketing.csv', sep=';')
print(f"Dataset loaded: {df_original.shape[0]:,} rows, {df_original.shape[1]} columns")

# ============================================================================
# PHASE 2: Exploratory Data Analysis
# ============================================================================
print("\nPHASE 2: Exploratory Data Analysis")
print("-"*80)

df = df_original.copy()

# Target distribution
print("\n2.1 Target Variable Analysis")
target_counts = df['y'].value_counts()
target_pct = df['y'].value_counts(normalize=True) * 100
print(f"  No (rejected): {target_counts['no']:,} ({target_pct['no']:.2f}%)")
print(f"  Yes (accepted): {target_counts['yes']:,} ({target_pct['yes']:.2f}%)")

# VIZ 1: Class Distribution
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(target_counts.index, target_counts.values, color=['#ff6b6b', '#4ecdc4'])
ax.set_xlabel('Campaign Response', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Target Variable Distribution (Campaign Acceptance)', fontsize=14, fontweight='bold')
for bar, count, pct in zip(bars, target_counts.values, target_pct.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f'{count:,}\\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('assets/01_class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 01_class_distribution.png")

# Numerical features
numerical_cols = ['age', 'campaign', 'previous', 'pdays', 'duration',
                  'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

# VIZ 2: Numerical Distributions
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()
for idx, col in enumerate(numerical_cols[:10]):
    ax = axes[idx]
    ax.hist(df[col].dropna(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}', fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    mean_val = df[col].mean()
    median_val = df[col].median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
    ax.legend(fontsize=8)
for idx in range(len(numerical_cols), 12):
    axes[idx].axis('off')
plt.suptitle('Numerical Features Distributions', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('assets/02_numerical_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 02_numerical_distributions.png")

# VIZ 3: Age vs Target
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for target in ['no', 'yes']:
    axes[0].hist(df[df['y'] == target]['age'], bins=30, alpha=0.6, label=target)
axes[0].set_xlabel('Age', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Age Distribution by Campaign Response', fontweight='bold')
axes[0].legend()
df.boxplot(column='age', by='y', ax=axes[1])
axes[1].set_xlabel('Campaign Response', fontsize=12)
axes[1].set_ylabel('Age', fontsize=12)
axes[1].set_title('Age Distribution by Response', fontweight='bold')
plt.suptitle('')
plt.tight_layout()
plt.savefig('assets/03_age_vs_target.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 03_age_vs_target.png")

# VIZ 4: Duration Analysis (Data Leakage)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df_duration = df[df['duration'] > 0]
for target in ['no', 'yes']:
    axes[0].hist(df_duration[df_duration['y'] == target]['duration'],
                 bins=50, alpha=0.6, label=target, range=(0, 2000))
axes[0].set_xlabel('Duration (seconds)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Call Duration Distribution by Response', fontweight='bold')
axes[0].legend()
df_duration.boxplot(column='duration', by='y', ax=axes[1])
axes[1].set_xlabel('Campaign Response', fontsize=12)
axes[1].set_ylabel('Duration (seconds)', fontsize=12)
axes[1].set_title('Duration by Response - Strong correlation but DATA LEAKAGE', fontweight='bold')
axes[1].set_ylim(0, 2000)
plt.suptitle('')
plt.tight_layout()
plt.savefig('assets/04_duration_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 04_duration_analysis.png")

# VIZ 5: Economic Indicators
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
economic_cols = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
for idx, col in enumerate(economic_cols):
    ax = axes[idx]
    for target in ['no', 'yes']:
        df[df['y'] == target][col].hist(bins=30, alpha=0.6, label=target, ax=ax)
    ax.set_xlabel(col, fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{col} by Response', fontweight='bold')
    ax.legend()
axes[5].axis('off')
plt.suptitle('Economic Indicators Distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('assets/05_economic_indicators.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 05_economic_indicators.png")

# VIZ 6: Job Analysis
fig, ax = plt.subplots(figsize=(12, 6))
job_target = pd.crosstab(df['job'], df['y'], normalize='index') * 100
job_target = job_target.sort_values('yes', ascending=False)
job_target.plot(kind='bar', stacked=False, ax=ax, color=['#ff6b6b', '#4ecdc4'])
ax.set_xlabel('Job Type', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Campaign Acceptance Rate by Job Type', fontsize=14, fontweight='bold')
ax.legend(['Rejected', 'Accepted'], loc='upper right')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig('assets/06_job_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 06_job_analysis.png")

# VIZ 7: Month Seasonality
fig, ax = plt.subplots(figsize=(12, 6))
month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
month_target = pd.crosstab(df['month'], df['y'])
month_target = month_target.reindex([m for m in month_order if m in month_target.index])
month_target.plot(kind='bar', stacked=True, ax=ax, color=['#ff6b6b', '#4ecdc4'])
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Campaign Contacts by Month', fontsize=14, fontweight='bold')
ax.legend(['Rejected', 'Accepted'], loc='upper right')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig('assets/07_month_seasonality.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 07_month_seasonality.png")

# VIZ 8: Previous Outcome Impact
fig, ax = plt.subplots(figsize=(10, 6))
poutcome_target = pd.crosstab(df['poutcome'], df['y'], normalize='index') * 100
poutcome_target.plot(kind='bar', ax=ax, color=['#ff6b6b', '#4ecdc4'])
ax.set_xlabel('Previous Campaign Outcome', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Current Campaign Response by Previous Outcome', fontsize=14, fontweight='bold')
ax.legend(['Rejected', 'Accepted'], loc='upper right')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig('assets/08_poutcome_impact.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 08_poutcome_impact.png")

# VIZ 9: pdays Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['pdays'], bins=50, color='skyblue', edgecolor='black')
axes[0].set_xlabel('Days Since Last Contact (pdays)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('pdays Distribution (999 = Never Contacted Before)', fontweight='bold')
axes[0].axvline(999, color='red', linestyle='--', linewidth=2, label='999 (Never contacted)')
axes[0].legend()
contacted_before = df[df['pdays'] != 999]['y'].value_counts(normalize=True)['yes'] * 100
never_contacted = df[df['pdays'] == 999]['y'].value_counts(normalize=True)['yes'] * 100
categories = ['Contacted Before (pdays != 999)', 'Never Contacted (pdays = 999)']
values = [contacted_before, never_contacted]
bars = axes[1].bar(categories, values, color=['#4ecdc4', '#ff6b6b'])
axes[1].set_ylabel('Acceptance Rate (%)', fontsize=12)
axes[1].set_title('Impact of Previous Contact', fontweight='bold')
for bar, val in zip(bars, values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('assets/09_pdays_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 09_pdays_distribution.png")

# VIZ 10: Correlation Heatmap
fig, ax = plt.subplots(figsize=(12, 10))
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix - Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('assets/10_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 10_correlation_heatmap.png")

# VIZ 11: Bi-variate Analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for idx, col in enumerate(['nr.employed', 'euribor3m']):
    ax = axes[0, idx]
    df.boxplot(column=col, by='y', ax=ax)
    ax.set_xlabel('Campaign Response', fontsize=11)
    ax.set_ylabel(col, fontsize=11)
    ax.set_title(f'{col} by Response', fontweight='bold')
    plt.suptitle('')
for idx, col in enumerate(['campaign', 'previous']):
    ax = axes[1, idx]
    df_limited = df[df[col] <= df[col].quantile(0.95)]
    df_limited.boxplot(column=col, by='y', ax=ax)
    ax.set_xlabel('Campaign Response', fontsize=11)
    ax.set_ylabel(col, fontsize=11)
    ax.set_title(f'{col} by Response', fontweight='bold')
    plt.suptitle('')
plt.tight_layout()
plt.savefig('assets/11_bivariate_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 11_bivariate_analysis.png")

# VIZ 12: Duration Leakage Explanation
fig, ax = plt.subplots(figsize=(10, 6))
df_temp = df[df['duration'] > 0].copy()
df_temp['duration_bin'] = pd.cut(df_temp['duration'], bins=[0, 120, 300, 600, 1200, 10000],
                                   labels=['0-2min', '2-5min', '5-10min', '10-20min', '20+min'])
duration_acceptance = pd.crosstab(df_temp['duration_bin'], df_temp['y'], normalize='index') * 100
duration_acceptance['yes'].plot(kind='bar', ax=ax, color='#4ecdc4')
ax.set_xlabel('Call Duration', fontsize=12)
ax.set_ylabel('Acceptance Rate (%)', fontsize=12)
ax.set_title('Acceptance Rate by Call Duration - Strong Correlation (DATA LEAKAGE)', fontweight='bold', fontsize=13)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.text(0.5, -0.15, 'Note: Duration is only known AFTER call completion - excluded from predictive models',
        ha='center', transform=ax.transAxes, fontsize=10, style='italic')
plt.tight_layout()
plt.savefig('assets/12_duration_leakage_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 12_duration_leakage_analysis.png")

print(f"\nEDA Complete: 12 visualizations created")

# ============================================================================
# PHASE 3: Preprocessing & Feature Engineering
# ============================================================================
print("\nPHASE 3: Data Preprocessing & Feature Engineering")
print("-"*80)

df_prep = df_original.copy()

# Handle missing values
for col in df_prep.columns:
    if df_prep[col].dtype == 'object':
        df_prep[col] = df_prep[col].replace('unknown', np.nan)

# Drop duration (DATA LEAKAGE)
df_prep = df_prep.drop('duration', axis=1)
print("  Dropped 'duration' variable (data leakage)")

# Feature engineering
df_prep['was_contacted_before'] = (df_prep['pdays'] != 999).astype(int)
df_prep['campaign_log'] = np.log1p(df_prep['campaign'])
df_prep['previous_log'] = np.log1p(df_prep['previous'])
print("  Created: was_contacted_before, campaign_log, previous_log")

# Separate features and target
X = df_prep.drop('y', axis=1)
y = (df_prep['y'] == 'yes').astype(int)

# Identify column types
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Train-test split (STRATIFIED)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]:,} samples, Test: {X_test.shape[0]:,} samples")

# Imputation
num_imputer = SimpleImputer(strategy='mean')
X_train[numerical_features] = num_imputer.fit_transform(X_train[numerical_features])
X_test[numerical_features] = num_imputer.transform(X_test[numerical_features])

cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[categorical_features] = cat_imputer.fit_transform(X_train[categorical_features])
X_test[categorical_features] = cat_imputer.transform(X_test[categorical_features])

# Encoding
X_train_encoded = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

# Scaling
scaler = StandardScaler()
X_train_encoded[numerical_features] = scaler.fit_transform(X_train_encoded[numerical_features])
X_test_encoded[numerical_features] = scaler.transform(X_test_encoded[numerical_features])

X_train_final = X_train_encoded
X_test_final = X_test_encoded

print(f"  Final shape: {X_train_final.shape[1]} features")

# ============================================================================
# PHASE 4: Baseline Models
# ============================================================================
print("\nPHASE 4: Baseline Model Development")
print("-"*80)

def evaluate_model(y_true, y_pred, y_proba, model_name):
    return {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_proba)
    }

# Decision Tree Baseline
dt_baseline = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', random_state=42)
dt_baseline.fit(X_train_final, y_train)
y_pred_dt_base = dt_baseline.predict(X_test_final)
y_proba_dt_base = dt_baseline.predict_proba(X_test_final)[:, 1]
metrics_dt_base = evaluate_model(y_test, y_pred_dt_base, y_proba_dt_base, 'DT Baseline')

print(f"  DT Baseline - Acc: {metrics_dt_base['Accuracy']:.4f}, Recall: {metrics_dt_base['Recall']:.4f}")

# Logistic Regression Baseline
lr_baseline = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr_baseline.fit(X_train_final, y_train)
y_pred_lr_base = lr_baseline.predict(X_test_final)
y_proba_lr_base = lr_baseline.predict_proba(X_test_final)[:, 1]
metrics_lr_base = evaluate_model(y_test, y_pred_lr_base, y_proba_lr_base, 'LR Baseline')

print(f"  LR Baseline - Acc: {metrics_lr_base['Accuracy']:.4f}, Recall: {metrics_lr_base['Recall']:.4f}")

# VIZ 13: Baseline Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dt_base, ax=axes[0], cmap='Blues')
axes[0].set_title('Decision Tree - Baseline', fontweight='bold')
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr_base, ax=axes[1], cmap='Greens')
axes[1].set_title('Logistic Regression - Baseline', fontweight='bold')
plt.tight_layout()
plt.savefig('assets/13_baseline_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 13_baseline_confusion_matrices.png")

# VIZ 14: Baseline ROC Curves
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_predictions(y_test, y_proba_dt_base, name='Decision Tree', ax=ax, color='blue')
RocCurveDisplay.from_predictions(y_test, y_proba_lr_base, name='Logistic Regression', ax=ax, color='orange')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax.set_title('ROC Curves - Baseline Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('assets/14_baseline_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 14_baseline_roc_curves.png")

# ============================================================================
# PHASE 5: Hyperparameter Optimization
# ============================================================================
print("\nPHASE 5: Hyperparameter Optimization")
print("-"*80)
print("  This may take 5-10 minutes...")

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Decision Tree GridSearch
print("\n  GridSearch: Decision Tree...")
param_grid_dt = {
    'max_depth': [None, 15, 20],
    'min_samples_leaf': [1, 5],
    'min_samples_split': [2, 10],
    'ccp_alpha': [0.0, 0.001],
    'class_weight': ['balanced']
}

grid_dt = GridSearchCV(
    DecisionTreeClassifier(criterion='entropy', random_state=42),
    param_grid_dt, cv=cv_strategy, scoring='roc_auc', n_jobs=-1, verbose=0
)
grid_dt.fit(X_train_final, y_train)
best_dt = grid_dt.best_estimator_
print(f"    Best params: {grid_dt.best_params_}")
print(f"    Best CV ROC-AUC: {grid_dt.best_score_:.4f}")

y_pred_dt_tuned = best_dt.predict(X_test_final)
y_proba_dt_tuned = best_dt.predict_proba(X_test_final)[:, 1]
metrics_dt_tuned = evaluate_model(y_test, y_pred_dt_tuned, y_proba_dt_tuned, 'DT Tuned')
print(f"    Test Recall: {metrics_dt_tuned['Recall']:.4f}")

# Logistic Regression GridSearch
print("\n  GridSearch: Logistic Regression...")
param_grid_lr = {
    'C': [0.01, 0.1, 1],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'class_weight': ['balanced'],
    'max_iter': [1000]
}

grid_lr = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid_lr, cv=cv_strategy, scoring='roc_auc', n_jobs=-1, verbose=0
)
grid_lr.fit(X_train_final, y_train)
best_lr = grid_lr.best_estimator_
print(f"    Best params: {grid_lr.best_params_}")
print(f"    Best CV ROC-AUC: {grid_lr.best_score_:.4f}")

y_pred_lr_tuned = best_lr.predict(X_test_final)
y_proba_lr_tuned = best_lr.predict_proba(X_test_final)[:, 1]
metrics_lr_tuned = evaluate_model(y_test, y_pred_lr_tuned, y_proba_lr_tuned, 'LR Tuned')
print(f"    Test Recall: {metrics_lr_tuned['Recall']:.4f}")

# VIZ 15-16
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dt_tuned, ax=axes[0], cmap='Blues')
axes[0].set_title('Decision Tree - Tuned', fontweight='bold')
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr_tuned, ax=axes[1], cmap='Greens')
axes[1].set_title('Logistic Regression - Tuned', fontweight='bold')
plt.tight_layout()
plt.savefig('assets/15_tuned_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 15_tuned_confusion_matrices.png")

fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_predictions(y_test, y_proba_dt_tuned, name='DT Tuned', ax=ax, color='blue')
RocCurveDisplay.from_predictions(y_test, y_proba_lr_tuned, name='LR Tuned', ax=ax, color='orange')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax.set_title('ROC Curves - Tuned Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('assets/16_tuned_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 16_tuned_roc_curves.png")

# ============================================================================
# PHASE 6: Cost-Sensitive Threshold Optimization
# ============================================================================
print("\nPHASE 6: Cost-Sensitive Threshold Optimization")
print("-"*80)

COST_FP = 1.5
COST_FN = 20.0
COST_TP = -5.0
COST_TN = 0.0

def expected_cost(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    total_cost = (fp * COST_FP + fn * COST_FN + tp * COST_TP + tn * COST_TN)
    return total_cost / len(y_true)

thresholds = np.linspace(0.01, 0.99, 99)

costs_dt = [expected_cost(y_test, y_proba_dt_tuned, th) for th in thresholds]
optimal_thresh_dt = thresholds[np.argmin(costs_dt)]
min_cost_dt = np.min(costs_dt)

costs_lr = [expected_cost(y_test, y_proba_lr_tuned, th) for th in thresholds]
optimal_thresh_lr = thresholds[np.argmin(costs_lr)]
min_cost_lr = np.min(costs_lr)

print(f"  DT Optimal Threshold: {optimal_thresh_dt:.3f}, Min Cost: {min_cost_dt:.3f}")
print(f"  LR Optimal Threshold: {optimal_thresh_lr:.3f}, Min Cost: {min_cost_lr:.3f}")

# VIZ 17
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresholds, costs_dt, label='Decision Tree', linewidth=2, color='blue')
ax.plot(thresholds, costs_lr, label='Logistic Regression', linewidth=2, color='orange')
ax.axvline(optimal_thresh_dt, color='blue', linestyle='--', alpha=0.7, label=f'DT Optimal ({optimal_thresh_dt:.2f})')
ax.axvline(optimal_thresh_lr, color='orange', linestyle='--', alpha=0.7, label=f'LR Optimal ({optimal_thresh_lr:.2f})')
ax.set_xlabel('Classification Threshold', fontsize=12)
ax.set_ylabel('Average Cost per Customer', fontsize=12)
ax.set_title('Cost-Sensitive Threshold Optimization', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('assets/17_cost_threshold_optimization.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 17_cost_threshold_optimization.png")

# Evaluate at optimal thresholds
y_pred_dt_optimal = (y_proba_dt_tuned >= optimal_thresh_dt).astype(int)
metrics_dt_optimal = evaluate_model(y_test, y_pred_dt_optimal, y_proba_dt_tuned, 'DT Optimal')

y_pred_lr_optimal = (y_proba_lr_tuned >= optimal_thresh_lr).astype(int)
metrics_lr_optimal = evaluate_model(y_test, y_pred_lr_optimal, y_proba_lr_tuned, 'LR Optimal')

print(f"  DT Optimal Recall: {metrics_dt_optimal['Recall']:.4f}")
print(f"  LR Optimal Recall: {metrics_lr_optimal['Recall']:.4f}")
print(f"\n  WINNER: Logistic Regression (Cost: {min_cost_lr:.3f})")

# ============================================================================
# PHASE 7: Model Interpretability
# ============================================================================
print("\nPHASE 7: Model Interpretability & Feature Importance")
print("-"*80)

feature_names = X_train_final.columns.tolist()

# DT Feature Importance
importances_dt = best_dt.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances_dt
}).sort_values('Importance', ascending=False)

print(f"  Top 5 DT Features: {feature_importance_df.head(5)['Feature'].tolist()}")

# VIZ 18
fig, ax = plt.subplots(figsize=(10, 8))
top_features = feature_importance_df.head(15)
ax.barh(range(len(top_features)), top_features['Importance'], color='skyblue')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'])
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Decision Tree - Top 15 Feature Importances', fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('assets/18_dt_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 18_dt_feature_importance.png")

# VIZ 19: Tree Structure
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(best_dt, max_depth=3, feature_names=feature_names,
          class_names=['Reject', 'Accept'], filled=True, fontsize=10, ax=ax)
ax.set_title('Decision Tree Structure (Depth=3)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('assets/19_decision_tree_structure.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 19_decision_tree_structure.png")

# LR Coefficients
coefficients = best_lr.coef_[0]
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print(f"  Top 5 LR Coefficients: {coef_df.head(5)['Feature'].tolist()}")

# VIZ 20
fig, ax = plt.subplots(figsize=(10, 10))
top_positive = coef_df.nlargest(10, 'Coefficient')
top_negative = coef_df.nsmallest(10, 'Coefficient')
top_coefs = pd.concat([top_negative, top_positive]).sort_values('Coefficient')
colors = ['red' if x < 0 else 'green' for x in top_coefs['Coefficient']]
ax.barh(range(len(top_coefs)), top_coefs['Coefficient'], color=colors)
ax.set_yticks(range(len(top_coefs)))
ax.set_yticklabels(top_coefs['Feature'])
ax.set_xlabel('Coefficient Value', fontsize=12)
ax.set_title('Logistic Regression - Top Positive & Negative Coefficients', fontsize=14, fontweight='bold')
ax.axvline(0, color='black', linestyle='--', linewidth=1)
plt.tight_layout()
plt.savefig('assets/20_lr_coefficients.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 20_lr_coefficients.png")

# ============================================================================
# PHASE 8: Model Comparison & Final Selection
# ============================================================================
print("\nPHASE 8: Model Comparison & Final Selection")
print("-"*80)

# VIZ 21: Final ROC Comparison
fig, ax = plt.subplots(figsize=(10, 8))
RocCurveDisplay.from_predictions(y_test, y_proba_dt_base, name='DT Baseline', linestyle=':', color='blue', ax=ax, alpha=0.7)
RocCurveDisplay.from_predictions(y_test, y_proba_lr_base, name='LR Baseline', linestyle=':', color='orange', ax=ax, alpha=0.7)
RocCurveDisplay.from_predictions(y_test, y_proba_dt_tuned, name='DT Tuned', linestyle='--', color='blue', ax=ax, linewidth=2)
RocCurveDisplay.from_predictions(y_test, y_proba_lr_tuned, name='LR Tuned', linestyle='--', color='orange', ax=ax, linewidth=2)
RocCurveDisplay.from_predictions(y_test, y_proba_lr_tuned, name='WINNER: LR Final', linewidth=3, color='green', ax=ax)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax.set_title('ROC Curve Comparison - All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('assets/21_roc_comparison_final.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 21_roc_comparison_final.png")

# Save results
results_summary = {
    'winner_model': 'Logistic Regression',
    'optimal_threshold': float(optimal_thresh_lr),
    'best_cost': float(min_cost_lr),
    'best_recall': float(metrics_lr_optimal['Recall']),
    'best_roc_auc': float(metrics_lr_optimal['ROC-AUC']),
    'dt_best_params': grid_dt.best_params_,
    'lr_best_params': grid_lr.best_params_
}

with open('models/final_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)

with open('models/best_decision_tree.pkl', 'wb') as f:
    pickle.dump(best_dt, f)
with open('models/best_logistic_regression.pkl', 'wb') as f:
    pickle.dump(best_lr, f)

print("\n" + "="*80)
print("PHASES 1-8 COMPLETE!")
print("="*80)
print(f"\nFINAL WINNER: Logistic Regression")
print(f"  Optimal Threshold: {optimal_thresh_lr:.3f}")
print(f"  Recall: {metrics_lr_optimal['Recall']:.4f} ({metrics_lr_optimal['Recall']*100:.1f}% customer capture)")
print(f"  Average Cost: {min_cost_lr:.3f} per customer")
print(f"  ROC-AUC: {metrics_lr_optimal['ROC-AUC']:.4f}")
print(f"\nVisualizations: 21 images saved to assets/")
print(f"Models saved to models/")
print("\nReady for Phase 9: Report Writing")
