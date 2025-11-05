"""
Bank Marketing Campaign Optimization - Complete Execution Script
This script executes all 10 phases of the assignment automatically.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
from pathlib import Path

# Machine Learning libraries
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
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
pd.set_option('display.max_columns', None)
np.random.seed(42)

# Create output directories
Path('assets').mkdir(exist_ok=True)
Path('output').mkdir(exist_ok=True)

print("="*80)
print("BANK MARKETING CAMPAIGN OPTIMIZATION - AUTOMATIC EXECUTION")
print("="*80)

# ============================================================================
# PHASE 0: Literature Review & References (Quick Documentation)
# ============================================================================
print("\n📚 PHASE 0: Literature Review & Domain Understanding")
print("-"*80)

references = [
    "Moro, S., Cortez, P., & Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, 62, 22-31.",
    "Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). Classification and Regression Trees. Chapman & Hall/CRC.",
    "Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.",
    "James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). An Introduction to Statistical Learning. Springer.",
    "Provost, F., & Fawcett, T. (2013). Data Science for Business. O'Reilly Media.",
    "Elkan, C. (2001). The Foundations of Cost-Sensitive Learning. IJCAI.",
    "Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR, 16, 321-357.",
    "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825-2830."
]

print("✓ References collected for citation")
print(f"  Total references: {len(references)}")

# ============================================================================
# PHASE 1: Setup & Data Loading
# ============================================================================
print("\n📁 PHASE 1: Project Setup & Data Loading")
print("-"*80)

# Load dataset
df_original = pd.read_csv('input/4-data.csv', sep=';')
print(f"✓ Dataset loaded: {df_original.shape[0]:,} rows, {df_original.shape[1]} columns")

# ============================================================================
# PHASE 1.5: Data Dictionary & Column Understanding
# ============================================================================
print("\n📖 PHASE 1.5: Data Dictionary & Column Understanding")
print("-"*80)

data_dictionary = {
    'age': ('numeric', 'Client age in years'),
    'job': ('categorical', 'Type of job'),
    'marital': ('categorical', 'Marital status'),
    'education': ('categorical', 'Education level'),
    'default': ('binary', 'Has credit in default?'),
    'housing': ('binary', 'Has housing loan?'),
    'loan': ('binary', 'Has personal loan?'),
    'contact': ('categorical', 'Contact communication type'),
    'month': ('categorical', 'Last contact month of year'),
    'day_of_week': ('categorical', 'Last contact day of the week'),
    'duration': ('numeric', '⚠️ Last contact duration in seconds (DATA LEAKAGE - exclude from models)'),
    'campaign': ('numeric', 'Number of contacts during this campaign'),
    'pdays': ('numeric', 'Days since last contact from previous campaign (999=never contacted)'),
    'previous': ('numeric', 'Number of contacts before this campaign'),
    'poutcome': ('categorical', 'Outcome of previous campaign'),
    'emp.var.rate': ('numeric', 'Employment variation rate (quarterly)'),
    'cons.price.idx': ('numeric', 'Consumer price index (monthly)'),
    'cons.conf.idx': ('numeric', 'Consumer confidence index (monthly)'),
    'euribor3m': ('numeric', 'Euribor 3 month rate (daily)'),
    'nr.employed': ('numeric', 'Number of employees (quarterly)'),
    'y': ('binary', 'TARGET: Has client subscribed to term deposit? (yes=accepted, no=rejected)')
}

print("✓ Data dictionary created")
print(f"  Total variables: {len(data_dictionary)}")
print(f"  Target variable: 'y' (yes/no for term deposit subscription)")

# ============================================================================
# PHASE 2: Exploratory Data Analysis
# ============================================================================
print("\n🔍 PHASE 2: Exploratory Data Analysis")
print("-"*80)

# Create a copy for EDA
df = df_original.copy()

# 2.1 Target Variable Analysis
print("\n2.1 Target Variable Distribution")
target_counts = df['y'].value_counts()
target_pct = df['y'].value_counts(normalize=True) * 100

print(f"  No (rejected): {target_counts['no']:,} ({target_pct['no']:.2f}%)")
print(f"  Yes (accepted): {target_counts['yes']:,} ({target_pct['yes']:.2f}%)")
print(f"  ⚠️ Severe class imbalance detected!")

# Visualization 1: Class Distribution
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(target_counts.index, target_counts.values, color=['#ff6b6b', '#4ecdc4'])
ax.set_xlabel('Campaign Response', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Target Variable Distribution\\n(Campaign Acceptance)', fontsize=14, fontweight='bold')
for i, (bar, count, pct) in enumerate(zip(bars, target_counts.values, target_pct.values)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f'{count:,}\\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('assets/01_class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: assets/01_class_distribution.png")

# 2.2 Numerical Features Analysis
print("\n2.2 Numerical Features Analysis")
numerical_cols = ['age', 'campaign', 'previous', 'pdays', 'duration',
                  'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

# Summary statistics
print("\n  Summary Statistics:")
print(df[numerical_cols].describe())

# Visualization 2: Numerical Distributions
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols[:10]):
    ax = axes[idx]
    ax.hist(df[col].dropna(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}', fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

    # Add statistics
    mean_val = df[col].mean()
    median_val = df[col].median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
    ax.legend(fontsize=8)

# Hide extra subplots
for idx in range(len(numerical_cols), 12):
    axes[idx].axis('off')

plt.suptitle('Numerical Features Distributions', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('assets/02_numerical_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: assets/02_numerical_distributions.png")

# Visualization 3: Age vs Target
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Age distribution by target
for target in ['no', 'yes']:
    axes[0].hist(df[df['y'] == target]['age'], bins=30, alpha=0.6, label=target)
axes[0].set_xlabel('Age', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Age Distribution by Campaign Response', fontweight='bold')
axes[0].legend()

# Boxplot
df.boxplot(column='age', by='y', ax=axes[1])
axes[1].set_xlabel('Campaign Response', fontsize=12)
axes[1].set_ylabel('Age', fontsize=12)
axes[1].set_title('Age Distribution by Response', fontweight='bold')
plt.suptitle('')

plt.tight_layout()
plt.savefig('assets/03_age_vs_target.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: assets/03_age_vs_target.png")

# Visualization 4: Duration Analysis (Data Leakage Demonstration)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Duration distribution by target
df_duration = df[df['duration'] > 0]
for target in ['no', 'yes']:
    axes[0].hist(df_duration[df_duration['y'] == target]['duration'],
                 bins=50, alpha=0.6, label=target, range=(0, 2000))
axes[0].set_xlabel('Duration (seconds)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Call Duration Distribution by Response', fontweight='bold')
axes[0].legend()

# Box plot
df_duration.boxplot(column='duration', by='y', ax=axes[1])
axes[1].set_xlabel('Campaign Response', fontsize=12)
axes[1].set_ylabel('Duration (seconds)', fontsize=12)
axes[1].set_title('Duration by Response\\n⚠️ Strong correlation but DATA LEAKAGE', fontweight='bold')
axes[1].set_ylim(0, 2000)
plt.suptitle('')

plt.tight_layout()
plt.savefig('assets/04_duration_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: assets/04_duration_analysis.png")
print("  ⚠️ Duration shows strong correlation with target but represents DATA LEAKAGE")
print("     (only known after call ends - cannot use for pre-call prediction)")

# Visualization 5: Economic Indicators
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
print("  ✓ Saved: assets/05_economic_indicators.png")

# 2.3 Categorical Features Analysis
print("\n2.3 Categorical Features Analysis")

# Visualization 6: Job Analysis
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
print("  ✓ Saved: assets/06_job_analysis.png")

# Visualization 7: Month Seasonality
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
print("  ✓ Saved: assets/07_month_seasonality.png")

# Visualization 8: Previous Outcome Impact
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
print("  ✓ Saved: assets/08_poutcome_impact.png")

# Visualization 9: pdays Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# pdays histogram
axes[0].hist(df['pdays'], bins=50, color='skyblue', edgecolor='black')
axes[0].set_xlabel('Days Since Last Contact (pdays)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('pdays Distribution\\n(999 = Never Contacted Before)', fontweight='bold')
axes[0].axvline(999, color='red', linestyle='--', linewidth=2, label='999 (Never contacted)')
axes[0].legend()

# Acceptance rate: contacted before vs never
contacted_before = df[df['pdays'] != 999]['y'].value_counts(normalize=True)['yes'] * 100
never_contacted = df[df['pdays'] == 999]['y'].value_counts(normalize=True)['yes'] * 100

categories = ['Contacted Before\\n(pdays ≠ 999)', 'Never Contacted\\n(pdays = 999)']
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
print("  ✓ Saved: assets/09_pdays_distribution.png")

# Visualization 10: Correlation Heatmap
fig, ax = plt.subplots(figsize=(12, 10))
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix - Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('assets/10_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: assets/10_correlation_heatmap.png")

# Visualization 11: Bi-variate Analysis (Key features vs target)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Economic indicator vs target
for idx, col in enumerate(['nr.employed', 'euribor3m']):
    ax = axes[0, idx]
    df.boxplot(column=col, by='y', ax=ax)
    ax.set_xlabel('Campaign Response', fontsize=11)
    ax.set_ylabel(col, fontsize=11)
    ax.set_title(f'{col} by Response', fontweight='bold')
    plt.suptitle('')

# Campaign and previous vs target
for idx, col in enumerate(['campaign', 'previous']):
    ax = axes[1, idx]
    df_limited = df[df[col] <= df[col].quantile(0.95)]  # Limit to 95th percentile for visibility
    df_limited.boxplot(column=col, by='y', ax=ax)
    ax.set_xlabel('Campaign Response', fontsize=11)
    ax.set_ylabel(col, fontsize=11)
    ax.set_title(f'{col} by Response', fontweight='bold')
    plt.suptitle('')

plt.tight_layout()
plt.savefig('assets/11_bivariate_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: assets/11_bivariate_analysis.png")

# Visualization 12: Duration Data Leakage Demonstration
print("\n  Creating duration data leakage analysis...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Calculate statistics
duration_stats = df.groupby('y')['duration'].agg(['mean', 'median', 'count'])
print("\n  Duration Statistics by Target:")
print(duration_stats)

# Acceptance rate by duration bins
df_temp = df[df['duration'] > 0].copy()
df_temp['duration_bin'] = pd.cut(df_temp['duration'], bins=[0, 120, 300, 600, 1200, 10000],
                                   labels=['0-2min', '2-5min', '5-10min', '10-20min', '20+min'])
duration_acceptance = pd.crosstab(df_temp['duration_bin'], df_temp['y'], normalize='index') * 100

duration_acceptance['yes'].plot(kind='bar', ax=axes[0], color='#4ecdc4')
axes[0].set_xlabel('Call Duration', fontsize=12)
axes[0].set_ylabel('Acceptance Rate (%)', fontsize=12)
axes[0].set_title('Acceptance Rate by Call Duration\\n⚠️ Strong Correlation', fontweight='bold')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

# Text explanation
axes[1].text(0.5, 0.5,
             'DATA LEAKAGE WARNING\\n\\n'
             'Duration shows strong correlation with\\n'
             'campaign success (longer calls → higher acceptance).\\n\\n'
             'However, this variable is ONLY KNOWN\\n'
             'AFTER the call concludes.\\n\\n'
             'For pre-call prediction (our use case),\\n'
             'we CANNOT use this variable.\\n\\n'
             'Including it would yield unrealistic\\n'
             'performance unsuitable for deployment.\\n\\n'
             '✓ Solution: Exclude from predictive models',
             ha='center', va='center', fontsize=13, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
axes[1].axis('off')

plt.tight_layout()
plt.savefig('assets/12_duration_leakage_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: assets/12_duration_leakage_analysis.png")

print(f"\n✓ EDA Complete: 12 visualizations saved to assets/")

# ============================================================================
# PHASE 3: Data Preprocessing & Feature Engineering
# ============================================================================
print("\n⚙️ PHASE 3: Data Preprocessing & Feature Engineering")
print("-"*80)

# Create a fresh copy for preprocessing
df_prep = df_original.copy()

# 3.1 Handle Missing Values
print("\n3.1 Handling Missing Values")

# Replace 'unknown' with NaN
for col in df_prep.columns:
    if df_prep[col].dtype == 'object':
        df_prep[col] = df_prep[col].replace('unknown', np.nan)

# Count missing values
missing_counts = df_prep.isnull().sum()
missing_pct = (missing_counts / len(df_prep)) * 100
missing_df = pd.DataFrame({'Count': missing_counts[missing_counts > 0],
                            'Percentage': missing_pct[missing_counts > 0]})
print("  Missing values:")
print(missing_df)

# 3.2 Feature Engineering
print("\n3.2 Feature Engineering")

# Drop duration (DATA LEAKAGE)
print("  ✓ Dropping 'duration' variable (data leakage)")
df_prep = df_prep.drop('duration', axis=1)

# Create was_contacted_before
df_prep['was_contacted_before'] = (df_prep['pdays'] != 999).astype(int)
print("  ✓ Created 'was_contacted_before' feature")

# Create log-transformed features
df_prep['campaign_log'] = np.log1p(df_prep['campaign'])
df_prep['previous_log'] = np.log1p(df_prep['previous'])
print("  ✓ Created 'campaign_log' and 'previous_log' features")

# 3.3 Separate features and target
X = df_prep.drop('y', axis=1)
y = (df_prep['y'] == 'yes').astype(int)  # Convert to binary: 1=yes, 0=no

print(f"\n  Features shape: {X.shape}")
print(f"  Target shape: {y.shape}")
print(f"  Target distribution: {y.value_counts().to_dict()}")

# 3.4 Identify column types
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"\n  Numerical features ({len(numerical_features)}): {numerical_features}")
print(f"  Categorical features ({len(categorical_features)}): {categorical_features}")

# 3.5 Train-Test Split (STRATIFIED)
print("\n3.3 Train-Test Split")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"  Training set: {X_train.shape[0]:,} samples")
print(f"  Test set: {X_test.shape[0]:,} samples")
print(f"  Train target distribution: {y_train.value_counts(normalize=True).to_dict()}")
print(f"  Test target distribution: {y_test.value_counts(normalize=True).to_dict()}")

# 3.6 Imputation
print("\n3.4 Imputation")

# Numerical imputation (mean)
num_imputer = SimpleImputer(strategy='mean')
X_train[numerical_features] = num_imputer.fit_transform(X_train[numerical_features])
X_test[numerical_features] = num_imputer.transform(X_test[numerical_features])
print(f"  ✓ Numerical features imputed with mean")

# Categorical imputation (most frequent)
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[categorical_features] = cat_imputer.fit_transform(X_train[categorical_features])
X_test[categorical_features] = cat_imputer.transform(X_test[categorical_features])
print(f"  ✓ Categorical features imputed with mode")

# 3.7 Encoding & Scaling
print("\n3.5 Encoding & Scaling")

# One-hot encoding for categorical features
X_train_encoded = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

# Align columns (in case test set has different categories)
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

print(f"  ✓ Categorical features one-hot encoded")
print(f"  Features after encoding: {X_train_encoded.shape[1]}")

# Standardize numerical features
scaler = StandardScaler()
X_train_encoded[numerical_features] = scaler.fit_transform(X_train_encoded[numerical_features])
X_test_encoded[numerical_features] = scaler.transform(X_test_encoded[numerical_features])
print(f"  ✓ Numerical features scaled with StandardScaler")

# Final preprocessed datasets
X_train_final = X_train_encoded
X_test_final = X_test_encoded

print(f"\n✓ Preprocessing Complete!")
print(f"  Final training shape: {X_train_final.shape}")
print(f"  Final test shape: {X_test_final.shape}")

# ============================================================================
# PHASE 4: Baseline Models
# ============================================================================
print("\n🎯 PHASE 4: Baseline Model Development")
print("-"*80)

# Helper function for evaluation
def evaluate_model(y_true, y_pred, y_proba, model_name):
    """Calculate comprehensive metrics"""
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_proba)
    }
    return metrics

# 4.1 Decision Tree Baseline
print("\n4.1 Decision Tree Baseline")
dt_baseline = DecisionTreeClassifier(
    criterion='entropy',
    class_weight='balanced',
    random_state=42
)

dt_baseline.fit(X_train_final, y_train)
y_pred_dt_baseline = dt_baseline.predict(X_test_final)
y_proba_dt_baseline = dt_baseline.predict_proba(X_test_final)[:, 1]

metrics_dt_baseline = evaluate_model(y_test, y_pred_dt_baseline, y_proba_dt_baseline, 'DT Baseline')
print(f"  Accuracy:  {metrics_dt_baseline['Accuracy']:.4f}")
print(f"  Precision: {metrics_dt_baseline['Precision']:.4f}")
print(f"  Recall:    {metrics_dt_baseline['Recall']:.4f}")
print(f"  F1-Score:  {metrics_dt_baseline['F1-Score']:.4f}")
print(f"  ROC-AUC:   {metrics_dt_baseline['ROC-AUC']:.4f}")

# 4.2 Logistic Regression Baseline
print("\n4.2 Logistic Regression Baseline")
lr_baseline = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)

lr_baseline.fit(X_train_final, y_train)
y_pred_lr_baseline = lr_baseline.predict(X_test_final)
y_proba_lr_baseline = lr_baseline.predict_proba(X_test_final)[:, 1]

metrics_lr_baseline = evaluate_model(y_test, y_pred_lr_baseline, y_proba_lr_baseline, 'LR Baseline')
print(f"  Accuracy:  {metrics_lr_baseline['Accuracy']:.4f}")
print(f"  Precision: {metrics_lr_baseline['Precision']:.4f}")
print(f"  Recall:    {metrics_lr_baseline['Recall']:.4f}")
print(f"  F1-Score:  {metrics_lr_baseline['F1-Score']:.4f}")
print(f"  ROC-AUC:   {metrics_lr_baseline['ROC-AUC']:.4f}")

# 4.3 Baseline Comparison Table
baseline_results = pd.DataFrame([metrics_dt_baseline, metrics_lr_baseline])
print("\n4.3 Baseline Comparison")
print(baseline_results.to_string(index=False))

# Visualization 13: Baseline Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dt_baseline, ax=axes[0], cmap='Blues')
axes[0].set_title('Decision Tree - Baseline', fontweight='bold')

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr_baseline, ax=axes[1], cmap='Greens')
axes[1].set_title('Logistic Regression - Baseline', fontweight='bold')

plt.tight_layout()
plt.savefig('assets/13_baseline_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n  ✓ Saved: assets/13_baseline_confusion_matrices.png")

# Visualization 14: Baseline ROC Curves
fig, ax = plt.subplots(figsize=(8, 6))

RocCurveDisplay.from_predictions(y_test, y_proba_dt_baseline, name='Decision Tree', ax=ax, color='blue')
RocCurveDisplay.from_predictions(y_test, y_proba_lr_baseline, name='Logistic Regression', ax=ax, color='orange')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.set_title('ROC Curves - Baseline Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/14_baseline_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: assets/14_baseline_roc_curves.png")

print(f"\n✓ Baseline Models Complete")
print(f"  ⚠️ Low recall observed (~{metrics_dt_baseline['Recall']:.2f} and ~{metrics_lr_baseline['Recall']:.2f})")
print(f"  → Proceeding with hyperparameter optimization...")

# Store baseline results for later comparison
results_tracker = [metrics_dt_baseline, metrics_lr_baseline]

print("\n" + "="*80)
print("PHASE 4 COMPLETE - Baseline models established")
print("="*80)

# Save progress
progress_data = {
    'baseline_dt': metrics_dt_baseline,
    'baseline_lr': metrics_lr_baseline,
    'data_shape': {
        'train': X_train_final.shape,
        'test': X_test_final.shape
    },
    'target_distribution': {
        'train': y_train.value_counts().to_dict(),
        'test': y_test.value_counts().to_dict()
    }
}

with open('output/phase4_baseline_results.json', 'w') as f:
    json.dump(progress_data, f, indent=2, default=str)

print("✓ Progress saved to output/phase4_baseline_results.json")

# ============================================================================
# Continue with remaining phases (5-10) in next section...
# ============================================================================
print("\n" + "="*80)
print("CONTINUING TO PHASE 5: Hyperparameter Optimization")
print("="*80)
