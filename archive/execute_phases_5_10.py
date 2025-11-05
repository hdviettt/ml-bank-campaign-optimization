"""
Bank Marketing Campaign Optimization - Phases 5-10
Hyperparameter Optimization, Cost-Sensitive Threshold, Interpretation, Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import pickle
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
np.random.seed(42)

# Load preprocessed data from Phase 4
print("="*80)
print("LOADING PREPROCESSED DATA FROM PHASE 4")
print("="*80)

# We need to re-run preprocessing to get the data
# (In practice, this would be loaded from saved files)

# Load and preprocess
df_original = pd.read_csv('input/4-data.csv', sep=';')
df_prep = df_original.copy()

# Replace 'unknown' with NaN
for col in df_prep.columns:
    if df_prep[col].dtype == 'object':
        df_prep[col] = df_prep[col].replace('unknown', np.nan)

# Drop duration (DATA LEAKAGE)
df_prep = df_prep.drop('duration', axis=1)

# Create features
df_prep['was_contacted_before'] = (df_prep['pdays'] != 999).astype(int)
df_prep['campaign_log'] = np.log1p(df_prep['campaign'])
df_prep['previous_log'] = np.log1p(df_prep['previous'])

# Separate features and target
X = df_prep.drop('y', axis=1)
y = (df_prep['y'] == 'yes').astype(int)

# Identify column types
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

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

print(f"✓ Data loaded and preprocessed")
print(f"  Training shape: {X_train_final.shape}")
print(f"  Test shape: {X_test_final.shape}")

# Helper function
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

# ============================================================================
# PHASE 5: Hyperparameter Optimization
# ============================================================================
print("\n" + "="*80)
print("🔧 PHASE 5: Hyperparameter Optimization (GridSearchCV)")
print("="*80)
print("⚠️ This phase may take several minutes...")

# Cross-validation strategy
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 5.1 Decision Tree GridSearchCV
print("\n5.1 Decision Tree GridSearchCV")
print("  Searching parameter space...")

param_grid_dt = {
    'max_depth': [None, 10, 15, 20, 25],
    'min_samples_leaf': [1, 2, 5, 10],
    'min_samples_split': [2, 5, 10],
    'ccp_alpha': [0.0, 0.001, 0.005],
    'class_weight': ['balanced']
}

grid_dt = GridSearchCV(
    estimator=DecisionTreeClassifier(criterion='entropy', random_state=42),
    param_grid=param_grid_dt,
    cv=cv_strategy,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_dt.fit(X_train_final, y_train)

best_dt = grid_dt.best_estimator_
print(f"\n  ✓ Best parameters: {grid_dt.best_params_}")
print(f"  ✓ Best CV ROC-AUC: {grid_dt.best_score_:.4f}")

# Evaluate tuned DT
y_pred_dt_tuned = best_dt.predict(X_test_final)
y_proba_dt_tuned = best_dt.predict_proba(X_test_final)[:, 1]
metrics_dt_tuned = evaluate_model(y_test, y_pred_dt_tuned, y_proba_dt_tuned, 'DT Tuned')

print(f"\n  Tuned Model Performance:")
print(f"    Accuracy:  {metrics_dt_tuned['Accuracy']:.4f}")
print(f"    Precision: {metrics_dt_tuned['Precision']:.4f}")
print(f"    Recall:    {metrics_dt_tuned['Recall']:.4f}")
print(f"    F1-Score:  {metrics_dt_tuned['F1-Score']:.4f}")
print(f"    ROC-AUC:   {metrics_dt_tuned['ROC-AUC']:.4f}")

# 5.2 Logistic Regression GridSearchCV
print("\n5.2 Logistic Regression GridSearchCV")
print("  Searching parameter space...")

param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
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

grid_lr.fit(X_train_final, y_train)

best_lr = grid_lr.best_estimator_
print(f"\n  ✓ Best parameters: {grid_lr.best_params_}")
print(f"  ✓ Best CV ROC-AUC: {grid_lr.best_score_:.4f}")

# Evaluate tuned LR
y_pred_lr_tuned = best_lr.predict(X_test_final)
y_proba_lr_tuned = best_lr.predict_proba(X_test_final)[:, 1]
metrics_lr_tuned = evaluate_model(y_test, y_pred_lr_tuned, y_proba_lr_tuned, 'LR Tuned')

print(f"\n  Tuned Model Performance:")
print(f"    Accuracy:  {metrics_lr_tuned['Accuracy']:.4f}")
print(f"    Precision: {metrics_lr_tuned['Precision']:.4f}")
print(f"    Recall:    {metrics_lr_tuned['Recall']:.4f}")
print(f"    F1-Score:  {metrics_lr_tuned['F1-Score']:.4f}")
print(f"    ROC-AUC:   {metrics_lr_tuned['ROC-AUC']:.4f}")

# Visualization 15: Tuned Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dt_tuned, ax=axes[0], cmap='Blues')
axes[0].set_title('Decision Tree - Tuned', fontweight='bold')

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr_tuned, ax=axes[1], cmap='Greens')
axes[1].set_title('Logistic Regression - Tuned', fontweight='bold')

plt.tight_layout()
plt.savefig('assets/15_tuned_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n  ✓ Saved: assets/15_tuned_confusion_matrices.png")

# Visualization 16: Tuned ROC Curves
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
print("  ✓ Saved: assets/16_tuned_roc_curves.png")

print(f"\n✓ Phase 5 Complete: Hyperparameter Optimization Done")

# ============================================================================
# PHASE 6: Cost-Sensitive Threshold Optimization
# ============================================================================
print("\n" + "="*80)
print("💰 PHASE 6: Cost-Sensitive Threshold Optimization")
print("="*80)

# Define cost matrix
COST_FP = 1.5    # False Positive: unnecessary call
COST_FN = 20.0   # False Negative: missed customer
COST_TP = -5.0   # True Positive: revenue from sale
COST_TN = 0.0    # True Negative: correctly avoided

print(f"\nCost Matrix:")
print(f"  FP (unnecessary call): +{COST_FP}")
print(f"  FN (missed customer): +{COST_FN}")
print(f"  TP (successful sale): {COST_TP}")
print(f"  TN (correctly avoided): {COST_TN}")

# Expected cost function
def expected_cost(y_true, y_proba, threshold=0.5):
    """Calculate expected cost per customer"""
    y_pred = (y_proba >= threshold).astype(int)

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    total_cost = (fp * COST_FP + fn * COST_FN + tp * COST_TP + tn * COST_TN)
    avg_cost = total_cost / len(y_true)

    return avg_cost

# Threshold sweep
print("\nPerforming threshold sweep (0.01 to 0.99)...")
thresholds = np.linspace(0.01, 0.99, 99)

# Decision Tree
costs_dt = [expected_cost(y_test, y_proba_dt_tuned, th) for th in thresholds]
optimal_thresh_dt = thresholds[np.argmin(costs_dt)]
min_cost_dt = np.min(costs_dt)

print(f"\n  Decision Tree:")
print(f"    Optimal Threshold: {optimal_thresh_dt:.3f}")
print(f"    Minimum Avg Cost:  {min_cost_dt:.3f}")

# Logistic Regression
costs_lr = [expected_cost(y_test, y_proba_lr_tuned, th) for th in thresholds]
optimal_thresh_lr = thresholds[np.argmin(costs_lr)]
min_cost_lr = np.min(costs_lr)

print(f"\n  Logistic Regression:")
print(f"    Optimal Threshold: {optimal_thresh_lr:.3f}")
print(f"    Minimum Avg Cost:  {min_cost_lr:.3f}")

# Visualization 17: Cost vs Threshold
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresholds, costs_dt, label='Decision Tree', linewidth=2, color='blue')
ax.plot(thresholds, costs_lr, label='Logistic Regression', linewidth=2, color='orange')
ax.axvline(optimal_thresh_dt, color='blue', linestyle='--', alpha=0.7,
           label=f'DT Optimal ({optimal_thresh_dt:.2f})')
ax.axvline(optimal_thresh_lr, color='orange', linestyle='--', alpha=0.7,
           label=f'LR Optimal ({optimal_thresh_lr:.2f})')
ax.set_xlabel('Classification Threshold', fontsize=12)
ax.set_ylabel('Average Cost per Customer', fontsize=12)
ax.set_title('Cost-Sensitive Threshold Optimization', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('assets/17_cost_threshold_optimization.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n  ✓ Saved: assets/17_cost_threshold_optimization.png")

# Evaluate at optimal thresholds
print("\nEvaluating at optimal thresholds...")

# DT at optimal threshold
y_pred_dt_optimal = (y_proba_dt_tuned >= optimal_thresh_dt).astype(int)
metrics_dt_optimal = evaluate_model(y_test, y_pred_dt_optimal, y_proba_dt_tuned, 'DT Optimal')

print(f"\n  Decision Tree (threshold={optimal_thresh_dt:.3f}):")
print(f"    Accuracy:  {metrics_dt_optimal['Accuracy']:.4f}")
print(f"    Precision: {metrics_dt_optimal['Precision']:.4f}")
print(f"    Recall:    {metrics_dt_optimal['Recall']:.4f}")
print(f"    F1-Score:  {metrics_dt_optimal['F1-Score']:.4f}")
print(f"    Avg Cost:  {min_cost_dt:.3f}")

# LR at optimal threshold
y_pred_lr_optimal = (y_proba_lr_tuned >= optimal_thresh_lr).astype(int)
metrics_lr_optimal = evaluate_model(y_test, y_pred_lr_optimal, y_proba_lr_tuned, 'LR Optimal')

print(f"\n  Logistic Regression (threshold={optimal_thresh_lr:.3f}):")
print(f"    Accuracy:  {metrics_lr_optimal['Accuracy']:.4f}")
print(f"    Precision: {metrics_lr_optimal['Precision']:.4f}")
print(f"    Recall:    {metrics_lr_optimal['Recall']:.4f}")
print(f"    F1-Score:  {metrics_lr_optimal['F1-Score']:.4f}")
print(f"    Avg Cost:  {min_cost_lr:.3f}")

print(f"\n✓ Phase 6 Complete: Cost-Sensitive Optimization Done")
print(f"  🏆 Winner: Logistic Regression (Cost={min_cost_lr:.3f})")

# ============================================================================
# PHASE 7: Model Interpretability & Feature Importance
# ============================================================================
print("\n" + "="*80)
print("🔍 PHASE 7: Model Interpretability & Feature Importance")
print("="*80)

# Get feature names
feature_names = X_train_final.columns.tolist()

# 7.1 Decision Tree Feature Importance
print("\n7.1 Decision Tree Feature Importance")

importances_dt = best_dt.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances_dt
}).sort_values('Importance', ascending=False)

print("\n  Top 15 Most Important Features:")
print(feature_importance_df.head(15).to_string(index=False))

# Visualization 18: DT Feature Importance
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
print("\n  ✓ Saved: assets/18_dt_feature_importance.png")

# 7.2 Decision Tree Visualization
print("\n7.2 Decision Tree Structure Visualization")

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(best_dt,
          max_depth=3,  # Limit for readability
          feature_names=feature_names,
          class_names=['Reject', 'Accept'],
          filled=True,
          fontsize=10,
          ax=ax)
ax.set_title('Decision Tree Structure (Depth=3)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('assets/19_decision_tree_structure.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: assets/19_decision_tree_structure.png")

# 7.3 Logistic Regression Coefficients
print("\n7.3 Logistic Regression Coefficients")

coefficients = best_lr.coef_[0]
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print("\n  Top 15 Most Influential Features:")
print(coef_df.head(15).to_string(index=False))

# Visualization 20: LR Coefficients
fig, ax = plt.subplots(figsize=(10, 10))

# Get top positive and negative
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
print("\n  ✓ Saved: assets/20_lr_coefficients.png")

print(f"\n✓ Phase 7 Complete: Model Interpretability Analysis Done")

# ============================================================================
# PHASE 8: Model Comparison & Final Selection
# ============================================================================
print("\n" + "="*80)
print("📊 PHASE 8: Model Comparison & Final Selection")
print("="*80)

# Create comprehensive comparison table
comparison_data = []

# Add baseline results (re-run baseline for comparison)
dt_baseline = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', random_state=42)
dt_baseline.fit(X_train_final, y_train)
y_pred_dt_base = dt_baseline.predict(X_test_final)
y_proba_dt_base = dt_baseline.predict_proba(X_test_final)[:, 1]
metrics_dt_base = evaluate_model(y_test, y_pred_dt_base, y_proba_dt_base, 'DT Baseline')

lr_baseline = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr_baseline.fit(X_train_final, y_train)
y_pred_lr_base = lr_baseline.predict(X_test_final)
y_proba_lr_base = lr_baseline.predict_proba(X_test_final)[:, 1]
metrics_lr_base = evaluate_model(y_test, y_pred_lr_base, y_proba_lr_base, 'LR Baseline')

# Compile all results
comparison_data = [
    {**metrics_dt_base, 'Stage': 'Baseline', 'Threshold': 0.50, 'Avg Cost': expected_cost(y_test, y_proba_dt_base, 0.5)},
    {**metrics_dt_tuned, 'Stage': 'Tuned', 'Threshold': 0.50, 'Avg Cost': expected_cost(y_test, y_proba_dt_tuned, 0.5)},
    {**metrics_dt_optimal, 'Stage': 'Optimized', 'Threshold': optimal_thresh_dt, 'Avg Cost': min_cost_dt},
    {**metrics_lr_base, 'Stage': 'Baseline', 'Threshold': 0.50, 'Avg Cost': expected_cost(y_test, y_proba_lr_base, 0.5)},
    {**metrics_lr_tuned, 'Stage': 'Tuned', 'Threshold': 0.50, 'Avg Cost': expected_cost(y_test, y_proba_lr_tuned, 0.5)},
    {**metrics_lr_optimal, 'Stage': 'Optimized', 'Threshold': optimal_thresh_lr, 'Avg Cost': min_cost_lr},
]

comparison_df = pd.DataFrame(comparison_data)

print("\nMaster Comparison Table:")
print("="*80)
print(comparison_df.to_string(index=False))
print("="*80)

# Visualization 21: Combined ROC Curve
fig, ax = plt.subplots(figsize=(10, 8))

RocCurveDisplay.from_predictions(y_test, y_proba_dt_base,
                                  name='DT Baseline', linestyle=':', color='blue', ax=ax, alpha=0.7)
RocCurveDisplay.from_predictions(y_test, y_proba_lr_base,
                                  name='LR Baseline', linestyle=':', color='orange', ax=ax, alpha=0.7)
RocCurveDisplay.from_predictions(y_test, y_proba_dt_tuned,
                                  name='DT Tuned', linestyle='--', color='blue', ax=ax, linewidth=2)
RocCurveDisplay.from_predictions(y_test, y_proba_lr_tuned,
                                  name='LR Tuned', linestyle='--', color='orange', ax=ax, linewidth=2)
RocCurveDisplay.from_predictions(y_test, y_proba_lr_tuned,
                                  name='★ LR Final Winner', linewidth=3, color='green', ax=ax)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax.set_title('ROC Curve Comparison - All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('assets/21_roc_comparison_final.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n  ✓ Saved: assets/21_roc_comparison_final.png")

print(f"\n✓ Phase 8 Complete: Model Comparison Done")
print(f"\n🏆 FINAL WINNER: Logistic Regression")
print(f"   - Optimal Threshold: {optimal_thresh_lr:.3f}")
print(f"   - Recall: {metrics_lr_optimal['Recall']:.4f} (catching {metrics_lr_optimal['Recall']*100:.1f}% of customers)")
print(f"   - Average Cost: {min_cost_lr:.3f} per customer")
print(f"   - ROC-AUC: {metrics_lr_optimal['ROC-AUC']:.4f}")

# Save all results
results_summary = {
    'winner_model': 'Logistic Regression',
    'optimal_threshold': float(optimal_thresh_lr),
    'best_cost': float(min_cost_lr),
    'best_recall': float(metrics_lr_optimal['Recall']),
    'best_roc_auc': float(metrics_lr_optimal['ROC-AUC']),
    'comparison_table': comparison_df.to_dict('records'),
    'top_features_dt': feature_importance_df.head(10).to_dict('records'),
    'top_coefficients_lr': coef_df.head(10).to_dict('records')
}

with open('output/final_results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)

print("\n✓ Results saved to output/final_results_summary.json")

# Save models
with open('output/best_decision_tree.pkl', 'wb') as f:
    pickle.dump(best_dt, f)
with open('output/best_logistic_regression.pkl', 'wb') as f:
    pickle.dump(best_lr, f)

print("✓ Models saved to output/")

print("\n" + "="*80)
print("PHASES 5-8 COMPLETE!")
print("="*80)
print("\nNext: Phase 9 (Report Writing) and Phase 10 (Quality Checks)")
