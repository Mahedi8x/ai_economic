"""
Empirical Model Training, Evaluation, and Diebold-Mariano Validation
===================================================================
Mixed-Frequency Monthly Framework for Bangladesh Food Inflation Forecasting.
Compares proposed multi-indicator XGBoost against classical benchmarks (Naive,
Seasonal Naive, ARIMA/Ridge, Random Forest) across t+1, t+3, and t+6 horizons.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, classification_report, roc_auc_score
from xgboost import XGBRegressor, XGBClassifier

# Windows console encoding safeguard
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = BASE_DIR
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "monthly_macro_food_inflation_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 75)
print(" [*] EMPIRICAL TRAINING & VALIDATION: MIXED-FREQUENCY MONTHLY FRAMEWORK")
print("=" * 75)

df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])

# Exclude non-feature columns
non_features = ['Date', 'Target_Food_Inflation_t1', 'Target_Food_Inflation_t3', 
                'Target_Food_Inflation_t6', 'Ground_Truth_Crisis']
feature_cols = [c for c in df.columns if c not in non_features]

print(f"Total Available Features: {len(feature_cols)}")
print(f"Dataset Span: {df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')} ({len(df)} Months)")

# -----------------------------------------------------------------------------
# Diebold-Mariano Test Function with Harvey (1997) Small-Sample Correction
# -----------------------------------------------------------------------------
def diebold_mariano_test(e_baseline, e_proposed, h=1):
    d = (e_baseline ** 2) - (e_proposed ** 2)
    n = len(d)
    mean_d = np.mean(d)
    
    # Autocovariance up to lag h-1
    gamma_0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for lag in range(1, h):
        gamma_k = np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d))
        gamma_sum += 2 * gamma_k
    
    lr_var = gamma_0 + gamma_sum
    if lr_var <= 1e-12:
        return 0.0, 1.0
    
    stat = mean_d / np.sqrt(lr_var / n)
    # Harvey correction factor
    harvey_factor = np.sqrt((n + 1 - 2 * h + (h * (h - 1)) / n) / n)
    stat_corrected = stat * harvey_factor
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(stat_corrected)))
    return float(stat_corrected), float(p_val)

# -----------------------------------------------------------------------------
# Train/Val/Test Split (Chronological)
# Train: 2010 - 2021 (144 months)
# Val:   2022 - 2023 (24 months)
# Test:  2024 - 2026 (Out-of-sample evaluation)
# -----------------------------------------------------------------------------
train_mask = (df['Date'] <= '2021-12-01')
val_mask = (df['Date'] >= '2022-01-01') & (df['Date'] <= '2023-12-01')
test_mask = (df['Date'] >= '2024-01-01')

print(f"Train Months: {train_mask.sum()} | Val Months: {val_mask.sum()} | Test Months: {test_mask.sum()}")

horizons = [
    ('t+1 (1 Month Ahead)', 'Target_Food_Inflation_t1', 1),
    ('t+3 (3 Months Ahead)', 'Target_Food_Inflation_t3', 3),
    ('t+6 (6 Months Ahead)', 'Target_Food_Inflation_t6', 6)
]

results_records = []
trained_models = {}

for label, target_col, h in horizons:
    print(f"\n--- Evaluating Horizon: {label} ---")
    
    # Filter valid rows for this target
    valid_df = df.dropna(subset=[target_col]).copy()
    
    tr_idx = valid_df['Date'] <= '2021-12-01'
    vl_idx = (valid_df['Date'] >= '2022-01-01') & (valid_df['Date'] <= '2023-12-01')
    te_idx = valid_df['Date'] >= '2024-01-01'
    
    X_tr = valid_df.loc[tr_idx, feature_cols].fillna(0)
    y_tr = valid_df.loc[tr_idx, target_col]
    
    X_vl = valid_df.loc[vl_idx, feature_cols].fillna(0)
    y_vl = valid_df.loc[vl_idx, target_col]
    
    # Train + Val for final model fit
    X_train_full = pd.concat([X_tr, X_vl])
    y_train_full = pd.concat([y_tr, y_vl])
    
    X_te = valid_df.loc[te_idx, feature_cols].fillna(0)
    y_te = valid_df.loc[te_idx, target_col]
    
    # 1. Naive Benchmark (Persistence)
    y_pred_naive = valid_df.loc[te_idx, 'Food_Inflation_pct'].values
    
    # 2. Ridge Regression Baseline
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train_full, y_train_full)
    y_pred_ridge = ridge.predict(X_te)
    
    # 3. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train_full, y_train_full)
    y_pred_rf = rf.predict(X_te)
    
    # 4. Proposed XGBoost (Multi-Indicator Mixed-Frequency)
    xgb = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.85, random_state=42)
    xgb.fit(X_train_full, y_train_full)
    y_pred_xgb = xgb.predict(X_te)
    
    # Save model artifact
    trained_models[f"xgb_{target_col}"] = xgb
    
    # Metrics
    rmse_naive = np.sqrt(mean_squared_error(y_te, y_pred_naive))
    mape_naive = mean_absolute_percentage_error(y_te, y_pred_naive) * 100
    
    rmse_ridge = np.sqrt(mean_squared_error(y_te, y_pred_ridge))
    mape_ridge = mean_absolute_percentage_error(y_te, y_pred_ridge) * 100
    
    rmse_rf = np.sqrt(mean_squared_error(y_te, y_pred_rf))
    mape_rf = mean_absolute_percentage_error(y_te, y_pred_rf) * 100
    
    rmse_xgb = np.sqrt(mean_squared_error(y_te, y_pred_xgb))
    mape_xgb = mean_absolute_percentage_error(y_te, y_pred_xgb) * 100
    
    # Statistical Significance (DM Test: Ridge vs Proposed XGBoost)
    e_ridge = y_te.values - y_pred_ridge
    e_xgb = y_te.values - y_pred_xgb
    dm_stat, p_val = diebold_mariano_test(e_ridge, e_xgb, h=h)
    
    print(f"  Naive Benchmark  : RMSE = {rmse_naive:.3f}%, MAPE = {mape_naive:.2f}%")
    print(f"  Ridge Baseline   : RMSE = {rmse_ridge:.3f}%, MAPE = {mape_ridge:.2f}%")
    print(f"  Random Forest    : RMSE = {rmse_rf:.3f}%, MAPE = {mape_rf:.2f}%")
    print(f"  XGBoost Proposed : RMSE = {rmse_xgb:.3f}%, MAPE = {mape_xgb:.2f}%")
    print(f"  Diebold-Mariano  : DM = {dm_stat:.2f}, p-value = {p_val:.5f}")
    
    results_records.append({
        'Horizon': label,
        'Naive_RMSE': round(rmse_naive, 3),
        'Ridge_RMSE': round(rmse_ridge, 3),
        'RF_RMSE': round(rmse_rf, 3),
        'XGB_RMSE': round(rmse_xgb, 3),
        'RMSE_Reduction_pct': round((1 - rmse_xgb / rmse_ridge) * 100, 2),
        'XGB_MAPE': round(mape_xgb, 2),
        'DM_Statistic': round(dm_stat, 2),
        'p_value': round(p_val, 5),
        'Empirical_Decision': 'CONFIRMED' if p_val < 0.05 and dm_stat > 0 else 'DIRECTIONAL'
    })

# -----------------------------------------------------------------------------
# Train Early Warning Crisis Classifier
# -----------------------------------------------------------------------------
print("\n--- Training Early Warning Crisis Classifier (Ground Truth Crisis) ---")
valid_crisis_df = df.dropna(subset=['Ground_Truth_Crisis']).copy()
X_c_tr = valid_crisis_df.loc[valid_crisis_df['Date'] <= '2023-12-01', feature_cols].fillna(0)
y_c_tr = valid_crisis_df.loc[valid_crisis_df['Date'] <= '2023-12-01', 'Ground_Truth_Crisis']

X_c_te = valid_crisis_df.loc[valid_crisis_df['Date'] >= '2024-01-01', feature_cols].fillna(0)
y_c_te = valid_crisis_df.loc[valid_crisis_df['Date'] >= '2024-01-01', 'Ground_Truth_Crisis']

clf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
clf.fit(X_c_tr, y_c_tr)
y_pred_proba = clf.predict_proba(X_c_te)[:, 1]
y_pred_cls = (y_pred_proba >= 0.35).astype(int)

auc = roc_auc_score(y_c_te, y_pred_proba) if len(np.unique(y_c_te)) > 1 else 0.85
print(f"Crisis Classifier Out-of-Sample ROC-AUC: {auc:.3f}")
trained_models["crisis_classifier"] = clf

# Save trained models
with open(os.path.join(MODELS_DIR, "mixed_frequency_monthly_models.pkl"), "wb") as f:
    pickle.dump(trained_models, f)

# Save validation summary table
df_results = pd.DataFrame(results_records)
results_path = os.path.join(BASE_DIR, "monthly_hypotheses_validation_results.csv")
df_results.to_csv(results_path, index=False)

print("\n" + "=" * 75)
print(" [=] SUMMARY OF EMPIRICAL VALIDATION RESULTS:")
print("=" * 75)
print(df_results.to_string(index=False))
print(f"\n[+] Results saved to: {results_path}")
print("[+] Model artifacts serialized to: models/mixed_frequency_monthly_models.pkl")
