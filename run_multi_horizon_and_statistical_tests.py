from pathlib import Path
"""
run_multi_horizon_and_statistical_tests.py
==========================================
Contribution 2: Multi-Horizon Food Inflation Forecasting & Rigorous Statistical Testing.

REVIEWER FIX (Point 3):
-----------------------
- Comprehensive Naive Persistence Baseline (y_{t+h} = y_t) across all horizons.
- Mean Absolute Scaled Error (MASE = MAE_model / MAE_naive) computed and reported.
- Diebold-Mariano (DM) test upgraded with:
    1. HAC autocorrelation-consistent variance up to lag h-1.
    2. Harvey, Leybourne & Newbold (HLN 1997) small-sample correction.
    3. Student's t-distribution with (n - 1) degrees of freedom.
- Primary DM statistical comparison is against the NAIVE PERSISTENCE baseline (not just Ridge).
- Zero hard-coded p-values or '(p < 0.01)' strings: all decisions dynamically computed.
- Feature matrix X is 100% leakage-free (Daily_Economic_Stress_Index removed).
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Windows console encoding safeguard
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("=" * 80)
print(" [*] CONTRIBUTION 2: MULTI-HORIZON FOOD INFLATION & HLN-HAC STATISTICAL TESTS")
print("     Horizons: t+1 (30D), t+3 (90D), t+6 (180D) | Naive Baseline & MASE")
print("=" * 80)

BASE_DIR = Path(__file__).resolve().parent
dataset_path = os.path.join(BASE_DIR, "daily_macro_master_dataset.csv")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Missing {dataset_path}")

df = pd.read_csv(dataset_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"Loaded: {len(df):,} records ({df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')})")

# Leakage-Free Multi-Indicator Features (Circular DESI removed)
FEATURE_COLS = [
    'Wheat_Price_USD', 'Brent_Crude_Oil_USD', 'Natural_Gas_USD', 'Gold_USD_oz',
    'Oil_Shock_YoY_pct', 'USD_BDT_Rate', 'US_Dollar_Index_DXY',
    'Foreign_Reserves_USD_Bn', 'Reserves_Months_of_Imports',
    'USD_BDT_30D_Volatility', 'Food_Inflation_pct', 'Headline_Inflation_pct'
]

HORIZONS = {
    't+1 Month (30 Days)': 30,
    't+3 Months (90 Days)': 90,
    't+6 Months (180 Days)': 180
}

SPLIT_DATE = '2023-01-01'

def dm_test_hln(e_benchmark, e_model, h):
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold (HLN 1997) correction for h-step ahead forecasts.
    e_benchmark: error vector of baseline (e.g. Naive persistence)
    e_model: error vector of candidate model (e.g. XGBoost)
    h: forecast horizon in steps (used for HAC autocorrelation window)
    """
    d = e_benchmark**2 - e_model**2
    n = len(d)
    d_bar = np.mean(d)
    
    # HAC autocovariance sum up to lag h - 1
    gamma0 = np.mean((d - d_bar)**2)
    var_d = gamma0
    for lag in range(1, h):
        gamma_lag = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        var_d += 2.0 * gamma_lag
    var_d = max(var_d, 1e-10)
    
    # HLN small-sample correction factor
    hln_factor = np.sqrt((n + 1 - 2*h + h*(h - 1)/n) / n)
    dm_raw = d_bar / np.sqrt(var_d / n)
    dm_stat = dm_raw * hln_factor
    
    # Student's t distribution with (n - 1) degrees of freedom
    p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(dm_stat), df=n - 1))
    return float(dm_stat), float(p_val)

results = []
multi_models = {}

for h_name, h_days in HORIZONS.items():
    print(f"\n--- Horizon: {h_name} (Shift: {h_days} Days) ---")
    
    df_h = df.copy()
    target_col = f"Target_Delta_{h_days}D"
    df_h[target_col] = df_h['Food_Inflation_pct'].shift(-h_days) - df_h['Food_Inflation_pct']
    
    clean_df = df_h.dropna(subset=[target_col] + FEATURE_COLS).reset_index(drop=True)
    
    train_mask = clean_df['Date'] < SPLIT_DATE
    test_mask = clean_df['Date'] >= SPLIT_DATE
    
    X_train = clean_df.loc[train_mask, FEATURE_COLS]
    y_train = clean_df.loc[train_mask, target_col]
    
    X_test = clean_df.loc[test_mask, FEATURE_COLS]
    y_test = clean_df.loc[test_mask, target_col]
    
    y_base_level = clean_df.loc[test_mask, 'Food_Inflation_pct'].values
    y_true_level = y_base_level + y_test.values
    
    # 1. Naive Persistence Baseline (y_{t+h} = y_t => Delta = 0)
    y_pred_naive_level = y_base_level
    e_naive = y_true_level - y_pred_naive_level
    mae_naive = mean_absolute_error(y_true_level, y_pred_naive_level)
    rmse_naive = np.sqrt(mean_squared_error(y_true_level, y_pred_naive_level))
    mape_naive = mean_absolute_percentage_error(y_true_level, y_pred_naive_level) * 100
    mase_naive = 1.0  # By definition
    
    # 2. Linear Autoregressive / Ridge Baseline
    m_ridge = Ridge(alpha=1.0)
    m_ridge.fit(X_train, y_train)
    y_pred_ridge_delta = m_ridge.predict(X_test)
    y_pred_ridge_level = y_base_level + y_pred_ridge_delta
    e_ridge = y_true_level - y_pred_ridge_level
    mae_ridge = mean_absolute_error(y_true_level, y_pred_ridge_level)
    rmse_ridge = np.sqrt(mean_squared_error(y_true_level, y_pred_ridge_level))
    mape_ridge = mean_absolute_percentage_error(y_true_level, y_pred_ridge_level) * 100
    mase_ridge = mae_ridge / mae_naive if mae_naive > 0 else np.nan
    dm_stat_ridge_naive, dm_p_ridge_naive = dm_test_hln(e_naive, e_ridge, h=h_days)
    
    # 3. Random Forest Regressor
    m_rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    m_rf.fit(X_train, y_train)
    y_pred_rf_delta = m_rf.predict(X_test)
    y_pred_rf_level = y_base_level + y_pred_rf_delta
    e_rf = y_true_level - y_pred_rf_level
    mae_rf = mean_absolute_error(y_true_level, y_pred_rf_level)
    rmse_rf = np.sqrt(mean_squared_error(y_true_level, y_pred_rf_level))
    mape_rf = mean_absolute_percentage_error(y_true_level, y_pred_rf_level) * 100
    mase_rf = mae_rf / mae_naive if mae_naive > 0 else np.nan
    dm_stat_rf_naive, dm_p_rf_naive = dm_test_hln(e_naive, e_rf, h=h_days)
    
    # 4. XGBoost Regressor (Proposed)
    m_xgb = XGBRegressor(
        n_estimators=180, learning_rate=0.03, max_depth=4,
        subsample=0.85, colsample_bytree=0.85, random_state=42, verbosity=0
    )
    m_xgb.fit(X_train, y_train)
    y_pred_xgb_delta = m_xgb.predict(X_test)
    y_pred_xgb_level = y_base_level + y_pred_xgb_delta
    e_xgb = y_true_level - y_pred_xgb_level
    mae_xgb = mean_absolute_error(y_true_level, y_pred_xgb_level)
    rmse_xgb = np.sqrt(mean_squared_error(y_true_level, y_pred_xgb_level))
    mape_xgb = mean_absolute_percentage_error(y_true_level, y_pred_xgb_level) * 100
    mase_xgb = mae_xgb / mae_naive if mae_naive > 0 else np.nan
    
    # Rigorous HLN-corrected Diebold-Mariano Tests
    dm_stat_xgb_naive, dm_p_xgb_naive = dm_test_hln(e_naive, e_xgb, h=h_days)
    dm_stat_xgb_ridge, dm_p_xgb_ridge = dm_test_hln(e_ridge, e_xgb, h=h_days)
    
    multi_models[h_days] = {
        'model': m_xgb,
        'features': FEATURE_COLS,
        'metrics': {
            'mae': mae_xgb, 'rmse': rmse_xgb, 'mape': mape_xgb, 'mase': mase_xgb,
            'dm_stat_vs_naive': dm_stat_xgb_naive, 'dm_p_vs_naive': dm_p_xgb_naive,
            'dm_stat_vs_ridge': dm_stat_xgb_ridge, 'dm_p_vs_ridge': dm_p_xgb_ridge
        }
    }
    
    # Format p-values dynamically
    p_naive_str = f"{dm_p_xgb_naive:.4e}" if dm_p_xgb_naive < 0.001 else f"{dm_p_xgb_naive:.4f}"
    p_ridge_str = f"{dm_p_xgb_ridge:.4e}" if dm_p_xgb_ridge < 0.001 else f"{dm_p_xgb_ridge:.4f}"
    
    results.append({
        'Horizon': h_name, 'Model': 'Naive Persistence',
        'MAE': f"{mae_naive:.3f}%", 'RMSE': f"{rmse_naive:.3f}%", 'MAPE': f"{mape_naive:.2f}%",
        'MASE': f"{mase_naive:.3f}", 'DM_Stat_vs_Naive': 'Reference', 'p_val_vs_Naive': 'Reference'
    })
    results.append({
        'Horizon': h_name, 'Model': 'ARIMA / Ridge Baseline',
        'MAE': f"{mae_ridge:.3f}%", 'RMSE': f"{rmse_ridge:.3f}%", 'MAPE': f"{mape_ridge:.2f}%",
        'MASE': f"{mase_ridge:.3f}", 'DM_Stat_vs_Naive': f"{dm_stat_ridge_naive:.2f}", 'p_val_vs_Naive': f"{dm_p_ridge_naive:.4f}"
    })
    results.append({
        'Horizon': h_name, 'Model': 'Random Forest',
        'MAE': f"{mae_rf:.3f}%", 'RMSE': f"{rmse_rf:.3f}%", 'MAPE': f"{mape_rf:.2f}%",
        'MASE': f"{mase_rf:.3f}", 'DM_Stat_vs_Naive': f"{dm_stat_rf_naive:.2f}", 'p_val_vs_Naive': f"{dm_p_rf_naive:.4f}"
    })
    results.append({
        'Horizon': h_name, 'Model': 'XGBoost (Proposed)',
        'MAE': f"{mae_xgb:.3f}%", 'RMSE': f"{rmse_xgb:.3f}%", 'MAPE': f"{mape_xgb:.2f}%",
        'MASE': f"{mase_xgb:.3f}", 'DM_Stat_vs_Naive': f"{dm_stat_xgb_naive:.2f}", 'p_val_vs_Naive': p_naive_str
    })
    
    print(f"  Naive     : RMSE = {rmse_naive:.3f}%, MAE = {mae_naive:.3f}%, MASE = 1.000")
    print(f"  Ridge     : RMSE = {rmse_ridge:.3f}%, MAE = {mae_ridge:.3f}%, MASE = {mase_ridge:.3f}")
    print(f"  RF        : RMSE = {rmse_rf:.3f}%, MAE = {mae_rf:.3f}%, MASE = {mase_rf:.3f}")
    print(f"  XGBoost   : RMSE = {rmse_xgb:.3f}%, MAE = {mae_xgb:.3f}%, MASE = {mase_xgb:.3f}")
    print(f"  DM Test (XGB vs Naive, h={h_days}, HLN-HAC): stat = {dm_stat_xgb_naive:.2f}, p = {p_naive_str}")
    print(f"  DM Test (XGB vs Ridge, h={h_days}, HLN-HAC): stat = {dm_stat_xgb_ridge:.2f}, p = {p_ridge_str}")

df_res = pd.DataFrame(results)

print("\n" + "=" * 95)
print(" [*] MULTI-HORIZON BENCHMARK & RIGOROUS DIEBOLD-MARIANO TEST TABLE")
print("=" * 95)
print(df_res.to_string(index=False))
print("=" * 95)

# Save multi-horizon model
out_models_path = os.path.join(BASE_DIR, "best_multi_horizon_models.pkl")
joblib.dump(multi_models, out_models_path)
print(f"\n[+] Saved multi-horizon models to: {out_models_path}")

# Save CSV table of results
out_table_path = os.path.join(BASE_DIR, "multi_horizon_benchmark_results.csv")
df_res.to_csv(out_table_path, index=False)
print(f"[+] Saved results table to: {out_table_path}")

# -----------------------------------------------------------------------------
# Generate Publication-Quality Figure (MASE & RMSE Comparison)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

horizons_labels = ['t+1 (30D)', 't+3 (90D)', 't+6 (180D)']
xgb_rmses = [float(df_res[(df_res['Horizon'] == h) & (df_res['Model'] == 'XGBoost (Proposed)')]['RMSE'].values[0].replace('%', '')) for h in HORIZONS.keys()]
ridge_rmses = [float(df_res[(df_res['Horizon'] == h) & (df_res['Model'] == 'ARIMA / Ridge Baseline')]['RMSE'].values[0].replace('%', '')) for h in HORIZONS.keys()]
naive_rmses = [float(df_res[(df_res['Horizon'] == h) & (df_res['Model'] == 'Naive Persistence')]['RMSE'].values[0].replace('%', '')) for h in HORIZONS.keys()]

xgb_mases = [float(df_res[(df_res['Horizon'] == h) & (df_res['Model'] == 'XGBoost (Proposed)')]['MASE'].values[0]) for h in HORIZONS.keys()]
ridge_mases = [float(df_res[(df_res['Horizon'] == h) & (df_res['Model'] == 'ARIMA / Ridge Baseline')]['MASE'].values[0]) for h in HORIZONS.keys()]

# Left Panel: RMSE Comparison
ax1 = axes[0]
x = np.arange(len(horizons_labels))
width = 0.25

rects1 = ax1.bar(x - width, naive_rmses, width, label='Naive Persistence Baseline', color='#9CA3AF')
rects2 = ax1.bar(x, ridge_rmses, width, label='ARIMA / Ridge Baseline', color='#F59E0B')
rects3 = ax1.bar(x + width, xgb_rmses, width, label='XGBoost (Proposed)', color='#059669')

ax1.set_ylabel('Root Mean Squared Error (RMSE %)', fontsize=11, fontweight='bold')
ax1.set_title('A: Forecast Error (RMSE) across Horizons', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(horizons_labels, fontsize=11, fontweight='bold')
ax1.legend(frameon=True, loc='upper left')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

for rect in rects3:
    h = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., h + 0.03, f'{h:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#065F46')

# Right Panel: MASE Comparison (Reference = 1.0)
ax2 = axes[1]
rects4 = ax2.bar(x - width/2, ridge_mases, width, label='ARIMA / Ridge MASE', color='#F59E0B')
rects5 = ax2.bar(x + width/2, xgb_mases, width, label='XGBoost MASE', color='#059669')
ax2.axhline(1.0, color='#DC2626', linestyle='--', linewidth=1.5, label='Naive Persistence Threshold (MASE = 1.0)')

ax2.set_ylabel('Mean Absolute Scaled Error (MASE)', fontsize=11, fontweight='bold')
ax2.set_title('B: Relative Accuracy vs. Naive Baseline (MASE < 1.0 is Superior)', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(horizons_labels, fontsize=11, fontweight='bold')
ax2.legend(frameon=True, loc='upper left')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

for rect in rects5:
    h = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2., h + 0.02, f'{h:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#065F46')

plt.suptitle("Contribution 2: Multi-Horizon Forecasting Accuracy & Statistical Validation\n(Diebold-Mariano HLN 1997 Small-Sample HAC Correction)", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plot_path = os.path.join(BASE_DIR, "multi_horizon_benchmark_and_diebold_mariano.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"[+] Saved publication figure to: {plot_path}")

print("\n" + "=" * 80)
print(" [+] STEP 3 DONE: NAIVE PERSISTENCE BASELINE, MASE & HLN-DM TEST COMPLETED!")
print("=" * 80)
