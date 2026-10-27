"""
run_all_hypotheses_validation.py
=================================
100% DYNAMIC, LEAKAGE-FREE Hypothesis & RQ Validation Suite.

REVIEWER FIX (Point 1):
-----------------------
BEFORE (Circular / Fabricated):
  - Crisis label = DESI_t+30 >= 45   (DESI is a formula of inflation+reserve+FX+oil)
  - Features X included Daily_Economic_Stress_Index + all 4 of its inputs
  - Model was reverse-engineering a math formula => "93% recall" was meaningless
  - 'CONFIRMED' and p-values were hard-coded strings regardless of computed result

AFTER (Honest / Leakage-Free):
  - Crisis label = REALIZED OUTCOME: did BDT depreciate >=3% OR reserves drop >=5%
    in the next 30 days? (observed market fact, not derived from a formula)
  - Features X: only externally observed market/macro signals
    REMOVED: Daily_Economic_Stress_Index, Economic_Risk_Score,
             IF_Anomaly_Flag, CUSUM_Anomaly_Flag, Ground_Truth_Crisis
  - Naive persistence baseline added to every regression comparison
  - DM test uses h=horizon (HAC-corrected), with HLN small-sample correction
  - 'CONFIRMED'/'REJECTED' computed from actual p-value < 0.05 threshold
  - Anomaly detection: IsolationForest fitted on TRAIN data only (pre-2022),
    evaluated on 2022+ out-of-sample. Lead time measured from crisis START, not peak.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             mean_absolute_percentage_error,
                             precision_score, recall_score, f1_score)
from xgboost import XGBClassifier, XGBRegressor

# Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("=" * 70)
print("  LEAKAGE-FREE HYPOTHESIS & RQ VALIDATION SUITE (Post-Reviewer Fix)")
print("  Reviewer Point 1: Circular label + hard-coded results REMOVED")
print("=" * 70)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "daily_macro_master_dataset.csv")

df = pd.read_csv(dataset_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"\nLoaded: {len(df):,} rows | "
      f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")

# =============================================================================
# LEAKAGE-FREE FEATURE SET
# Removed: Daily_Economic_Stress_Index (label component => circular)
#          Economic_Risk_Score         (derived formula column in raw file)
#          IF_Anomaly_Flag             (fit on full data including test)
#          CUSUM_Anomaly_Flag          (same problem)
#          Ground_Truth_Crisis         (the label itself)
#          Day_of_Week                 (not a macro signal)
# =============================================================================
FEATURE_COLS = [
    # Global commodity & energy
    'Wheat_Price_USD',
    'Brent_Crude_Oil_USD',
    'Natural_Gas_USD',
    'Gold_USD_oz',
    'Oil_Shock_YoY_pct',
    # FX & external
    'USD_BDT_Rate',
    'US_Dollar_Index_DXY',
    'USD_BDT_30D_Volatility',
    'USD_BDT_YoY_Return_pct',
    'DXY_30D_MA',
    # Domestic macro (official published, not derived)
    'Foreign_Reserves_USD_Bn',
    'Reserves_Months_of_Imports',
    'Reserves_to_Short_Term_Debt_Ratio',
    'Remittance_USD_Bn',
    # Inflation (published BBS/BB monthly official values)
    'Food_Inflation_pct',
    'Headline_Inflation_pct',
]

SPLIT_DATE  = '2023-01-01'   # Chronological train/test split
HORIZON_30  = 30
HORIZON_90  = 90
HORIZON_180 = 180

# Verify all feature columns exist
missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    print(f"  [WARNING] Missing feature columns: {missing}")
    FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

print(f"  Feature set: {len(FEATURE_COLS)} leakage-free columns")
print(f"  Removed:     Daily_Economic_Stress_Index, Economic_Risk_Score,")
print(f"               IF_Anomaly_Flag, CUSUM_Anomaly_Flag, Ground_Truth_Crisis")

# =============================================================================
# REALIZED OUTCOME CRISIS LABEL (Leakage-Free)
# Crisis = BDT depreciated >=3% OR Reserves dropped >=5% in next 30 days
# This is an observed market fact — NOT derived from any formula
# =============================================================================
df['BDT_Return_30D_Fwd']      = (df['USD_BDT_Rate'].shift(-HORIZON_30) - df['USD_BDT_Rate']) / df['USD_BDT_Rate'] * 100
df['Reserve_Change_30D_Fwd']  = (df['Foreign_Reserves_USD_Bn'].shift(-HORIZON_30) - df['Foreign_Reserves_USD_Bn']) / df['Foreign_Reserves_USD_Bn'] * 100

df['Crisis_Label_Realized'] = (
    (df['BDT_Return_30D_Fwd']     >= 3.0) |   # BDT weakened >=3% in 30 days
    (df['Reserve_Change_30D_Fwd'] <= -5.0)     # Reserves fell >=5% in 30 days
).astype(int)

crisis_rate = df['Crisis_Label_Realized'].mean() * 100
print(f"\n  Realized crisis label: {df['Crisis_Label_Realized'].sum()} days flagged "
      f"({crisis_rate:.1f}% of dataset) — based on actual BDT/reserve moves")

# =============================================================================
# HLN-corrected Diebold-Mariano test (Harvey, Leybourne & Newbold 1997)
# Uses h-lag HAC variance, small-sample t-distribution correction
# =============================================================================
def dm_test_hln(e1, e2, h):
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold small-sample correction.
    e1, e2: forecast error arrays (actual - predicted)
    h: forecast horizon (used for HAC lag window)
    Returns: dm_stat (HLN-corrected), p_value (two-sided t-test)
    """
    d = e1**2 - e2**2
    n   = len(d)
    d_bar = np.mean(d)
    # HAC variance with h-1 lags
    gamma0 = np.mean((d - d_bar)**2)
    var_d  = gamma0
    for lag in range(1, h):
        gamma_lag = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        var_d += 2.0 * gamma_lag
    var_d = max(var_d, 1e-10)
    dm_raw = d_bar / np.sqrt(var_d / n)
    # HLN correction factor
    hln_factor = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    dm_stat = dm_raw * hln_factor
    # Use t-distribution with n-1 df
    p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(dm_stat), df=n-1))
    return float(dm_stat), float(p_val)

def decision(p_val, alpha=0.05):
    """Honest decision: computed from actual p-value, never hard-coded."""
    return "CONFIRMED (p < 0.05)" if p_val < alpha else f"NOT CONFIRMED (p = {p_val:.4f})"

records = []

# =============================================================================
# RQ1 / H1 — XGBoost vs Ridge vs Naive Persistence (30-Day Food Inflation)
# =============================================================================
print("\n[1/5] RQ1 / H1 — Food Inflation 30-Day Forecast ...")
df_30 = df.copy()
df_30['Target_30D_Delta'] = df_30['Food_Inflation_pct'].shift(-HORIZON_30) - df_30['Food_Inflation_pct']
clean_30 = df_30.dropna(subset=['Target_30D_Delta'] + FEATURE_COLS).reset_index(drop=True)

tr = clean_30['Date'] < SPLIT_DATE
te = clean_30['Date'] >= SPLIT_DATE

X_tr, y_tr = clean_30.loc[tr, FEATURE_COLS], clean_30.loc[tr, 'Target_30D_Delta']
X_te, y_te = clean_30.loc[te, FEATURE_COLS], clean_30.loc[te, 'Target_30D_Delta']
base_te     = clean_30.loc[te, 'Food_Inflation_pct'].values
y_true_30   = base_te + y_te.values

# Naive Persistence: predict zero change (y_t+30 = y_t)
pred_naive_30   = base_te.copy()
e_naive         = y_true_30 - pred_naive_30
rmse_naive      = np.sqrt(mean_squared_error(y_true_30, pred_naive_30))

# Ridge Baseline
m_ridge         = Ridge(alpha=1.0).fit(X_tr, y_tr)
pred_ridge_30   = base_te + m_ridge.predict(X_te)
e_ridge         = y_true_30 - pred_ridge_30
rmse_ridge      = np.sqrt(mean_squared_error(y_true_30, pred_ridge_30))

# XGBoost
m_xgb = XGBRegressor(n_estimators=180, learning_rate=0.03, max_depth=4,
                      subsample=0.85, colsample_bytree=0.85, random_state=42,
                      verbosity=0)
m_xgb.fit(X_tr, y_tr)
pred_xgb_30   = base_te + m_xgb.predict(X_te)
e_xgb         = y_true_30 - pred_xgb_30
rmse_xgb      = np.sqrt(mean_squared_error(y_true_30, pred_xgb_30))
mape_xgb      = mean_absolute_percentage_error(y_true_30, pred_xgb_30) * 100

# MASE (vs naive)
mae_xgb      = mean_absolute_error(y_true_30, pred_xgb_30)
mae_naive_in = mean_absolute_error(y_true_30, pred_naive_30)
mase_xgb     = mae_xgb / mae_naive_in if mae_naive_in > 0 else np.nan

# DM: XGBoost vs Naive (the relevant comparison — not vs Ridge)
dm_stat_xgb_naive, dm_p_xgb_naive = dm_test_hln(e_naive, e_xgb, h=HORIZON_30)
# DM: XGBoost vs Ridge
dm_stat_xgb_ridge, dm_p_xgb_ridge = dm_test_hln(e_ridge, e_xgb, h=HORIZON_30)

# Pct improvement over naive
pct_imp_naive = (rmse_naive - rmse_xgb) / rmse_naive * 100

print(f"  Naive  RMSE = {rmse_naive:.3f}%")
print(f"  Ridge  RMSE = {rmse_ridge:.3f}%")
print(f"  XGBoost RMSE = {rmse_xgb:.3f}%  MAPE = {mape_xgb:.2f}%  MASE vs Naive = {mase_xgb:.3f}")
print(f"  DM(XGB vs Naive):  stat = {dm_stat_xgb_naive:.2f}, p = {dm_p_xgb_naive:.4e}")
print(f"  DM(XGB vs Ridge):  stat = {dm_stat_xgb_ridge:.2f}, p = {dm_p_xgb_ridge:.4e}")
print(f"  Improvement over Naive: {pct_imp_naive:.1f}%")

records.append({
    'RQ': 'RQ1 / H1',
    'Investigation': 'XGBoost vs Naive Persistence & Ridge (30-Day Food Inflation)',
    'Naive_RMSE':   f'{rmse_naive:.3f}%',
    'Ridge_RMSE':   f'{rmse_ridge:.3f}%',
    'XGB_RMSE':     f'{rmse_xgb:.3f}%',
    'XGB_MAPE':     f'{mape_xgb:.2f}%',
    'MASE_vs_Naive': f'{mase_xgb:.3f}',
    'DM_stat_vs_Naive': f'{dm_stat_xgb_naive:.2f}',
    'p_vs_Naive':   f'{dm_p_xgb_naive:.4e}',
    'DM_stat_vs_Ridge': f'{dm_stat_xgb_ridge:.2f}',
    'p_vs_Ridge':   f'{dm_p_xgb_ridge:.4e}',
    'Improvement_over_Naive_pct': f'{pct_imp_naive:.1f}%',
    'Decision':     decision(dm_p_xgb_naive),
})

# =============================================================================
# RQ2 / H2 — Multi-Horizon Forecast Accuracy (30, 90, 180 Days)
# =============================================================================
print("\n[2/5] RQ2 / H2 — Multi-Horizon Comparison ...")

horizon_results = {}
for H, h_name in [(HORIZON_30, '30D'), (HORIZON_90, '90D'), (HORIZON_180, '180D')]:
    df_h = df.copy()
    df_h[f'Target_{h_name}'] = df_h['Food_Inflation_pct'].shift(-H) - df_h['Food_Inflation_pct']
    clean_h = df_h.dropna(subset=[f'Target_{h_name}'] + FEATURE_COLS).reset_index(drop=True)

    tr_h = clean_h['Date'] < SPLIT_DATE
    te_h = clean_h['Date'] >= SPLIT_DATE

    X_tr_h, y_tr_h = clean_h.loc[tr_h, FEATURE_COLS], clean_h.loc[tr_h, f'Target_{h_name}']
    X_te_h, y_te_h = clean_h.loc[te_h, FEATURE_COLS], clean_h.loc[te_h, f'Target_{h_name}']
    base_te_h       = clean_h.loc[te_h, 'Food_Inflation_pct'].values
    y_true_h        = base_te_h + y_te_h.values

    pred_naive_h  = base_te_h.copy()
    rmse_naive_h  = np.sqrt(mean_squared_error(y_true_h, pred_naive_h))
    e_naive_h     = y_true_h - pred_naive_h

    m_h = XGBRegressor(n_estimators=180, learning_rate=0.03, max_depth=4,
                       subsample=0.85, colsample_bytree=0.85, random_state=42,
                       verbosity=0)
    m_h.fit(X_tr_h, y_tr_h)
    pred_xgb_h   = base_te_h + m_h.predict(X_te_h)
    e_xgb_h      = y_true_h - pred_xgb_h
    rmse_xgb_h   = np.sqrt(mean_squared_error(y_true_h, pred_xgb_h))
    mae_xgb_h    = mean_absolute_error(y_true_h, pred_xgb_h)
    mae_naive_h  = mean_absolute_error(y_true_h, pred_naive_h)
    mase_h       = mae_xgb_h / mae_naive_h if mae_naive_h > 0 else np.nan

    dm_h, p_h = dm_test_hln(e_naive_h, e_xgb_h, h=H)

    horizon_results[h_name] = {
        'RMSE_Naive': rmse_naive_h, 'RMSE_XGB': rmse_xgb_h,
        'MASE': mase_h, 'DM': dm_h, 'p': p_h
    }
    print(f"  {h_name}: Naive RMSE={rmse_naive_h:.3f}%  XGB RMSE={rmse_xgb_h:.3f}%  "
          f"MASE={mase_h:.3f}  DM={dm_h:.2f}  p={p_h:.4e}")

records.append({
    'RQ': 'RQ2 / H2',
    'Investigation': 'Multi-horizon XGBoost accuracy vs Naive Persistence (30/90/180 days)',
    '30D_RMSE_XGB':  f"{horizon_results['30D']['RMSE_XGB']:.3f}%",
    '90D_RMSE_XGB':  f"{horizon_results['90D']['RMSE_XGB']:.3f}%",
    '180D_RMSE_XGB': f"{horizon_results['180D']['RMSE_XGB']:.3f}%",
    '30D_MASE':      f"{horizon_results['30D']['MASE']:.3f}",
    '90D_MASE':      f"{horizon_results['90D']['MASE']:.3f}",
    '180D_MASE':     f"{horizon_results['180D']['MASE']:.3f}",
    '30D_p':         f"{horizon_results['30D']['p']:.4e}",
    '90D_p':         f"{horizon_results['90D']['p']:.4e}",
    '180D_p':        f"{horizon_results['180D']['p']:.4e}",
    'Decision':      decision(horizon_results['30D']['p']),
})

# =============================================================================
# RQ3 / H3 — SHAP Feature Importance (Top-3 concentration)
# =============================================================================
print("\n[3/5] RQ3 / H3 — SHAP Feature Importance ...")
try:
    import shap
    # Use 30-Day model trained above (m_xgb), test set
    clean_30_te = clean_30[clean_30['Date'] >= SPLIT_DATE].reset_index(drop=True)
    X_shap = clean_30_te[FEATURE_COLS]
    explainer   = shap.TreeExplainer(m_xgb)
    shap_values = explainer.shap_values(X_shap)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    total_shap    = mean_abs_shap.sum()
    shap_df = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'Mean_Abs_SHAP': mean_abs_shap,
        'Pct_Importance': mean_abs_shap / total_shap * 100
    }).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)

    top3_pct = shap_df['Pct_Importance'].iloc[:3].sum()
    top_feat = shap_df['Feature'].iloc[0]
    print(f"  Top feature: {top_feat} ({shap_df['Pct_Importance'].iloc[0]:.1f}%)")
    print(f"  Top-3 concentration: {top3_pct:.1f}%")
    print(shap_df[['Feature','Pct_Importance']].head(8).to_string(index=False))

    # Save SHAP importance
    shap_csv = os.path.join(BASE_DIR, "shap_feature_importance_leakage_free.csv")
    shap_df.to_csv(shap_csv, index=False)

    records.append({
        'RQ': 'RQ3 / H3',
        'Investigation': 'Top-3 SHAP feature concentration (leakage-free feature set)',
        'Top_Feature': top_feat,
        'Top1_pct': f"{shap_df['Pct_Importance'].iloc[0]:.1f}%",
        'Top3_pct': f"{top3_pct:.1f}%",
        'Decision': "CONFIRMED (top-3 > 40%)" if top3_pct >= 40.0 else f"Top-3 = {top3_pct:.1f}% (< 40%)",
    })
except ImportError:
    print("  [SKIP] shap package not installed. Run: pip install shap")
    records.append({'RQ': 'RQ3 / H3', 'Decision': 'SHAP skipped (pip install shap)'})

# =============================================================================
# RQ4 / H4 — Realized Crisis Label: XGBoost Classifier vs Naive All-Stable
# (No DESI in features — completely leakage-free)
# =============================================================================
print("\n[4/5] RQ4 / H4 — Leakage-Free Crisis Classification ...")
df_clf = df.copy()
clf_cols = ['Crisis_Label_Realized'] + FEATURE_COLS
clean_clf = df_clf.dropna(subset=clf_cols).reset_index(drop=True)

tr_clf = clean_clf['Date'] < SPLIT_DATE
te_clf = clean_clf['Date'] >= SPLIT_DATE

X_tr_clf = clean_clf.loc[tr_clf, FEATURE_COLS]
y_tr_clf = clean_clf.loc[tr_clf, 'Crisis_Label_Realized']
X_te_clf = clean_clf.loc[te_clf, FEATURE_COLS]
y_te_clf = clean_clf.loc[te_clf, 'Crisis_Label_Realized']

scale_pos = max(1, (y_tr_clf == 0).sum() / max(1, (y_tr_clf == 1).sum()))
clf_xgb = XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=scale_pos,
                         random_state=42, verbosity=0, eval_metric='logloss')
clf_xgb.fit(X_tr_clf, y_tr_clf)
pred_clf = clf_xgb.predict(X_te_clf)

# Naive baseline: always predict 0 (stable)
pred_naive_clf = np.zeros(len(y_te_clf), dtype=int)

prec_xgb  = precision_score(y_te_clf, pred_clf, zero_division=0)
rec_xgb   = recall_score(y_te_clf, pred_clf, zero_division=0)
f1_xgb    = f1_score(y_te_clf, pred_clf, zero_division=0)
f1_naive  = f1_score(y_te_clf, pred_naive_clf, zero_division=0)

crisis_rate_te = y_te_clf.mean() * 100
print(f"  Test set crisis rate: {crisis_rate_te:.1f}%")
print(f"  Naive F1 (always stable): {f1_naive:.3f}")
print(f"  XGBoost: Precision={prec_xgb:.3f}  Recall={rec_xgb:.3f}  F1={f1_xgb:.3f}")

# McNemar test for classifier comparison (no statsmodels dependency)
# Using chi-squared approximation with continuity correction
correct_xgb   = (pred_clf == y_te_clf.values).astype(int)
correct_naive = (pred_naive_clf == y_te_clf.values).astype(int)
b = int(((correct_xgb == 1) & (correct_naive == 0)).sum())  # XGB right, naive wrong
c = int(((correct_xgb == 0) & (correct_naive == 1)).sum())  # XGB wrong, naive right
try:
    # McNemar with continuity correction: chi2 = (|b-c| - 1)^2 / (b+c)
    if (b + c) > 0:
        chi2 = (abs(b - c) - 1.0)**2 / (b + c)
        mc_p = float(1.0 - stats.chi2.cdf(chi2, df=1))
    else:
        mc_p = 1.0
except Exception:
    mc_p = float('nan')
print(f"  McNemar p-value (XGB vs Naive): {mc_p:.4e}  [b={b}, c={c}]")

records.append({
    'RQ': 'RQ4 / H4',
    'Investigation': 'Realized outcome crisis classification (leakage-free label)',
    'Naive_F1':     f'{f1_naive:.3f}',
    'XGB_Precision': f'{prec_xgb:.3f}',
    'XGB_Recall':   f'{rec_xgb:.3f}',
    'XGB_F1':       f'{f1_xgb:.3f}',
    'McNemar_p':    f'{mc_p:.4e}',
    'Decision':     decision(mc_p) if not np.isnan(mc_p) else 'McNemar inconclusive',
})

# =============================================================================
# RQ5 — Anomaly Detection (Train-only fit, out-of-sample eval, start-based lead)
# =============================================================================
print("\n[5/5] RQ5 — Anomaly Detection (Train-Only Fit, OOS Evaluation) ...")

# Leakage-free anomaly features: no DESI, no formula-derived columns
ANOM_FEATURES = [
    'USD_BDT_Rate', 'USD_BDT_30D_Volatility', 'USD_BDT_YoY_Return_pct',
    'Brent_Crude_Oil_USD', 'Oil_Shock_YoY_pct',
    'Foreign_Reserves_USD_Bn', 'Reserves_Months_of_Imports',
    'Food_Inflation_pct', 'Headline_Inflation_pct',
    'Wheat_Price_USD',
]

TRAIN_CUTOFF = '2022-01-01'   # Fit only on pre-2022 data
EVAL_START   = '2022-01-01'   # Evaluate on 2022 onwards (OOS)

df_anom = df.copy()
tr_anom = df_anom['Date'] < TRAIN_CUTOFF

X_anom_tr = df_anom.loc[tr_anom, ANOM_FEATURES].ffill().bfill()
X_anom_all = df_anom[ANOM_FEATURES].ffill().bfill()

# Fit ONLY on training data (pre-2022)
contamination_rate = tr_anom.sum() * 0.07 / len(df_anom)  # relative to full
iso = IsolationForest(n_estimators=300, contamination=0.07,
                      random_state=42)
iso.fit(X_anom_tr)

# Predict on ALL data (flag = -1 means anomaly)
df_anom['IF_Flag_OOS'] = (iso.predict(X_anom_all) == -1).astype(int)

# CUSUM on training stats only (mu, sigma from train)
stress_all  = df_anom['Food_Inflation_pct'].values
train_mean  = np.mean(stress_all[tr_anom])
train_std   = np.std(stress_all[tr_anom])
if train_std == 0: train_std = 1.0
norm_s = (stress_all - train_mean) / train_std

s_pos = np.zeros(len(norm_s))
cusum_flag = np.zeros(len(norm_s), dtype=int)
K = 0.5    # slack parameter
H = 5.0    # threshold
for t in range(1, len(norm_s)):
    s_pos[t] = max(0, s_pos[t-1] + norm_s[t] - K)
    # CUSUM reset after flag (reviewer note: no reset was a bug)
    if s_pos[t] > H:
        cusum_flag[t] = 1
        s_pos[t] = 0   # reset

df_anom['CUSUM_Flag_OOS'] = cusum_flag
df_anom['Combined_Flag_OOS'] = ((df_anom['IF_Flag_OOS'] == 1) | (df_anom['CUSUM_Flag_OOS'] == 1)).astype(int)

# Crisis episodes: Lead time measured from CRISIS START (not peak)
CRISIS_EPISODES = [
    {'name': '2011-12 BDT Squeeze',      'start': '2011-11-01', 'end': '2012-08-31'},
    {'name': '2020 COVID Supply Freeze',  'start': '2020-03-15', 'end': '2020-08-15'},
    {'name': '2022-23 Ukraine Forex Drain','start': '2022-04-15', 'end': '2023-11-30'},
    {'name': '2024 Political Disruption', 'start': '2024-06-25', 'end': '2024-11-15'},
]

df_anom['Ground_Truth_OOS'] = 0
for ep in CRISIS_EPISODES:
    mask = (df_anom['Date'] >= ep['start']) & (df_anom['Date'] <= ep['end'])
    df_anom.loc[mask, 'Ground_Truth_OOS'] = 1

# Evaluate on OOS period only (2022+)
eval_mask = df_anom['Date'] >= EVAL_START
df_eval   = df_anom[eval_mask].copy()

prec_oos = precision_score(df_eval['Ground_Truth_OOS'], df_eval['Combined_Flag_OOS'], zero_division=0)
rec_oos  = recall_score(df_eval['Ground_Truth_OOS'], df_eval['Combined_Flag_OOS'], zero_division=0)
f1_oos   = f1_score(df_eval['Ground_Truth_OOS'], df_eval['Combined_Flag_OOS'], zero_division=0)

# Lead time: days from crisis START to first alert (not peak)
lead_times = []
detected   = []
for ep in CRISIS_EPISODES:
    c_start = pd.to_datetime(ep['start'])
    if c_start < pd.to_datetime(EVAL_START):
        continue  # skip episodes before OOS period

    # Look for alert in 90-day window BEFORE crisis start
    pre_window = (df_anom['Date'] >= (c_start - pd.Timedelta(days=90))) & \
                 (df_anom['Date'] <  c_start)
    early_flags = df_anom[pre_window & (df_anom['IF_Flag_OOS'] == 1)]

    if len(early_flags) > 0:
        first_alert = early_flags['Date'].min()
        lead_days   = (c_start - first_alert).days
        lead_months = lead_days / 30.0
        lead_times.append(lead_months)
        detected.append(ep['name'])
        print(f"    DETECTED: {ep['name']} | First alert: {first_alert.strftime('%Y-%m-%d')} | "
              f"Lead: {lead_months:.1f} months before START")
    else:
        print(f"    MISSED:   {ep['name']} (no alert in 90-day pre-window)")

avg_lead = np.mean(lead_times) if lead_times else 0.0
oos_episodes = [ep for ep in CRISIS_EPISODES if pd.to_datetime(ep['start']) >= pd.to_datetime(EVAL_START)]
recall_ep = len(detected) / len(oos_episodes) if oos_episodes else 0.0

print(f"\n  OOS Precision={prec_oos:.3f}  Recall={rec_oos:.3f}  F1={f1_oos:.3f}")
print(f"  Episode-level recall: {len(detected)}/{len(oos_episodes)} = {recall_ep*100:.0f}%")
print(f"  Avg lead time (from crisis START): {avg_lead:.1f} months")

records.append({
    'RQ': 'RQ5',
    'Investigation': 'Anomaly detection OOS (2022+): IsoForest fit on pre-2022 train data',
    'OOS_Precision': f'{prec_oos:.3f}',
    'OOS_Recall':    f'{rec_oos:.3f}',
    'OOS_F1':        f'{f1_oos:.3f}',
    'Episode_Recall': f'{len(detected)}/{len(oos_episodes)}',
    'Avg_Lead_Months': f'{avg_lead:.1f}',
    'Decision': f"Lead = {avg_lead:.1f} months, recall = {recall_ep*100:.0f}% (from crisis start, OOS only)",
})

# =============================================================================
# Save results
# =============================================================================
df_out = pd.DataFrame(records)
out_csv = os.path.join(BASE_DIR, "hypotheses_testing_results.csv")
df_out.to_csv(out_csv, index=False)
print(f"\n  Saved results -> {out_csv}")

# Save leakage-free crisis label for use by other scripts
label_csv = os.path.join(BASE_DIR, "realized_crisis_label.csv")
df[['Date', 'BDT_Return_30D_Fwd', 'Reserve_Change_30D_Fwd', 'Crisis_Label_Realized']].to_csv(label_csv, index=False)
print(f"  Saved realized crisis label -> {label_csv}")

print("\n" + "=" * 70)
print("  RESULTS SUMMARY (all computed dynamically, none hard-coded)")
print("=" * 70)
summary_cols = [c for c in ['RQ', 'Decision', 'XGB_RMSE', 'XGB_F1', 'MASE_vs_Naive', 'Episode_Recall'] if c in df_out.columns]
print(df_out[summary_cols].to_string(index=False))
print("=" * 70)
print("\n  Reviewer Fix verified:")
print("  [x] Daily_Economic_Stress_Index REMOVED from features")
print("  [x] Economic_Risk_Score REMOVED from features")
print("  [x] Crisis label = realized BDT/reserve outcome, NOT formula-derived")
print("  [x] No hard-coded 'CONFIRMED' — decision computed from p-value < 0.05")
print("  [x] Naive persistence baseline added to all regression comparisons")
print("  [x] DM test uses HLN correction (h-lag HAC + t-distribution)")
print("  [x] MASE (vs naive) reported alongside RMSE")
print("  [x] IsolationForest fitted on PRE-2022 train data only")
print("  [x] CUSUM with reset (reviewer bug fix)")
print("  [x] Lead time measured from crisis START, not peak")
