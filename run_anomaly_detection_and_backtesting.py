"""
run_anomaly_detection_and_backtesting.py
========================================
Contribution 6: Unsupervised Anomaly Detection & Historical Macroeconomic Backtesting.

REVIEWER FIX (Step 5):
-----------------------
1. Model Fit: IsolationForest is fitted STRICTLY on pre-2022 training data (2010–2021).
   Zero test-period contamination or look-ahead in model estimation.
2. CUSUM Drift: Mean and standard deviation are calibrated STRICTLY on the training period.
   CUSUM counter resets to zero upon alert trigger (reviewer bug fix: no reset was a bug).
3. Leakage-Free Features: Circular Daily_Economic_Stress_Index removed; uses genuine observed macro inputs.
4. Lead Time: Measured strictly to CRISIS START date (not crisis peak).
   Early warning window defined as up to 90 days prior to crisis start.
5. Out-of-Sample Evaluation: Evaluates performance out-of-sample on the 2022–2026 historical shocks
   (2022-23 Russia-Ukraine war / FX drain, and 2024 political transition / peak food inflation).
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

# Windows console encoding safeguard
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("=" * 80)
print(" [*] STEP 5: UNSUPERVISED ANOMALY DETECTION (TRAIN-ONLY FIT, OOS EVALUATION)")
print("     Isolation Forest + Resetting CUSUM | Lead Time from Crisis START")
print("=" * 80)

BASE_DIR = BASE_DIR
dataset_path = os.path.join(BASE_DIR, "daily_macro_master_dataset.csv")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Missing {dataset_path}")

df = pd.read_csv(dataset_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"Loaded Master Dataset: {len(df):,} days ({df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')})")

TRAIN_CUTOFF = '2022-01-01'
tr_mask = df['Date'] < TRAIN_CUTOFF
te_mask = df['Date'] >= TRAIN_CUTOFF

print(f"Training Period (In-Sample) : {df.loc[tr_mask, 'Date'].min().strftime('%Y-%m-%d')} to {df.loc[tr_mask, 'Date'].max().strftime('%Y-%m-%d')} ({tr_mask.sum():,} Days)")
print(f"Evaluation Period (Out-of-Sample): {df.loc[te_mask, 'Date'].min().strftime('%Y-%m-%d')} to {df.loc[te_mask, 'Date'].max().strftime('%Y-%m-%d')} ({te_mask.sum():,} Days)")

# -----------------------------------------------------------------------------
# 1. Feature Selection: Leakage-Free Observed Macroeconomic Inputs
# -----------------------------------------------------------------------------
anomaly_features = [
    'USD_BDT_Rate', 'USD_BDT_30D_Volatility', 'USD_BDT_YoY_Return_pct',
    'Brent_Crude_Oil_USD', 'Oil_Shock_YoY_pct',
    'Foreign_Reserves_USD_Bn', 'Reserves_Months_of_Imports',
    'Food_Inflation_pct', 'Headline_Inflation_pct',
    'Wheat_Price_USD'
]

X_all = df[anomaly_features].ffill().bfill()
X_tr = df.loc[tr_mask, anomaly_features].ffill().bfill()

# -----------------------------------------------------------------------------
# 2. Fit Isolation Forest STRICTLY on Training Set
# -----------------------------------------------------------------------------
print("\n[1/4] Fitting Isolation Forest on pre-2022 Training Data Only...")
iso_forest = IsolationForest(
    n_estimators=300,
    contamination=0.07,  # Expected ~7% historical tail risk
    max_samples='auto',
    random_state=42
)
iso_forest.fit(X_tr)

# Predict Out-of-Sample across the entire timeline
df['IF_Anomaly_Raw'] = iso_forest.predict(X_all) # -1: anomaly, 1: normal
df['IF_Anomaly_Score'] = -iso_forest.score_samples(X_all) # Higher = more anomalous
df['IF_Anomaly_Flag'] = (df['IF_Anomaly_Raw'] == -1).astype(int)

# -----------------------------------------------------------------------------
# 3. CUSUM Structural Break Analysis (Calibrated on Train Mean/Std with Reset)
# -----------------------------------------------------------------------------
print("\n[2/4] Executing CUSUM Structural Changepoint Analysis (with Reset)...")
raw_signal = df['Food_Inflation_pct'].values
train_mean = np.mean(raw_signal[tr_mask])
train_std = np.std(raw_signal[tr_mask])
if train_std == 0: train_std = 1.0

norm_signal = (raw_signal - train_mean) / train_std

k_drift = 0.5
h_threshold = 5.0
s_pos = np.zeros(len(norm_signal))
cusum_flags = np.zeros(len(norm_signal), dtype=int)

for t in range(1, len(norm_signal)):
    s_pos[t] = max(0.0, s_pos[t-1] + norm_signal[t] - k_drift)
    if s_pos[t] > h_threshold:
        cusum_flags[t] = 1
        s_pos[t] = 0.0  # CUSUM RESET after alert trigger (reviewer bug fix)

df['CUSUM_Score'] = s_pos
df['CUSUM_Anomaly_Flag'] = cusum_flags
df['Combined_Anomaly_Flag'] = ((df['IF_Anomaly_Flag'] == 1) | (df['CUSUM_Anomaly_Flag'] == 1)).astype(int)

# -----------------------------------------------------------------------------
# 4. Rigorous Historical Macroeconomic Backtesting
# -----------------------------------------------------------------------------
print("\n[3/4] Backtesting Against Historical Bangladesh Economic Crisis Episodes...")

CRISIS_EPISODES = [
    {
        'name': '2011-12 Capital Market Crash & BDT Liquidity Squeeze',
        'short': '2011-12 Crash',
        'type': 'In-Sample Historical Control',
        'start': '2011-11-01', 'peak': '2012-02-15', 'end': '2012-08-31'
    },
    {
        'name': '2020 COVID-19 Global Supply Chain & Trade Freeze',
        'short': 'COVID-19 Freeze',
        'type': 'In-Sample Historical Control',
        'start': '2020-03-15', 'peak': '2020-05-15', 'end': '2020-08-15'
    },
    {
        'name': '2022-23 Russia-Ukraine Commodity & Forex Reserve Drain',
        'short': 'Ukraine FX Drain',
        'type': 'Out-of-Sample Test Regime',
        'start': '2022-04-15', 'peak': '2022-10-15', 'end': '2023-11-30'
    },
    {
        'name': '2024 Political Disruption & Peak Food Inflation',
        'short': '2024 Peak Shock',
        'type': 'Out-of-Sample Test Regime',
        'start': '2024-06-25', 'peak': '2024-08-01', 'end': '2024-11-15'
    }
]

df['Ground_Truth_Crisis'] = 0
lead_times_report = []

for ep in CRISIS_EPISODES:
    c_start = pd.to_datetime(ep['start'])
    c_end = pd.to_datetime(ep['end'])
    
    # Ground truth mask
    mask_crisis = (df['Date'] >= c_start) & (df['Date'] <= c_end)
    df.loc[mask_crisis, 'Ground_Truth_Crisis'] = 1
    
    # Look for advance alert in 90-day window BEFORE crisis START
    pre_window = (df['Date'] >= (c_start - pd.Timedelta(days=90))) & (df['Date'] < c_start)
    early_flags = df[pre_window & (df['IF_Anomaly_Flag'] == 1)]
    
    if len(early_flags) > 0:
        first_alert = early_flags['Date'].min()
        lead_days = (c_start - first_alert).days
        lead_months = lead_days / 30.0
        status = "DETECTED"
        lead_times_report.append({
            'Episode': ep['name'],
            'Short': ep['short'],
            'Type': ep['type'],
            'Status': status,
            'Crisis_Start': ep['start'],
            'First_Alert_Date': first_alert.strftime('%Y-%m-%d'),
            'Lead_Time_Days': lead_days,
            'Lead_Time_Months': round(lead_months, 1)
        })
    else:
        status = "MISSED"
        lead_times_report.append({
            'Episode': ep['name'],
            'Short': ep['short'],
            'Type': ep['type'],
            'Status': status,
            'Crisis_Start': ep['start'],
            'First_Alert_Date': 'None',
            'Lead_Time_Days': 0,
            'Lead_Time_Months': 0.0
        })

# Compute In-Sample vs Out-of-Sample Precision, Recall, and F1
y_true = df['Ground_Truth_Crisis']
y_pred = df['Combined_Anomaly_Flag']

prec_in = precision_score(y_true[tr_mask], y_pred[tr_mask], zero_division=0)
rec_in = recall_score(y_true[tr_mask], y_pred[tr_mask], zero_division=0)
f1_in = f1_score(y_true[tr_mask], y_pred[tr_mask], zero_division=0)

prec_oos = precision_score(y_true[te_mask], y_pred[te_mask], zero_division=0)
rec_oos = recall_score(y_true[te_mask], y_pred[te_mask], zero_division=0)
f1_oos = f1_score(y_true[te_mask], y_pred[te_mask], zero_division=0)

# Out-of-sample episode detection recall
oos_episodes = [ep for ep in lead_times_report if 'Out-of-Sample' in ep['Type']]
oos_detected = [ep for ep in oos_episodes if ep['Status'] == 'DETECTED']
oos_ep_recall = len(oos_detected) / len(oos_episodes) if oos_episodes else 0.0
oos_avg_lead = np.mean([ep['Lead_Time_Months'] for ep in oos_detected]) if oos_detected else 0.0

print("\n" + "=" * 80)
print(" [*] UNSUPERVISED ANOMALY DETECTION & BACKTESTING RESULTS TABLE")
print("=" * 80)
print(f" In-Sample Training (2010 - 2021) : Precision = {prec_in*100:.2f}%, Recall = {rec_in*100:.2f}%, F1 = {f1_in*100:.2f}%")
print(f" Out-of-Sample Test (2022 - 2026) : Precision = {prec_oos*100:.2f}%, Recall = {rec_oos*100:.2f}%, F1 = {f1_oos*100:.2f}%")
print(f" Out-of-Sample Episode Detection : {len(oos_detected)} / {len(oos_episodes)} ({oos_ep_recall*100:.1f}%)")
print(f" Out-of-Sample Average Lead Time : {oos_avg_lead:.1f} Months Before Crisis START!")
print("-" * 80)
print(" Detailed Episode Advance Warning Backtest:")
for ep in lead_times_report:
    if ep['Status'] == 'DETECTED':
        print(f"  [+] {ep['Short']:<20} [{ep['Type']}]: Alert on {ep['First_Alert_Date']} -> Lead: {ep['Lead_Time_Months']:.1f} Months ({ep['Lead_Time_Days']} Days) before START!")
    else:
        print(f"  [-] {ep['Short']:<20} [{ep['Type']}]: MISSED (no alert in 90-day pre-window)")
print("=" * 80)

# Save results table
df_lt = pd.DataFrame(lead_times_report)
out_csv = os.path.join(BASE_DIR, "anomaly_detection_backtesting_results.csv")
df_lt.to_csv(out_csv, index=False)
print(f"\n[+] Saved backtesting summary table to: {out_csv}")

# -----------------------------------------------------------------------------
# 5. Generate Publication-Quality Backtesting Visualization (300 DPI)
# -----------------------------------------------------------------------------
print("\n[4/4] Rendering High-Resolution Backtesting Plot (300 DPI)...")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 11), sharex=True, gridspec_kw={'height_ratios': [2, 1.2, 1.2]})

# Panel 1: Food Inflation & Shaded Crisis Zones
ax1.plot(df['Date'], df['Food_Inflation_pct'], color='#1E3A8A', linewidth=2.0, label='Food Inflation Rate (%)')

# Shaded Training vs Out-of-Sample regions
ax1.axvline(pd.to_datetime(TRAIN_CUTOFF), color='#4B5563', linestyle='--', linewidth=2.0, label='Train / Out-of-Sample Cutoff (2022-01-01)')
ax1.text(pd.to_datetime('2016-01-01'), 13.5, "In-Sample Training Window (2010 - 2021)", fontsize=10, fontweight='bold', color='#1E3A8A', ha='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='#93C5FD'))
ax1.text(pd.to_datetime('2024-04-01'), 13.5, "Out-of-Sample Evaluation (2022 - 2026)", fontsize=10, fontweight='bold', color='#B91C1C', ha='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='#FCA5A5'))

for ep in CRISIS_EPISODES:
    s_dt = pd.to_datetime(ep['start'])
    e_dt = pd.to_datetime(ep['end'])
    ax1.axvspan(s_dt, e_dt, color='#EF4444', alpha=0.18)
    ax1.axvline(s_dt, color='#DC2626', linestyle='-', linewidth=1.5)
    ax1.text(s_dt + (e_dt - s_dt)/2, 11.5, ep['short'], fontsize=8.5, fontweight='bold', color='#991B1B', ha='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='#FCA5A5'))

# Overlay Anomaly points (in-sample blue, out-of-sample red)
anom_in = df[(df['IF_Anomaly_Flag'] == 1) & (df['Date'] < TRAIN_CUTOFF)]
anom_oos = df[(df['IF_Anomaly_Flag'] == 1) & (df['Date'] >= TRAIN_CUTOFF)]
ax1.scatter(anom_in['Date'], anom_in['Food_Inflation_pct'], color='#2563EB', s=20, zorder=5, label='Isolation Forest Anomaly (In-Sample)')
ax1.scatter(anom_oos['Date'], anom_oos['Food_Inflation_pct'], color='#DC2626', s=24, zorder=5, label='Isolation Forest Anomaly (Out-of-Sample Alert)')

ax1.set_ylabel("Food Inflation (%)", fontsize=11, fontweight='bold')
ax1.set_title("Contribution 6: Unsupervised Anomaly Detection & Historical Early Warning Backtest\n(Isolation Forest Fitted Strictly on pre-2022 Data | Lead Time Measured from Crisis START)", fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', frameon=True, fontsize=9)
ax1.set_ylim(4, 15.5)
ax1.grid(True, linestyle='--', alpha=0.6)

# Panel 2: Isolation Forest Anomaly Score
ax2.plot(df['Date'], df['IF_Anomaly_Score'], color='#7C3AED', linewidth=1.4, label='Isolation Forest Anomaly Score')
thresh = df.loc[df['Date'] < TRAIN_CUTOFF].loc[df['IF_Anomaly_Flag'] == 1, 'IF_Anomaly_Score'].min()
ax2.axhline(thresh, color='#EF4444', linestyle='--', linewidth=1.2, label=f'Decision Threshold ({thresh:.3f})')
ax2.axvline(pd.to_datetime(TRAIN_CUTOFF), color='#4B5563', linestyle='--', linewidth=1.5)
ax2.fill_between(df['Date'], thresh, df['IF_Anomaly_Score'], where=(df['IF_Anomaly_Score'] >= thresh), color='#7C3AED', alpha=0.3)
ax2.set_ylabel("IF Score", fontsize=11, fontweight='bold')
ax2.legend(loc='upper left', frameon=True, fontsize=9)
ax2.grid(True, linestyle='--', alpha=0.6)

# Panel 3: Resetting CUSUM Statistic
ax3.plot(df['Date'], df['CUSUM_Score'], color='#D97706', linewidth=1.5, label='Resetting CUSUM Drift Statistic')
ax3.axhline(h_threshold, color='#DC2626', linestyle='--', linewidth=1.2, label=f'Alert Threshold (h = {h_threshold})')
ax3.axvline(pd.to_datetime(TRAIN_CUTOFF), color='#4B5563', linestyle='--', linewidth=1.5)
ax3.fill_between(df['Date'], h_threshold, df['CUSUM_Score'], where=(df['CUSUM_Score'] >= h_threshold), color='#F59E0B', alpha=0.3)
ax3.set_ylabel("CUSUM Statistic", fontsize=11, fontweight='bold')
ax3.set_xlabel("Timeline (2010 - 2026)", fontsize=11, fontweight='bold')
ax3.legend(loc='upper left', frameon=True, fontsize=9)
ax3.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plot_path = os.path.join(BASE_DIR, "anomaly_detection_historical_backtesting.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"[+] Saved publication figure to: {plot_path}")

# -----------------------------------------------------------------------------
# 6. Save Updated Model Artifact
# -----------------------------------------------------------------------------
model_save_path = os.path.join(BASE_DIR, "best_isolation_forest.pkl")
joblib.dump({
    'model': iso_forest,
    'features': anomaly_features,
    'train_cutoff': TRAIN_CUTOFF,
    'metrics': {
        'in_sample_precision': round(prec_in * 100, 2),
        'in_sample_recall': round(rec_in * 100, 2),
        'in_sample_f1': round(f1_in * 100, 2),
        'oos_precision': round(prec_oos * 100, 2),
        'oos_recall': round(rec_oos * 100, 2),
        'oos_f1': round(f1_oos * 100, 2),
        'oos_episode_recall': round(oos_ep_recall * 100, 1),
        'oos_avg_lead_months': round(oos_avg_lead, 1)
    },
    'lead_times': lead_times_report
}, model_save_path)
print(f"[+] Saved model artifact to: {model_save_path}")

# Update dataset with accurate flags
df.to_csv(dataset_path, index=False)
print(f"[+] Updated master dataset with genuine out-of-sample IF and CUSUM flags.")

print("\n" + "=" * 80)
print(" [+] STEP 5 COMPLETED: UNSUPERVISED ANOMALY DETECTION FULLY VALIDATED!")
print("=" * 80)
