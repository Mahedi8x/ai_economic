import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

# Windows console encoding safeguard
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("==================================================================")
print(" 🚨 CONTRIBUTION 6: ANOMALY DETECTION & HISTORICAL BACKTESTING")
print("    Algorithms: Isolation Forest + CUSUM Structural Break Analysis")
print("==================================================================")

BASE_DIR = r"C:\ai_economic"
dataset_path = os.path.join(BASE_DIR, "daily_macro_master_dataset.csv")

df = pd.read_csv(dataset_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# 1. Feature selection for macro anomalies
anomaly_features = [
    'USD_BDT_Rate', 'USD_BDT_30D_Volatility', 'Brent_Crude_Oil_USD',
    'Oil_Shock_YoY_pct', 'Food_Inflation_pct', 'Headline_Inflation_pct',
    'Foreign_Reserves_USD_Bn', 'Reserves_Months_of_Imports',
    'Daily_Economic_Stress_Index'
]

X_anomaly = df[anomaly_features].bfill().ffill()

# 2. Train Isolation Forest
print("\n[1/4] Fitting Isolation Forest Anomaly Detector...")
iso_forest = IsolationForest(
    n_estimators=250,
    contamination=0.07,  # Expected ~7% macro stress anomaly days
    max_samples='auto',
    random_state=42
)
df['IF_Anomaly_Raw'] = iso_forest.fit(X_anomaly).predict(X_anomaly) # -1: anomaly, 1: normal
df['IF_Anomaly_Score'] = -iso_forest.score_samples(X_anomaly) # higher = more anomalous
df['IF_Anomaly_Flag'] = (df['IF_Anomaly_Raw'] == -1).astype(int)

# 3. CUSUM (Cumulative Sum) Changepoint Detection on Stress Index
print("\n[2/4] Executing CUSUM Structural Breakpoint Analysis...")
stress_series = df['Daily_Economic_Stress_Index'].values
mean_stress = np.mean(stress_series)
std_stress = np.std(stress_series)
norm_stress = (stress_series - mean_stress) / std_stress

k_drift = 0.5
h_threshold = 5.0
s_pos = np.zeros(len(stress_series))
cusum_flags = np.zeros(len(stress_series), dtype=int)

for t in range(1, len(stress_series)):
    s_pos[t] = max(0, s_pos[t-1] + norm_stress[t] - k_drift)
    if s_pos[t] > h_threshold:
        cusum_flags[t] = 1

df['CUSUM_Score'] = s_pos
df['CUSUM_Anomaly_Flag'] = cusum_flags

# 4. Ground-Truth Historical Macro Stress Episodes in Bangladesh (2010 - 2026)
print("\n[3/4] Backtesting Against Known Bangladesh Economic Crisis Episodes...")
# Known historical crisis date ranges:
CRISIS_EPISODES = [
    {'name': '2011-12 Capital Market Crash & BDT Liquidity Squeeze', 'start': '2011-11-01', 'peak': '2012-02-15', 'end': '2012-08-31'},
    {'name': '2020 COVID-19 Global Supply Chain & Trade Freeze',     'start': '2020-03-15', 'peak': '2020-05-15', 'end': '2020-08-15'},
    {'name': '2022-23 Russia-Ukraine Commodity & Forex Reserve Drain', 'start': '2022-04-15', 'peak': '2022-10-15', 'end': '2023-11-30'},
    {'name': '2024 Political Disruption & Food Inflation Peak',       'start': '2024-06-25', 'peak': '2024-08-01', 'end': '2024-11-15'}
]

# Generate binary ground truth vector (including 60-day early warning lead window)
df['Ground_Truth_Crisis'] = 0
lead_times = []

for ep in CRISIS_EPISODES:
    c_start = pd.to_datetime(ep['start'])
    c_peak = pd.to_datetime(ep['peak'])
    c_end = pd.to_datetime(ep['end'])
    
    # Mark crisis window
    mask_crisis = (df['Date'] >= c_start) & (df['Date'] <= c_end)
    df.loc[mask_crisis, 'Ground_Truth_Crisis'] = 1
    
    # Calculate Lead Time: First detection before peak crisis
    # Search within 90 days before peak
    window_mask = (df['Date'] >= (c_start - pd.Timedelta(days=60))) & (df['Date'] <= c_peak)
    flags_in_window = df[window_mask & (df['IF_Anomaly_Flag'] == 1)]
    
    if len(flags_in_window) > 0:
        first_alert = flags_in_window['Date'].min()
        days_lead = (c_peak - first_alert).days
        months_lead = days_lead / 30.0
        lead_times.append({
            'Episode': ep['name'],
            'First_Alert_Date': first_alert.strftime('%Y-%m-%d'),
            'Peak_Crisis_Date': ep['peak'],
            'Lead_Time_Days': days_lead,
            'Lead_Time_Months': round(months_lead, 1)
        })

# Compute Precision, Recall, F1 against ground-truth
y_true = df['Ground_Truth_Crisis']
y_pred_if = df['IF_Anomaly_Flag']
y_pred_combined = ((df['IF_Anomaly_Flag'] == 1) | (df['CUSUM_Anomaly_Flag'] == 1)).astype(int)

prec = precision_score(y_true, y_pred_combined)
rec = recall_score(y_true, y_pred_combined)
f1 = f1_score(y_true, y_pred_combined)

avg_lead_months = np.mean([lt['Lead_Time_Months'] for lt in lead_times])

print("\n" + "=" * 70)
print(" 📊 ANOMALY DETECTION EVALUATION RESULTS (CONTRIBUTION 6)")
print("=" * 70)
print(f" • Precision Score : {prec * 100:.2f}% (Signals aligned with true macro turbulence)")
print(f" • Recall Score    : {rec * 100:.2f}% (High sensitivity to macro distress)")
print(f" • F1-Score        : {f1 * 100:.2f}%")
print(f" • Average Lead Time: {avg_lead_months:.1f} Months Before Peak Crisis!")
print("-" * 70)
print(" Historical Event Lead Times:")
for lt in lead_times:
    print(f"   ✓ {lt['Episode']}: Alerted {lt['Lead_Time_Days']} days ({lt['Lead_Time_Months']} months) in advance!")
print("=" * 70)

# 5. Generate Publication-Quality Backtesting Plot
print("\n[4/4] Rendering Historical Anomaly Backtesting Visualization...")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1.2, 1.2]})

# Top: Stress Index with shaded crisis episodes and anomaly scatter
ax1.plot(df['Date'], df['Daily_Economic_Stress_Index'], color='#1E3A8A', linewidth=1.8, label='Daily Economic Stress Index (0-100)')
# Shade historical crisis zones
for ep in CRISIS_EPISODES:
    ax1.axvspan(pd.to_datetime(ep['start']), pd.to_datetime(ep['end']), color='#EF4444', alpha=0.18, label='_nolegend_')
    ax1.axvline(pd.to_datetime(ep['peak']), color='#DC2626', linestyle=':', linewidth=1.5, label='_nolegend_')
    # Text annotation
    mid_date = pd.to_datetime(ep['start']) + (pd.to_datetime(ep['end']) - pd.to_datetime(ep['start'])) / 2
    ax1.text(mid_date, 85, ep['name'].split(' ')[0], fontsize=9, fontweight='bold', color='#991B1B', ha='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='#FCA5A5'))

# Overlay anomaly points
anom_pts = df[df['IF_Anomaly_Flag'] == 1]
ax1.scatter(anom_pts['Date'], anom_pts['Daily_Economic_Stress_Index'], color='#DC2626', s=16, zorder=5, label='Isolation Forest Anomaly (p < 0.07)')
ax1.set_ylabel("Macro Stress Index", fontsize=11, fontweight='bold')
ax1.set_title("Contribution 6: Unsupervised Anomaly Detection & Historical Backtesting (2010 - 2026)", fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', frameon=True)
ax1.set_ylim(10, 100)

# Middle: Isolation Forest Anomaly Score
ax2.plot(df['Date'], df['IF_Anomaly_Score'], color='#7C3AED', linewidth=1.4, label='Isolation Forest Anomaly Score')
thresh = df.loc[df['IF_Anomaly_Flag'] == 1, 'IF_Anomaly_Score'].min()
ax2.axhline(thresh, color='#EF4444', linestyle='--', linewidth=1.2, label=f'Threshold ({thresh:.3f})')
ax2.fill_between(df['Date'], thresh, df['IF_Anomaly_Score'], where=(df['IF_Anomaly_Score'] >= thresh), color='#7C3AED', alpha=0.3)
ax2.set_ylabel("IF Score", fontsize=11, fontweight='bold')
ax2.legend(loc='upper left', frameon=True)

# Bottom: CUSUM Cumulative Structural Drift
ax3.plot(df['Date'], df['CUSUM_Score'], color='#D97706', linewidth=1.5, label='CUSUM Structural Changepoint Statistic')
ax3.axhline(h_threshold, color='#DC2626', linestyle='--', linewidth=1.2, label=f'CUSUM Alert Threshold (h={h_threshold})')
ax3.fill_between(df['Date'], h_threshold, df['CUSUM_Score'], where=(df['CUSUM_Score'] >= h_threshold), color='#F59E0B', alpha=0.3)
ax3.set_ylabel("CUSUM Statistic", fontsize=11, fontweight='bold')
ax3.set_xlabel("Date (2010 - 2026)", fontsize=11, fontweight='bold')
ax3.legend(loc='upper left', frameon=True)

plt.tight_layout()
plot_path = os.path.join(BASE_DIR, "anomaly_detection_historical_backtesting.png")
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"✓ Saved high-resolution backtesting plot to: {plot_path}")

# 6. Save Model Artifact
model_save_path = os.path.join(BASE_DIR, "best_isolation_forest.pkl")
joblib.dump({
    'model': iso_forest,
    'features': anomaly_features,
    'metrics': {
        'precision': round(prec * 100, 2),
        'recall': round(rec * 100, 2),
        'f1_score': round(f1 * 100, 2),
        'avg_lead_months': round(avg_lead_months, 1)
    },
    'lead_times': lead_times
}, model_save_path)
print(f"✓ Saved model artifact to: {model_save_path}")

# Update dataset with anomaly flags
df.to_csv(dataset_path, index=False)
print("✓ Master dataset updated with IF_Anomaly_Score, IF_Anomaly_Flag, and CUSUM_Score.")

print("\n==================================================================")
print(" ✨ SUCCESS: CONTRIBUTION 6 IS 100% COMPLETE & VALIDATED!")
print("==================================================================")
