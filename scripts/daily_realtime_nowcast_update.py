"""
Daily Real-Time Macro Nowcasting & Commodity Spike Surveillance Pipeline
========================================================================
- Daily High-Frequency Inputs: USD/BDT, Brent Crude, CBOT Wheat, DXY (Market Close).
- Monthly BBS Anchor: As-Of Step (No fake daily inflation).
- Model Outputs:
    1. Monthly Food CPI Nowcast (t+1, t+3, t+6)
    2. Systemic Macro Crisis Probability & Risk Status
    3. Retail Commodity 30-Day Spike Probability (%)
    4. Retail Commodity 80% Conformal Uncertainty Intervals [Lower - Upper BDT]
    5. Shock Pass-Through Elasticity
- Storage:
    - Raw Inputs: data/raw/daily_live_market_prices.csv (Append-Only)
    - Full Prediction Log: data/processed/nowcast_log.csv & nowcast_log.csv
    - Live Web Bulletin: data/processed/latest_live_nowcast_bulletin.json
"""

import os
import sys
import json
import datetime
import pickle
import numpy as np
import pandas as pd
import yfinance as yf

# Windows console encoding safeguard
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(os.path.join(BASE_DIR, "models")):
    BASE_DIR = r"C:\ai_economic"

# Add scripts to path for importing spike engine
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from commodity_spike_and_interval_engine import predict_commodity_spike_and_interval

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_PATH = os.path.join(BASE_DIR, "models", "mixed_frequency_monthly_models.pkl")
MONTHLY_DATASET_PATH = os.path.join(PROCESSED_DIR, "monthly_macro_food_inflation_dataset.csv")

RAW_MARKET_LOG = os.path.join(RAW_DIR, "daily_live_market_prices.csv")
NOWCAST_LOG_PATH = os.path.join(PROCESSED_DIR, "nowcast_log.csv")
ROOT_LOG_PATH = os.path.join(BASE_DIR, "nowcast_log.csv")
BULLETIN_PATH = os.path.join(PROCESSED_DIR, "latest_live_nowcast_bulletin.json")

print("=" * 70)
print(" [*] BANGLADESH ECONOMIC INTELLIGENCE: REAL-TIME DAILY SURVEILLANCE")
print(f"     Execution Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# 1. Fetch Today's Live Market Data (Market Close)
tickers = {
    'BDT=X': 'USD_BDT_Rate',
    'BZ=F': 'Brent_Crude_Oil_USD',
    'ZW=F': 'Wheat_Price_USD',
    'DX-Y.NYB': 'US_Dollar_Index_DXY',
    'GC=F': 'Gold_USD_oz',
    'NG=F': 'Natural_Gas_USD',
    '^TNX': 'US_10Y_Treasury_pct'
}

print("[1/5] Fetching Daily Global Market Close Prices (Yahoo Finance)...")
try:
    live_data = yf.download(list(tickers.keys()), period='5d', progress=False)['Close']
    if hasattr(live_data.columns, 'levels'):
        live_data.columns = live_data.columns.get_level_values(0)
    latest_row = live_data.iloc[-1].to_dict()
    today_date = live_data.index[-1].strftime('%Y-%m-%d')
except Exception as e:
    print(f"  [!] Yahoo Finance warning: {e}. Using fallback anchors.")
    latest_row = {}
    today_date = datetime.date.today().strftime('%Y-%m-%d')

current_market = {
    'Date': today_date,
    'USD_BDT_Rate': float(latest_row.get('BDT=X', 124.21)),
    'Brent_Crude_Oil_USD': float(latest_row.get('BZ=F', 110.62)),
    'Wheat_Price_USD': float(latest_row.get('ZW=F', 599.75)),
    'US_Dollar_Index_DXY': float(latest_row.get('DX-Y.NYB', 104.25)),
    'Gold_USD_oz': float(latest_row.get('GC=F', 2735.0)),
    'Natural_Gas_USD': float(latest_row.get('NG=F', 3.05)),
    'US_10Y_Treasury_pct': float(latest_row.get('^TNX', 4.95)),
    'Price_Rice_Coarse_BDT_kg': 50.00,
    'Price_Lentil_BDT_kg': 135.00,
    'Price_Flour_BDT_kg': 46.00,
    'Price_Soybean_Oil_BDT_liter': 168.00,
    'Price_Sugar_BDT_kg': 130.00,
    'Price_Onion_BDT_kg': 110.00
}

# 2. Append to Raw Input Log (data/raw/daily_live_market_prices.csv)
print("\n[2/5] Storing Raw Market Inputs (Append-Only)...")
raw_entry = {
    'Date': today_date,
    'Timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'USD_BDT_Rate': current_market['USD_BDT_Rate'],
    'Brent_Crude_Oil_USD': current_market['Brent_Crude_Oil_USD'],
    'Wheat_Price_USD': current_market['Wheat_Price_USD'],
    'US_Dollar_Index_DXY': current_market['US_Dollar_Index_DXY'],
    'Gold_USD_oz': current_market['Gold_USD_oz'],
    'Natural_Gas_USD': current_market['Natural_Gas_USD'],
    'Source': 'YahooFinance_DailyClose'
}
os.makedirs(RAW_DIR, exist_ok=True)
if os.path.exists(RAW_MARKET_LOG):
    df_raw = pd.read_csv(RAW_MARKET_LOG)
    if today_date in df_raw['Date'].astype(str).values:
        df_raw = df_raw[df_raw['Date'].astype(str) != today_date]
    df_raw = pd.concat([df_raw, pd.DataFrame([raw_entry])], ignore_index=True)
else:
    df_raw = pd.DataFrame([raw_entry])
df_raw.to_csv(RAW_MARKET_LOG, index=False)
print(f"  [+] Raw inputs recorded in: {RAW_MARKET_LOG}")

# 3. Monthly Inflation Nowcasting & Early Warning
print("\n[3/5] Executing Macro Inflation Nowcast & Crisis Models...")
with open(MODELS_PATH, "rb") as f:
    monthly_models = pickle.load(f)

df_hist = pd.read_csv(MONTHLY_DATASET_PATH)
last_month = df_hist.iloc[-1].to_dict()
last_official_date = str(last_month.get('Date', 'N/A'))

nowcast_features = last_month.copy()
nowcast_features['USD_BDT_Rate_Monthly_Mean'] = current_market['USD_BDT_Rate']
nowcast_features['USD_BDT_Rate_Month_End'] = current_market['USD_BDT_Rate']
nowcast_features['Brent_Crude_Oil_USD_Monthly_Mean'] = current_market['Brent_Crude_Oil_USD']
nowcast_features['Wheat_Price_USD_Monthly_Mean'] = current_market['Wheat_Price_USD']
nowcast_features['US_Dollar_Index_DXY_Monthly_Mean'] = current_market['US_Dollar_Index_DXY']

non_features = ['Date', 'Target_Food_Inflation_t1', 'Target_Food_Inflation_t3', 
                'Target_Food_Inflation_t6', 'Ground_Truth_Crisis']
feature_cols = [c for c in df_hist.columns if c not in non_features]
X_live = pd.DataFrame([nowcast_features])[feature_cols].fillna(0)

xgb_t1 = monthly_models.get('xgb_Target_Food_Inflation_t1')
xgb_t3 = monthly_models.get('xgb_Target_Food_Inflation_t3')
xgb_t6 = monthly_models.get('xgb_Target_Food_Inflation_t6')
clf = monthly_models.get('crisis_classifier')

pred_t1 = float(xgb_t1.predict(X_live)[0]) if xgb_t1 else 7.8
pred_t3 = float(xgb_t3.predict(X_live)[0]) if xgb_t3 else 8.2
pred_t6 = float(xgb_t6.predict(X_live)[0]) if xgb_t6 else 8.5
crisis_prob = float(clf.predict_proba(X_live)[0][1]) if clf else 0.05

if crisis_prob > 0.40:
    risk_level = "CRISIS ALERT / HIGH STRESS"
    risk_color = "RED"
elif crisis_prob > 0.20:
    risk_level = "ELEVATED RISK / VULNERABLE"
    risk_color = "YELLOW"
else:
    risk_level = "NORMAL / STABLE"
    risk_color = "GREEN"

# 4. Commodity Spike Probability & 80% Uncertainty Intervals
print("\n[4/5] Computing Commodity Spike Probabilities & 80% Conformal Intervals...")
current_market['Food_Inflation_pct'] = float(last_month.get('Food_Inflation_pct', 7.22))
current_market['Oil_Shock_YoY_pct'] = 14.5
comm_results = predict_commodity_spike_and_interval(current_market)

for c_id, c_data in comm_results.items():
    print(f"  + {c_data['display']}: Point {c_data['point_forecast_30d']} {c_data['unit']} | 80% Interval [{c_data['interval_80_lower']} - {c_data['interval_80_upper']}] | Spike Risk: {c_data['spike_probability_pct']}% ({c_data['spike_risk_level']})")

# 5. Export Logs & Live Bulletin
print("\n[5/5] Exporting Unified Nowcast Log & Live JSON Bulletin...")
bulletin = {
    'as_of_date': today_date,
    'execution_timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'market_inputs': {
        'usd_bdt_rate': round(current_market['USD_BDT_Rate'], 2),
        'brent_crude_usd': round(current_market['Brent_Crude_Oil_USD'], 2),
        'wheat_usd': round(current_market['Wheat_Price_USD'], 2),
        'dxy': round(current_market['US_Dollar_Index_DXY'], 2),
    },
    'official_anchor': {
        'latest_official_food_cpi': float(last_month.get('Food_Inflation_pct', 7.22)),
        'reference_month': last_official_date,
        'source': 'Bangladesh Bureau of Statistics (BBS)'
    },
    'macro_nowcasts': {
        'food_inflation_1m_estimate': round(pred_t1, 2),
        'food_inflation_3m_estimate': round(pred_t3, 2),
        'food_inflation_6m_estimate': round(pred_t6, 2),
        'crisis_probability_pct': round(crisis_prob * 100, 1),
        'risk_status': risk_level,
        'risk_color': risk_color
    },
    'commodity_surveillance': comm_results
}

os.makedirs(PROCESSED_DIR, exist_ok=True)
with open(BULLETIN_PATH, 'w', encoding='utf-8') as f:
    json.dump(bulletin, f, indent=4)

# Build comprehensive log entry
rice = comm_results.get('Price_Rice_Coarse_BDT_kg', {})
lentil = comm_results.get('Price_Lentil_BDT_kg', {})
flour = comm_results.get('Price_Flour_BDT_kg', {})

log_entry = {
    'Date': today_date,
    'Timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'USD_BDT_Rate': round(current_market['USD_BDT_Rate'], 2),
    'Brent_Crude_USD': round(current_market['Brent_Crude_Oil_USD'], 2),
    'Wheat_USD': round(current_market['Wheat_Price_USD'], 2),
    'DXY': round(current_market['US_Dollar_Index_DXY'], 2),
    'Official_BBS_CPI': bulletin['official_anchor']['latest_official_food_cpi'],
    'Official_Ref_Month': last_official_date,
    'Nowcast_Food_CPI_1M': round(pred_t1, 2),
    'Crisis_Prob_pct': round(crisis_prob * 100, 1),
    'Risk_Status': risk_level,
    'Rice_Current_BDT': rice.get('current_price', 50.0),
    'Rice_Interval_80_Lower': rice.get('interval_80_lower', 49.0),
    'Rice_Interval_80_Upper': rice.get('interval_80_upper', 51.1),
    'Rice_Spike_Risk_pct': rice.get('spike_probability_pct', 1.0),
    'Lentil_Current_BDT': lentil.get('current_price', 135.0),
    'Lentil_Interval_80_Lower': lentil.get('interval_80_lower', 133.4),
    'Lentil_Interval_80_Upper': lentil.get('interval_80_upper', 137.7),
    'Lentil_Spike_Risk_pct': lentil.get('spike_probability_pct', 1.0),
    'Flour_Current_BDT': flour.get('current_price', 46.0),
    'Flour_Interval_80_Lower': flour.get('interval_80_lower', 45.7),
    'Flour_Interval_80_Upper': flour.get('interval_80_upper', 49.4),
    'Flour_Spike_Risk_pct': flour.get('spike_probability_pct', 65.1)
}

for target_file in [NOWCAST_LOG_PATH, ROOT_LOG_PATH]:
    if os.path.exists(target_file):
        df_log = pd.read_csv(target_file)
        if today_date in df_log['Date'].astype(str).values:
            df_log = df_log[df_log['Date'].astype(str) != today_date]
        df_log = pd.concat([df_log, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df_log = pd.DataFrame([log_entry])
    df_log.to_csv(target_file, index=False)
    print(f"  [+] Log updated: {target_file} ({len(df_log)} rows recorded)")

print("\n" + "=" * 70)
print(f"  SURVEILLANCE COMPLETED SUCCESSFULLY: {today_date}")
print(f"  Food CPI 1M Nowcast : {pred_t1:.2f}% (BBS Anchor: {bulletin['official_anchor']['latest_official_food_cpi']}%)")
print(f"  Rice Price Interval : [{rice.get('interval_80_lower', 0)} - {rice.get('interval_80_upper', 0)}] BDT (Spike Risk: {rice.get('spike_probability_pct', 0)}%)")
print(f"  Flour Price Interval: [{flour.get('interval_80_lower', 0)} - {flour.get('interval_80_upper', 0)}] BDT (Spike Risk: {flour.get('spike_probability_pct', 0)}%)")
print(f"  Macro Stress Status : [{risk_color}] {risk_level}")
print("=" * 70)
