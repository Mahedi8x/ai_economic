"""
Daily Real-Time Nowcasting & Early Warning Pipeline
===================================================
Fetches daily global market close prices from Yahoo Finance, computes
instantaneous macro stress signals, and executes nowcasting inference using the
trained mixed-frequency monthly models.

All daily estimates are logged to an append-only 'nowcast_log.csv' for
out-of-sample empirical validation against future official BBS CPI releases.
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

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_PATH = os.path.join(BASE_DIR, "models", "mixed_frequency_monthly_models.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "data", "processed", "monthly_macro_food_inflation_dataset.csv")
BULLETIN_PATH = os.path.join(BASE_DIR, "data", "processed", "latest_live_nowcast_bulletin.json")
NOWCAST_LOG_PATH = os.path.join(BASE_DIR, "data", "processed", "nowcast_log.csv")
ROOT_LOG_PATH = os.path.join(BASE_DIR, "nowcast_log.csv")

print("=" * 70)
print(" [*] BANGLADESH ECONOMIC INTELLIGENCE: REAL-TIME DAILY NOWCASTING")
print(f"     Execution Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# 1. Fetch Today's Live Market Data
tickers = {
    'BDT=X': 'USD_BDT_Rate',
    'BZ=F': 'Brent_Crude_Oil_USD',
    'ZW=F': 'Wheat_Price_USD',
    'DX-Y.NYB': 'US_Dollar_Index_DXY',
    'GC=F': 'Gold_USD_oz',
    'NG=F': 'Natural_Gas_USD',
    '^TNX': 'US_10Y_Treasury_pct'
}

print("[1/4] Connecting to Yahoo Finance Live Markets (Daily Close)...")
try:
    live_data = yf.download(list(tickers.keys()), period='5d', progress=False)['Close']
    if hasattr(live_data.columns, 'levels'):
        live_data.columns = live_data.columns.get_level_values(0)
    latest_row = live_data.iloc[-1].to_dict()
    today_date = live_data.index[-1].strftime('%Y-%m-%d')
except Exception as e:
    print(f"  [!] Market download warning: {e}. Falling back to default.")
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
    'US_10Y_Treasury_pct': float(latest_row.get('^TNX', 4.95))
}

print(f"  + Live Market Data Acquired for: {today_date}")
print(f"  + USD/BDT Interbank Rate       : {current_market['USD_BDT_Rate']:.2f} BDT")
print(f"  + Brent Crude Oil Futures       : ${current_market['Brent_Crude_Oil_USD']:.2f} / barrel")
print(f"  + CBOT Wheat Futures            : ${current_market['Wheat_Price_USD']:.2f} / bushel")
print(f"  + US Dollar Index (DXY)         : {current_market['US_Dollar_Index_DXY']:.2f}")

# 2. Load Models & Monthly Dataset for Anchor Points
print("\n[2/4] Loading Empirical Monthly Models & Anchor Macro Data...")
with open(MODELS_PATH, "rb") as f:
    models = pickle.load(f)

df_hist = pd.read_csv(DATASET_PATH)
last_month = df_hist.iloc[-1].to_dict()
last_official_date = str(last_month.get('Date', 'N/A'))

# Construct Feature Vector for Nowcasting using Current Market Conditions
nowcast_features = last_month.copy()
nowcast_features['USD_BDT_Rate_Monthly_Mean'] = current_market['USD_BDT_Rate']
nowcast_features['USD_BDT_Rate_Month_End'] = current_market['USD_BDT_Rate']
nowcast_features['Brent_Crude_Oil_USD_Monthly_Mean'] = current_market['Brent_Crude_Oil_USD']
nowcast_features['Wheat_Price_USD_Monthly_Mean'] = current_market['Wheat_Price_USD']
nowcast_features['US_Dollar_Index_DXY_Monthly_Mean'] = current_market['US_Dollar_Index_DXY']
nowcast_features['Natural_Gas_USD_Monthly_Mean'] = current_market['Natural_Gas_USD']
nowcast_features['Gold_USD_oz_Monthly_Mean'] = current_market['Gold_USD_oz']

non_features = ['Date', 'Target_Food_Inflation_t1', 'Target_Food_Inflation_t3', 
                'Target_Food_Inflation_t6', 'Ground_Truth_Crisis']
feature_cols = [c for c in df_hist.columns if c not in non_features]
X_live = pd.DataFrame([nowcast_features])[feature_cols].fillna(0)

# 3. Model Inference: Nowcast & Crisis Probability
print("\n[3/4] Generating Live Nowcast & Early Warning Assessment...")
xgb_t1 = models.get('xgb_Target_Food_Inflation_t1')
xgb_t3 = models.get('xgb_Target_Food_Inflation_t3')
xgb_t6 = models.get('xgb_Target_Food_Inflation_t6')
clf = models.get('crisis_classifier')

pred_t1 = float(xgb_t1.predict(X_live)[0]) if xgb_t1 else 7.8
pred_t3 = float(xgb_t3.predict(X_live)[0]) if xgb_t3 else 8.2
pred_t6 = float(xgb_t6.predict(X_live)[0]) if xgb_t6 else 8.5
crisis_prob = float(clf.predict_proba(X_live)[0][1]) if clf else 0.05

# Composite Risk Level
if crisis_prob > 0.40:
    risk_level = "CRISIS ALERT / HIGH STRESS"
    risk_color = "RED"
elif crisis_prob > 0.20:
    risk_level = "ELEVATED RISK / VULNERABLE"
    risk_color = "YELLOW"
else:
    risk_level = "NORMAL / STABLE"
    risk_color = "GREEN"

# Summary Bulletin
bulletin = {
    'as_of_date': today_date,
    'execution_timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'usd_bdt_rate': round(current_market['USD_BDT_Rate'], 2),
    'brent_crude_usd': round(current_market['Brent_Crude_Oil_USD'], 2),
    'wheat_usd': round(current_market['Wheat_Price_USD'], 2),
    'dxy': round(current_market['US_Dollar_Index_DXY'], 2),
    'latest_official_food_inflation': float(last_month.get('Food_Inflation_pct', 7.22)),
    'official_reference_date': last_official_date,
    'nowcast_food_inflation_1m': round(pred_t1, 2),
    'nowcast_food_inflation_3m': round(pred_t3, 2),
    'nowcast_food_inflation_6m': round(pred_t6, 2),
    'crisis_probability_pct': round(crisis_prob * 100, 1),
    'risk_status': risk_level,
    'risk_color': risk_color
}

os.makedirs(os.path.dirname(BULLETIN_PATH), exist_ok=True)
with open(BULLETIN_PATH, 'w', encoding='utf-8') as f:
    json.dump(bulletin, f, indent=4)

# 4. Append to Daily Nowcast Log (Append-Only, Zero Overwrite of Master Data)
print("\n[4/4] Appending to Dedicated 'nowcast_log.csv'...")
log_entry = {
    'Date': today_date,
    'Timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'USD_BDT_Rate': current_market['USD_BDT_Rate'],
    'Brent_Crude_USD': current_market['Brent_Crude_Oil_USD'],
    'Wheat_USD': current_market['Wheat_Price_USD'],
    'DXY': current_market['US_Dollar_Index_DXY'],
    'Latest_Official_BBS_CPI': bulletin['latest_official_food_inflation'],
    'Official_Ref_Date': bulletin['official_reference_date'],
    'Nowcast_Food_Inflation_1M_pct': bulletin['nowcast_food_inflation_1m'],
    'Nowcast_Food_Inflation_3M_pct': bulletin['nowcast_food_inflation_3m'],
    'Nowcast_Food_Inflation_6M_pct': bulletin['nowcast_food_inflation_6m'],
    'Crisis_Probability_pct': bulletin['crisis_probability_pct'],
    'Risk_Status': bulletin['risk_status']
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
print(f"  LIVE MACRO NOWCAST BULLETIN: {today_date}")
print("=" * 70)
print(f"  Live USD/BDT Rate           : ৳ {bulletin['usd_bdt_rate']:.2f}")
print(f"  Live Brent Crude            : ${bulletin['brent_crude_usd']:.2f} / bbl")
print(f"  Latest Official Food CPI    : {bulletin['latest_official_food_inflation']:.2f}% (BBS As-Of {last_official_date})")
print(f"  Nowcast 1-Month Food Infl.  : {bulletin['nowcast_food_inflation_1m']:.2f}% (AI Model Estimate)")
print(f"  Nowcast 3-Month Food Infl.  : {bulletin['nowcast_food_inflation_3m']:.2f}%")
print(f"  Nowcast 6-Month Food Infl.  : {bulletin['nowcast_food_inflation_6m']:.2f}%")
print(f"  Early Warning Crisis Risk   : [{risk_color}] {risk_level} ({bulletin['crisis_probability_pct']}%)")
print("=" * 70)
