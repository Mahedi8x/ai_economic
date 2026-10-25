import os
import sys
import datetime
import joblib
import pandas as pd
import numpy as np
import yfinance as yf

# Windows console encoding safeguard
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "daily_macro_master_dataset.csv")
BUSINESS_DATASET_PATH = os.path.join(BASE_DIR, "daily_macro_business_days.csv")
RAW_YF_PATH = os.path.join(BASE_DIR, "raw_data", "yahoo_finance_market_daily_raw.csv")
LOG_PATH = os.path.join(BASE_DIR, "daily_update_log.txt")

REG_MODEL_PATH = os.path.join(BASE_DIR, "best_forecasting_model.pkl")
CLF_MODEL_PATH = os.path.join(BASE_DIR, "best_crisis_classifier.pkl")
FOOD_MODEL_PATH = os.path.join(BASE_DIR, "best_food_inflation_model.pkl")
COMM_MODEL_PATH = os.path.join(BASE_DIR, "best_commodity_models.pkl")
RAW_COMM_PATH = os.path.join(BASE_DIR, "raw_data", "tcb_dam_retail_commodity_prices_2010_2026_raw.csv")

def log(msg):
    print(msg)
    try:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")
    except Exception:
        pass

log(f"\n{'='*70}")
log(f" 🚀 BANGLADESH ECONOMIC INTELLIGENCE: DAILY LIVE UPDATE")
log(f"    Execution Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"{'='*70}")

# 1. Load Current Dataset
if not os.path.exists(DATASET_PATH):
    log(f"❌ Error: Master dataset not found at {DATASET_PATH}")
    sys.exit(1)

df = pd.read_csv(DATASET_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
last_date = df['Date'].max()

log(f"\n[1/4] Current Database Status:")
log(f"  • Latest Recorded Date : {last_date.strftime('%Y-%m-%d')}")
log(f"  • Total Days Stored    : {len(df):,} Days")

# 2. Fetch Latest Tickers from Yahoo Finance
log("\n[2/4] Connecting to Yahoo Finance Live Markets...")
tickers = {
    'BDT=X': 'USD_BDT_Rate',
    'BZ=F': 'Brent_Crude_Oil_USD',
    'DX-Y.NYB': 'US_Dollar_Index_DXY',
    'GC=F': 'Gold_USD_oz',
    '^TNX': 'US_10Y_Treasury_pct',
    'NG=F': 'Natural_Gas_USD',
    'ZW=F': 'Wheat_Price_USD'
}

try:
    live_df = yf.download(list(tickers.keys()), period="5d", progress=False)['Close']
    if isinstance(live_df.columns, pd.MultiIndex):
        live_df.columns = live_df.columns.get_level_values(0)
    
    rename_map = {k: tickers[k] for k in tickers.keys() if k in live_df.columns}
    live_df = live_df.rename(columns=rename_map).dropna(how='all')
    live_df = live_df.reset_index()
    live_df['Date'] = pd.to_datetime(live_df['Date'])
    
    latest_live_row = live_df.iloc[-1]
    live_date = latest_live_row['Date']
    log(f"  ✓ Live Market Data Received for: {live_date.strftime('%Y-%m-%d')}")
except Exception as e:
    log(f"  ⚠️ Warning: Live API fetch issue ({e}). Using latest available row.")
    latest_live_row = df.iloc[-1]
    live_date = last_date

# 3. Update Dataset if new day available
log("\n[3/4] Updating Database & Engineering Features...")
new_row = df.iloc[-1].copy()
new_row['Date'] = live_date
new_row['Day_of_Week'] = live_date.day_name()

for col in tickers.values():
    if col in latest_live_row and pd.notnull(latest_live_row[col]):
        new_row[col] = round(float(latest_live_row[col]), 2)

# If today is newer than database, append it
if live_date > last_date:
    df_temp = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # High-Frequency Commodity Shock for Daily Food Inflation
    wheat_shock = (df_temp['Wheat_Price_USD'].pct_change(7).iloc[-1])
    oil_shock = (df_temp['Brent_Crude_Oil_USD'].pct_change(7).iloc[-1])
    usd_shock = (df_temp['USD_BDT_Rate'].pct_change(7).iloc[-1])
    d_shock = (0.05 * wheat_shock + 0.03 * oil_shock + 0.04 * usd_shock) * 10.0
    
    df_temp.loc[df_temp.index[-1], 'Food_Inflation_pct'] = round(df_temp.loc[df_temp.index[-2], 'Food_Inflation_pct'] + d_shock, 2)
    
    # Rolling features
    df_temp['USD_BDT_30D_Volatility'] = df_temp['USD_BDT_Rate'].rolling(30, min_periods=1).std().round(3)
    df_temp['USD_BDT_YoY_Return_pct'] = (df_temp['USD_BDT_Rate'].pct_change(365).fillna(0) * 100).round(2)
    df_temp['Oil_Shock_YoY_pct'] = (df_temp['Brent_Crude_Oil_USD'].pct_change(365).fillna(0) * 100).round(2)
    df_temp['DXY_30D_MA'] = df_temp['US_Dollar_Index_DXY'].rolling(30, min_periods=1).mean().round(2)
    
    # Stress Index
    stress_score = (
        (df_temp['Headline_Inflation_pct'] * 2.5) +
        (np.maximum(0, 5.0 - df_temp['Reserves_Months_of_Imports']) * 7.0) +
        (np.maximum(0, df_temp['USD_BDT_YoY_Return_pct']) * 1.2) +
        (np.maximum(0, df_temp['Brent_Crude_Oil_USD'] - 80.0) * 0.3)
    )
    df_temp['Daily_Economic_Stress_Index'] = np.clip(stress_score, 10.0, 100.0).round(1)
    
    # Save back to CSVs
    df_temp.to_csv(DATASET_PATH, index=False)
    df_temp[df_temp['Date'].dt.dayofweek < 5].to_csv(BUSINESS_DATASET_PATH, index=False)
    
    # Also append raw market data to raw_data/yahoo_finance_market_daily_raw.csv if path exists
    if os.path.exists(RAW_YF_PATH):
        try:
            df_raw_yf = pd.read_csv(RAW_YF_PATH)
            df_raw_yf['Date'] = pd.to_datetime(df_raw_yf['Date'])
            if live_date > df_raw_yf['Date'].max():
                raw_new = {'Date': live_date.strftime('%Y-%m-%d')}
                for k, v in tickers.items():
                    if v in latest_live_row and pd.notnull(latest_live_row[v]):
                        raw_new[k] = float(latest_live_row[v])
                df_raw_yf = pd.concat([df_raw_yf, pd.DataFrame([raw_new])], ignore_index=True)
                df_raw_yf.to_csv(RAW_YF_PATH, index=False)
                log(f"  ✓ Synchronized raw market file: {RAW_YF_PATH}")
        except Exception as e_raw:
            log(f"  ⚠️ Could not append to raw market file: {e_raw}")
    
    eval_row = df_temp.iloc[-1]
    log(f"  ✓ Successfully appended new record for {live_date.strftime('%Y-%m-%d')} to {DATASET_PATH}!")
else:
    eval_row = df.iloc[-1]
    log(f"  ✓ Database is already up to date as of {last_date.strftime('%Y-%m-%d')}.")

# 4. Run AI Models for Today's Forecast
log("\n[4/4] Executing AI Predictive Intelligence Engine...")
feature_cols = [
    'USD_BDT_Rate', 'Brent_Crude_Oil_USD', 'US_Dollar_Index_DXY', 'Gold_USD_oz',
    'Natural_Gas_USD', 'Wheat_Price_USD', 'US_10Y_Treasury_pct',
    'Headline_Inflation_pct', 'Food_Inflation_pct', 'Foreign_Reserves_USD_Bn',
    'Remittance_USD_Bn', 'Reserves_Months_of_Imports', 'External_Debt_USD_Bn',
    'Short_Term_Debt_USD_Bn', 'USD_BDT_30D_Volatility', 'USD_BDT_YoY_Return_pct',
    'Oil_Shock_YoY_pct', 'DXY_30D_MA', 'Reserves_to_Short_Term_Debt_Ratio',
    'Daily_Economic_Stress_Index'
]

reg_model = joblib.load(REG_MODEL_PATH)
clf_model = joblib.load(CLF_MODEL_PATH)
food_model = joblib.load(FOOD_MODEL_PATH) if os.path.exists(FOOD_MODEL_PATH) else None
comm_models = joblib.load(COMM_MODEL_PATH) if os.path.exists(COMM_MODEL_PATH) else None

input_data = pd.DataFrame([eval_row[feature_cols]])
pred_return = reg_model.predict(input_data)[0]
current_rate = eval_row['USD_BDT_Rate']
pred_rate_30d = current_rate * (1.0 + pred_return / 100.0)

crisis_prob = clf_model.predict_proba(input_data)[0, 1]
is_alert = crisis_prob >= 0.50

# Food Inflation Forecast
food_feature_cols = [
    'Wheat_Price_USD', 'Brent_Crude_Oil_USD', 'Natural_Gas_USD', 'Gold_USD_oz',
    'Oil_Shock_YoY_pct', 'USD_BDT_Rate', 'US_Dollar_Index_DXY',
    'Foreign_Reserves_USD_Bn', 'Reserves_Months_of_Imports',
    'USD_BDT_30D_Volatility', 'Food_Inflation_pct', 'Headline_Inflation_pct',
    'Daily_Economic_Stress_Index'
]
current_food_inf = eval_row['Food_Inflation_pct']
if food_model:
    food_input = pd.DataFrame([eval_row[food_feature_cols]])
    pred_food_delta = food_model.predict(food_input)[0]
    pred_food_inf_30d = current_food_inf + pred_food_delta
else:
    pred_food_inf_30d = current_food_inf

# 5. Output Executive Intelligence Report
status_symbol = "🔴 RED ALERT" if is_alert else "🟢 STABLE / NORMAL"
status_color_text = "HIGH CRISIS RISK" if is_alert else "LOW / ACCEPTABLE RISK"
food_status = "🔴 SEVERE STRESS (>10%)" if pred_food_inf_30d >= 10.0 else ("🟡 ELEVATED (7-10%)" if pred_food_inf_30d >= 7.0 else "🟢 STABLE (<7%)")

log("\n" + "=" * 66)
log(f" 📊 LIVE ECONOMIC INTELLIGENCE BULLETIN: {eval_row['Date'].strftime('%d %B, %Y')}")
log("=" * 66)
log(f" 💵 Current USD/BDT Exchange Rate : ৳ {current_rate:.2f}")
log(f" 🛢️ Brent Crude Oil Price         : ${eval_row['Brent_Crude_Oil_USD']:.2f} / barrel ({eval_row['Oil_Shock_YoY_pct']:+.1f}% YoY)")
log(f" 🌾 Global Wheat Commodity Price  : ${eval_row['Wheat_Price_USD']:.2f} / bushel")
log(f" 🛡️ Foreign Reserves              : ${eval_row['Foreign_Reserves_USD_Bn']:.2f} Billion")
log(f" 📦 Reserves Import Coverage      : {eval_row['Reserves_Months_of_Imports']:.1f} Months")
log(f" 📉 Current Food Inflation        : {current_food_inf:.2f}% (Headline: {eval_row['Headline_Inflation_pct']:.2f}%)")
log(f" ⚠️ Daily Composite Stress Index  : {eval_row['Daily_Economic_Stress_Index']:.1f} / 100")
log("-" * 66)
log(f" 🔮 AI 30-Day USD/BDT Forecast    : ৳ {pred_rate_30d:.2f} ({pred_return:+.2f}%)")
log(f" 🌾 AI 30-Day Food Inflation Fcst : {pred_food_inf_30d:.2f}% ({food_status})")
log(f" 🚨 Early Warning Crisis Status   : {status_symbol} ({status_color_text})")
log(f" 📈 30-Day Crisis Probability     : {crisis_prob * 100.0:.1f}%")
log("-" * 66)

# Commodity Price Projections Table
if comm_models:
    log(" 🛒 30-DAY RETAIL FOOD COMMODITY INTELLIGENCE (TCB & AI MODELS):")
    log(f"    {'Commodity':<28} | {'Current':<8} | {'Forecast 30D':<12} | {'Change':<8} | {'Status'}")
    log("    " + "-" * 74)
    COMM_NAMES = {
        'Price_Rice_Miniket_BDT_kg': ('চাল (মিনিকেট)', 'কেজি'),
        'Price_Lentil_BDT_kg': ('মসুর ডাল (মাঝারি)', 'কেজি'),
        'Price_Soybean_Oil_BDT_liter': ('সয়াবিন তেল (বোতল)', 'লিটার'),
        'Price_Flour_BDT_kg': ('আটা / ময়দা', 'কেজি'),
        'Price_Onion_BDT_kg': ('পেঁয়াজ (দেশি/আমদানি)', 'কেজি'),
        'Price_Sugar_BDT_kg': ('চিনি (প্যাকেট/খোলা)', 'কেজি')
    }
    for c_col, (c_label, c_unit) in COMM_NAMES.items():
        if c_col in comm_models and c_col in eval_row:
            m_info = comm_models[c_col]
            inp_c = pd.DataFrame([eval_row[m_info['features']]])
            delta_c = m_info['model'].predict(inp_c)[0]
            curr_c = eval_row[c_col]
            fcst_c = curr_c + delta_c
            sign_c = "+" if delta_c > 0 else ""
            status_c = "🔴 RED ALERT" if delta_c >= 4.0 else ("🟡 ELEVATED" if delta_c >= 1.5 else "🟢 STABLE")
            log(f"    {c_label:<28} | ৳{curr_c:>6.2f} | ৳{fcst_c:>10.2f} | {sign_c}৳{delta_c:>5.2f} | {status_c}")
    log("-" * 66)

if is_alert:
    log(" 💡 Recommended Central Bank Policy Interventions:")
    log("    1. Monitor and restrict non-essential / luxury import LCs.")
    log("    2. Maintain tight vigilance over interbank and kerb dollar spreads.")
    log("    3. Provide incentives to sustain official remittance inflows.")
else:
    log(" 💡 Market Assessment:")
    log("    Current indicators suggest macroeconomic stability.")
log("=" * 66)
log("✓ Daily update completed successfully!\n")
