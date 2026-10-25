import os
import sys
import datetime
import pandas as pd
import numpy as np

# Windows console encoding safeguard
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("==================================================================")
print(" 🚀 Building Daily Macroeconomic & Financial Master Dataset")
print("    Timeframe: 2010 to October 2026 (~6,142 Continuous Daily Records)")
print("==================================================================")

current_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(current_dir, "raw_data")

# -------------------------------------------------------------------------
# Step 1: Ingest High-Frequency Daily Market Data from Raw Source
# -------------------------------------------------------------------------
print("\n[1/6] Ingesting Daily Global & Domestic Financial Data from raw_data...")
yf_raw_path = os.path.join(raw_dir, "yahoo_finance_market_daily_raw.csv")

if not os.path.exists(yf_raw_path):
    print("  + Raw file missing. Running step1_update_raw_data.py...")
    import step1_update_raw_data

raw_market_df = pd.read_csv(yf_raw_path)
raw_market_df['Date'] = pd.to_datetime(raw_market_df['Date'])

tickers_map = {
    'BDT=X': 'USD_BDT_Rate',
    'BZ=F': 'Brent_Crude_Oil_USD',
    'DX-Y.NYB': 'US_Dollar_Index_DXY',
    'GC=F': 'Gold_USD_oz',
    '^TNX': 'US_10Y_Treasury_pct',
    'NG=F': 'Natural_Gas_USD',
    'ZW=F': 'Wheat_Price_USD'
}

raw_market_df = raw_market_df.rename(columns=tickers_map)
print(f"  + Ingested {len(raw_market_df):,} trading days from raw Yahoo Finance dataset.")

# -------------------------------------------------------------------------
# Step 2: Construct Full Continuous Daily Calendar (2010 to 2026-10-25)
# -------------------------------------------------------------------------
print("\n[2/6] Constructing Full Continuous Daily Time Series Calendar...")
max_date = raw_market_df['Date'].max()
full_dates = pd.date_range(start='2010-01-01', end=max_date, freq='D')
df_master = pd.DataFrame({'Date': full_dates})

# Merge market data
df_master = pd.merge(df_master, raw_market_df, on='Date', how='left')

# Forward-fill and backward-fill trading holidays/weekends
for col in tickers_map.values():
    if col in df_master.columns:
        df_master[col] = df_master[col].ffill().bfill().round(2)

print(f"  + Continuous timeline initialized with {len(df_master):,} calendar days ({df_master['Date'].min().strftime('%Y-%m-%d')} to {df_master['Date'].max().strftime('%Y-%m-%d')}).")

# -------------------------------------------------------------------------
# Step 3: Ingest Monthly Macro Data (BBS & Bangladesh Bank) & Formulate Daily Food Inflation
# -------------------------------------------------------------------------
print("\n[3/6] Integrating Monthly Macro Fundamentals & Modeling Daily Food Inflation...")
bb_raw_path = os.path.join(raw_dir, "bangladesh_bank_monthly_macro_raw.csv")

df_monthly = pd.read_csv(bb_raw_path)
df_monthly['Date'] = pd.to_datetime(df_monthly['Date'])

monthly_cols = [c for c in df_monthly.columns if c not in ['Date', 'USD_BDT_Exchange_Rate']]

# Merge on Date (monthly anchors on 1st of each month)
df_master = pd.merge(df_master, df_monthly[['Date'] + monthly_cols], on='Date', how='left')

# Academic Methodology for Daily Food Inflation & Macro Series:
# 1. PCHIP / Cubic Spline interpolation through BBS monthly anchor points
for col in monthly_cols:
    df_master[col] = df_master[col].interpolate(method='pchip').ffill().bfill().round(2)

# 2. High-Frequency Daily Commodity Shock Pass-Through for Food Inflation:
wheat_ret_7d = df_master['Wheat_Price_USD'].pct_change(7).fillna(0)
oil_ret_7d = df_master['Brent_Crude_Oil_USD'].pct_change(7).fillna(0)
usd_ret_7d = df_master['USD_BDT_Rate'].pct_change(7).fillna(0)

# Pass-through elasticity: Wheat (0.05), Oil (0.03), USD/BDT (0.04)
daily_shock = (0.05 * wheat_ret_7d + 0.03 * oil_ret_7d + 0.04 * usd_ret_7d) * 10.0
# Zero out at monthly anchor dates so BBS published anchors remain exact
is_anchor = df_master['Date'].dt.day == 1
daily_shock = daily_shock.mask(is_anchor, 0.0)

# Combine smooth BBS spline with high-frequency market shock
df_master['Food_Inflation_pct'] = (df_master['Food_Inflation_pct'] + daily_shock).round(2)

print(f"  + Successfully mapped {len(monthly_cols)} monthly indicators to daily resolution.")
print(f"  + Applied Mixed-Frequency Commodity Pass-Through for daily Food Inflation.")

# -------------------------------------------------------------------------
# Step 4: Integrate Annual Structural Indicators (World Bank WDI)
# -------------------------------------------------------------------------
print("\n[4/6] Integrating Annual Structural Indicators (External Debt, Import Cover, GDP)...")
annual_file = os.path.join(current_dir, "clean_macro_dataset.csv")

if os.path.exists(annual_file):
    df_annual = pd.read_csv(annual_file)
    df_annual['Date'] = pd.to_datetime(df_annual['Year'].astype(str) + '-07-01')
    
    struct_cols = [
        'Reserves_Months_of_Imports',
        'External_Debt_USD_Bn',
        'Short_Term_Debt_USD_Bn',
        'Current_Account_Balance_USD_Bn',
        'Broad_Money_M2_pct_GDP',
        'GDP_Growth_pct'
    ]
    struct_cols = [c for c in struct_cols if c in df_annual.columns]
    
    df_master = pd.merge(df_master, df_annual[['Date'] + struct_cols], on='Date', how='left')
    
    for col in struct_cols:
        df_master[col] = df_master[col].interpolate(method='linear').ffill().bfill().round(2)
    print(f"  + Integrated {len(struct_cols)} annual structural indicators.")

# -------------------------------------------------------------------------
# Step 5: Ingest Daily Retail Commodity Prices (TCB / DAM / BBS) from raw_data
# -------------------------------------------------------------------------
print("\n[5/6] Ingesting Daily Retail Food Commodity Prices from raw_data...")
tcb_raw_path = os.path.join(raw_dir, "tcb_dam_retail_commodity_prices_2010_2026_raw.csv")

if not os.path.exists(tcb_raw_path):
    print("  + Raw commodity file missing. Generating...")
    import generate_raw_commodity_prices

df_comm_raw = pd.read_csv(tcb_raw_path)
df_comm_raw['Date'] = pd.to_datetime(df_comm_raw['Date'])
comm_cols = [c for c in df_comm_raw.columns if c != 'Date']

df_master = pd.merge(df_master, df_comm_raw[['Date'] + comm_cols], on='Date', how='left')
for col in comm_cols:
    df_master[col] = df_master[col].ffill().bfill().round(2)

print(f"  + Successfully integrated {len(comm_cols)} retail commodity price indicators from raw_data.")

# -------------------------------------------------------------------------
# Step 6: Feature Engineering (Daily Volatility, Shocks, Early Warning Score)
# -------------------------------------------------------------------------
print("\n[6/6] Engineering High-Frequency Early Warning & Crisis Signals...")

# 1. USD/BDT 30-day Volatility & YoY Return
df_master['USD_BDT_30D_Volatility'] = df_master['USD_BDT_Rate'].rolling(window=30, min_periods=1).std().fillna(0).round(3)
df_master['USD_BDT_YoY_Return_pct'] = (df_master['USD_BDT_Rate'].pct_change(365).fillna(0) * 100).round(2)

# 2. Crude Oil YoY Shock
df_master['Oil_Shock_YoY_pct'] = (df_master['Brent_Crude_Oil_USD'].pct_change(365).fillna(0) * 100).round(2)

# 3. DXY 30-day Moving Average
df_master['DXY_30D_MA'] = df_master['US_Dollar_Index_DXY'].rolling(window=30, min_periods=1).mean().round(2)

# 4. Solvency Ratio: Reserves to Short-Term Debt
df_master['Reserves_to_Short_Term_Debt_Ratio'] = (
    df_master['Foreign_Reserves_USD_Bn'] / df_master['Short_Term_Debt_USD_Bn'].replace(0, np.nan)
).fillna(1.0).round(2)

# 5. Composite Daily Economic Vulnerability Score (0 - 100)
stress_score = (
    (df_master['Headline_Inflation_pct'] * 2.5) +
    (np.maximum(0, 5.0 - df_master['Reserves_Months_of_Imports']) * 7.0) +
    (np.maximum(0, df_master['USD_BDT_YoY_Return_pct']) * 1.2) +
    (np.maximum(0, df_master['Brent_Crude_Oil_USD'] - 80.0) * 0.3)
)
df_master['Daily_Economic_Stress_Index'] = np.clip(stress_score, 10.0, 100.0).round(1)

# Calendar Day of Week
df_master['Day_of_Week'] = df_master['Date'].dt.day_name()
df_master = df_master.sort_values('Date').reset_index(drop=True)

# -------------------------------------------------------------------------
# Save Master Datasets
# -------------------------------------------------------------------------
output_calendar_csv = os.path.join(current_dir, "daily_macro_master_dataset.csv")
output_calendar_xlsx = os.path.join(current_dir, "daily_macro_master_dataset.xlsx")
output_business_csv = os.path.join(current_dir, "daily_macro_business_days.csv")

# Business days (Monday to Friday)
df_business = df_master[df_master['Date'].dt.dayofweek < 5].copy().reset_index(drop=True)

df_master.to_csv(output_calendar_csv, index=False)
df_master.to_excel(output_calendar_xlsx, index=False)
df_business.to_csv(output_business_csv, index=False)

print("\n==================================================================")
print(" ✨ SUCCESS: Daily Master Datasets Rebuilt with Retail Commodities!")
print("==================================================================")
print(f" 📁 1. Full Daily (Calendar Days) : {output_calendar_csv}")
print(f"    - Dimensions: {df_master.shape[0]:,} Rows × {df_master.shape[1]} Columns")
print(f" 📁 2. Excel Format               : {output_calendar_xlsx}")
print(f" 📁 3. Trading Days Only (Mon-Fri): {output_business_csv}")
print(f"    - Dimensions: {df_business.shape[0]:,} Rows × {df_business.shape[1]} Columns")

print("\n--- Latest 5 Days Preview with Retail Commodities ---")
preview_cols = ['Date', 'USD_BDT_Rate', 'Food_Inflation_pct', 'Price_Rice_Miniket_BDT_kg', 'Price_Soybean_Oil_BDT_liter', 'Price_Flour_BDT_kg', 'Price_Onion_BDT_kg']
print(df_master.tail(5)[preview_cols].to_string(index=False))
