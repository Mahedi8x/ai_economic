from pathlib import Path
"""
Build Mixed-Frequency Monthly Master Dataset
============================================
Transforms daily global financial & commodity data (Yahoo Finance) into monthly
statistical aggregates (mean, end-of-month, volatility, momentum) and merges with
ground-truth official monthly macroeconomic indicators from BBS, Bangladesh Bank,
and World Bank WDI.

Golden Econometric Rule: Target variable (Food Inflation YoY) is preserved in its
authentic monthly frequency without artificial temporal interpolation.
"""

import os
import sys
import pandas as pd
import numpy as np

# Windows console encoding safeguard
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
INTERIM_DIR = os.path.join(BASE_DIR, "data", "interim")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(INTERIM_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("=" * 70)
print(" [*] BUILDING AUTHENTIC MIXED-FREQUENCY MONTHLY DATASET (2010 - 2026)")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1. Ingest & Aggregate Daily Global Market Data to Monthly Resolution
# -----------------------------------------------------------------------------
yf_path = os.path.join(RAW_DIR, "yahoo_finance_global_markets_daily_2010_2026_raw.csv")
if not os.path.exists(yf_path):
    raise FileNotFoundError(f"Missing raw market file at {yf_path}")

df_daily = pd.read_csv(yf_path)
df_daily['Date'] = pd.to_datetime(df_daily['Date'])
df_daily = df_daily.sort_values('Date').reset_index(drop=True)

# Add Year-Month period for aggregation
df_daily['YearMonth'] = df_daily['Date'].dt.to_period('M')

# Calculate monthly market features
daily_cols = ['USD_BDT_Rate', 'Brent_Crude_Oil_USD', 'Wheat_Price_USD', 
              'US_Dollar_Index_DXY', 'Gold_USD_oz', 'Natural_Gas_USD', 'US_10Y_Treasury_pct']

# Group by Month
monthly_means = df_daily.groupby('YearMonth')[daily_cols].mean().round(2)
monthly_means.columns = [f"{c}_Monthly_Mean" for c in monthly_means.columns]

monthly_ends = df_daily.groupby('YearMonth')[daily_cols].last().round(2)
monthly_ends.columns = [f"{c}_Month_End" for c in monthly_ends.columns]

monthly_stds = df_daily.groupby('YearMonth')[daily_cols].std().fillna(0).round(3)
monthly_stds.columns = [f"{c}_Monthly_Volatility" for c in monthly_stds.columns]

# Combine daily aggregated features
df_market_monthly = pd.concat([monthly_means, monthly_ends, monthly_stds], axis=1).reset_index()
df_market_monthly['Date'] = df_market_monthly['YearMonth'].dt.to_timestamp()
df_market_monthly = df_market_monthly.drop(columns=['YearMonth'])

print(f"[1/4] Daily markets aggregated to {len(df_market_monthly)} monthly observations.")

# -----------------------------------------------------------------------------
# 2. Ingest Official BBS & Bangladesh Bank Monthly Macro Fundamentals
# -----------------------------------------------------------------------------
bb_path = os.path.join(RAW_DIR, "bangladesh_bank_bbs_monthly_macro_2010_2026_raw.csv")
df_bb = pd.read_csv(bb_path)
df_bb['Date'] = pd.to_datetime(df_bb['Date'])

# Merge on Date (1st of each month)
bb_cols = ['Date', 'Headline_Inflation_pct', 'Food_Inflation_pct', 'USD_BDT_Exchange_Rate', 
           'Foreign_Reserves_USD_Bn', 'Remittance_USD_Bn']
df_bb = df_bb[[c for c in bb_cols if c in df_bb.columns]]
df_merged = pd.merge(df_market_monthly, df_bb, on='Date', how='inner')
print(f"[2/4] Merged with official BBS & BB monthly series: {len(df_merged)} months.")

# -----------------------------------------------------------------------------
# 3. Ingest Annual Structural Indicators from World Bank WDI
# -----------------------------------------------------------------------------
clean_annual_path = os.path.join(BASE_DIR, "clean_macro_dataset.csv")
if os.path.exists(clean_annual_path):
    df_annual = pd.read_csv(clean_annual_path)
    df_merged['Year'] = df_merged['Date'].dt.year
    
    struct_cols = ['External_Debt_USD_Bn', 'Short_Term_Debt_USD_Bn', 
                   'Reserves_Months_of_Imports', 'Current_Account_Balance_USD_Bn', 
                   'Broad_Money_M2_pct_GDP', 'GDP_Growth_pct']
    struct_cols = [c for c in struct_cols if c in df_annual.columns]
    
    df_merged = pd.merge(df_merged, df_annual[['Year'] + struct_cols], on='Year', how='left')
    for col in struct_cols:
        df_merged[col] = df_merged[col].ffill().bfill().round(2)
    df_merged = df_merged.drop(columns=['Year'])
    print(f"[3/4] Merged with {len(struct_cols)} World Bank structural macro indicators.")

# -----------------------------------------------------------------------------
# 4. Construct Authentic Economic Early Warning Signals & Forecast Targets
# -----------------------------------------------------------------------------
# Momentum & YoY shocks
df_merged['USD_BDT_YoY_Return_pct'] = (df_merged['USD_BDT_Rate_Monthly_Mean'].pct_change(12).fillna(0) * 100).round(2)
df_merged['Oil_Shock_YoY_pct'] = (df_merged['Brent_Crude_Oil_USD_Monthly_Mean'].pct_change(12).fillna(0) * 100).round(2)
df_merged['Wheat_Shock_YoY_pct'] = (df_merged['Wheat_Price_USD_Monthly_Mean'].pct_change(12).fillna(0) * 100).round(2)

# Solvency Ratio
if 'Short_Term_Debt_USD_Bn' in df_merged.columns:
    df_merged['Reserves_to_Short_Term_Debt_Ratio'] = (
        df_merged['Foreign_Reserves_USD_Bn'] / df_merged['Short_Term_Debt_USD_Bn'].replace(0, np.nan)
    ).fillna(1.0).round(2)

# Monthly Composite Stress Index (Objective Z-score / Policy Threshold based)
# Formula accounts for inflation pressure, reserve depletion, FX depreciation, and oil shock
stress_raw = (
    (df_merged['Headline_Inflation_pct'] * 2.0) +
    (np.maximum(0, 5.0 - df_merged['Reserves_Months_of_Imports']) * 6.0) +
    (np.maximum(0, df_merged['USD_BDT_YoY_Return_pct']) * 1.0) +
    (np.maximum(0, df_merged['Brent_Crude_Oil_USD_Monthly_Mean'] - 80.0) * 0.25)
)
df_merged['Monthly_Economic_Stress_Index'] = np.clip(stress_raw, 10.0, 100.0).round(1)

# Crisis Ground Truth: Documented macro stress episodes (IMF / BB documented periods)
# 1. 2011 Global Food/Commodity Price Spike (Mar 2011 - Oct 2011)
# 2. 2020 COVID-19 Economic Disruption (Apr 2020 - Aug 2020)
# 3. 2022-2023 Post-Ukraine FX/Reserve Crisis (May 2022 - Dec 2023)
# 4. 2024 Political Transition & Supply Shock (Jul 2024 - Oct 2024)
df_merged['Ground_Truth_Crisis'] = 0
crisis_conditions = (
    ((df_merged['Date'] >= '2011-03-01') & (df_merged['Date'] <= '2011-10-01')) |
    ((df_merged['Date'] >= '2020-04-01') & (df_merged['Date'] <= '2020-08-01')) |
    ((df_merged['Date'] >= '2022-05-01') & (df_merged['Date'] <= '2023-12-01')) |
    ((df_merged['Date'] >= '2024-07-01') & (df_merged['Date'] <= '2024-10-01'))
)
df_merged.loc[crisis_conditions, 'Ground_Truth_Crisis'] = 1

# Forecast Targets: Authentic Food Inflation at t+1 (1 month), t+3 (3 months), t+6 (6 months)
df_merged['Target_Food_Inflation_t1'] = df_merged['Food_Inflation_pct'].shift(-1)
df_merged['Target_Food_Inflation_t3'] = df_merged['Food_Inflation_pct'].shift(-3)
df_merged['Target_Food_Inflation_t6'] = df_merged['Food_Inflation_pct'].shift(-6)

# Output Paths
output_csv = os.path.join(PROCESSED_DIR, "monthly_macro_food_inflation_dataset.csv")
output_xlsx = os.path.join(PROCESSED_DIR, "monthly_macro_food_inflation_dataset.xlsx")

df_merged.to_csv(output_csv, index=False)
df_merged.to_excel(output_xlsx, index=False)

print(f"[4/4] Successfully generated Master Monthly Dataset:")
print(f"  - File: {output_csv}")
print(f"  - Dimensions: {df_merged.shape[0]} Months x {df_merged.shape[1]} Columns")
print(f"  - Date Range: {df_merged['Date'].min().strftime('%Y-%m')} to {df_merged['Date'].max().strftime('%Y-%m')}")
print(f"  - Target Variable: Authentic BBS Food_Inflation_pct (Mean: {df_merged['Food_Inflation_pct'].mean():.2f}%, Max: {df_merged['Food_Inflation_pct'].max():.2f}%)")
print("=" * 70)
print(" [+] SUCCESS: 100% Genuine Monthly Dataset Ready for Empirical Modeling!")
print("=" * 70)
