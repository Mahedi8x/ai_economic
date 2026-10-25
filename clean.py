import os
import sys
import pandas as pd
import numpy as np

# Windows কনসোলে যাতে কোনো এনকোডিং এরর না আসে
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ১. ফাইলের পাথ স্বয়ংক্রিয়ভাবে সনাক্তকরণ (Auto-detect file path)
current_dir = os.path.dirname(os.path.abspath(__file__))

possible_paths = [
    os.path.join(current_dir, "Data.csv"),
    os.path.join(current_dir, "..", "world_bank_wdi_bangladesh_raw.xls"),
    os.path.join(current_dir, "world_bank_wdi_bangladesh_raw.xls"),
    "Data.csv",
    "world_bank_wdi_bangladesh_raw.xls"
]

data_file = None
for path in possible_paths:
    if os.path.exists(path):
        data_file = path
        break

if not data_file:
    raise FileNotFoundError("Data file not found! Please place 'Data.csv' or 'world_bank_wdi_bangladesh_raw.xls' in the folder.")

print(f"Loading data from: {data_file}")

# ২. ফাইল পড়া (CSV অথবা Excel উভয় ফরম্যাটই সাপোর্ট করে)
if data_file.lower().endswith('.csv'):
    df_raw = pd.read_csv(data_file, skiprows=3)
else:
    df_raw = pd.read_excel(data_file, sheet_name="Data", skiprows=3)

# ৩. প্রয়োজনীয় ইন্ডিকেটরগুলোর ম্যাপিং ডিকশনারি
indicator_mapping = {
    'FP.CPI.TOTL.ZG': 'Inflation_CPI_pct',
    'FP.CPI.TOTL': 'CPI_Index_2010_100',
    'PA.NUS.FCRF': 'Exchange_Rate_USD_BDT',
    'FI.RES.TOTL.CD': 'Foreign_Reserves_USD_Bn',
    'FI.RES.TOTL.MO': 'Reserves_Months_of_Imports',
    'BX.TRF.PWKR.CD.DT': 'Remittance_USD_Bn',
    'BX.TRF.PWKR.DT.GD.ZS': 'Remittance_pct_GDP',
    'NY.GDP.MKTP.KD.ZG': 'GDP_Growth_pct',
    'NY.GDP.MKTP.CD': 'GDP_USD_Bn',
    'DT.DOD.DECT.CD': 'External_Debt_USD_Bn',
    'DT.DOD.DSTC.CD': 'Short_Term_Debt_USD_Bn',
    'BN.CAB.XOKA.CD': 'Current_Account_Balance_USD_Bn',
    'NE.EXP.GNFS.ZS': 'Exports_pct_GDP',
    'NE.IMP.GNFS.ZS': 'Imports_pct_GDP',
    'FM.LBL.BMNY.GD.ZS': 'Broad_Money_M2_pct_GDP',
    'AG.PRD.FOOD.XD': 'Food_Production_Index'
}

# ৪. শুধু দরকারি ইন্ডিকেটরগুলো ফিল্টার করা
df_filtered = df_raw[df_raw['Indicator Code'].isin(indicator_mapping.keys())].copy()
df_filtered['Metric'] = df_filtered['Indicator Code'].map(indicator_mapping)

# ৫. ১৯৮০ থেকে ২০২৪ সাল পর্যন্ত বছরগুলো ডাইনামিকালি ম্যাচ করা ('1980' বা '1980.0' উভয় ফরম্যাটেই)
year_col_map = {}
for col in df_filtered.columns:
    try:
        y = int(float(col))
        if 1980 <= y <= 2024:
            year_col_map[col] = y
    except (ValueError, TypeError):
        pass

# ৬. Wide Format থেকে Long Format-এ রূপান্তর (Melt -> Pivot)
df_melted = df_filtered.melt(
    id_vars=['Metric'], 
    value_vars=list(year_col_map.keys()), 
    var_name='Year_Raw', 
    value_name='Value'
)
df_melted['Year'] = df_melted['Year_Raw'].map(year_col_map)

# Pivot টেবিল তৈরি (Year হবে Row, Metrics হবে Columns)
df_clean = df_melted.pivot(index='Year', columns='Metric', values='Value').reset_index()
df_clean = df_clean.sort_values('Year').reset_index(drop=True)

# ৭. বড় মানগুলোকে বিলিয়ন ডলারে (Billion USD) রূপান্তর
usd_cols_to_scale = [
    'Foreign_Reserves_USD_Bn', 
    'Remittance_USD_Bn', 
    'GDP_USD_Bn', 
    'External_Debt_USD_Bn',
    'Short_Term_Debt_USD_Bn',
    'Current_Account_Balance_USD_Bn'
]

for col in usd_cols_to_scale:
    if col in df_clean.columns:
        df_clean[col] = (df_clean[col] / 1e9).round(2)

# ৮. মিসিং ভ্যালু ক্লিন করা (Forward fill + Backward fill)
df_clean = df_clean.ffill().bfill()

# ৯. দশমিক মান ২ ঘরে রাউন্ড করা
for col in df_clean.columns:
    if col != 'Year':
        df_clean[col] = df_clean[col].round(2)

# ১০. পরিষ্কার ডেটা সেভ করা (CSV ও Excel উভয় ফরম্যাটে)
output_csv = os.path.join(current_dir, "clean_macro_dataset.csv")
output_xlsx = os.path.join(current_dir, "clean_macro_dataset.xlsx")

df_clean.to_csv(output_csv, index=False)
df_clean.to_excel(output_xlsx, index=False)

print("[OK] Data cleaning completed successfully!")
print(f"Saved CSV : {output_csv}")
print(f"Saved XLSX: {output_xlsx}")
print("\n--- Latest 5 Years Preview ---")
preview_cols = ['Year', 'Inflation_CPI_pct', 'Exchange_Rate_USD_BDT', 'Foreign_Reserves_USD_Bn', 'Reserves_Months_of_Imports', 'GDP_Growth_pct']
available_preview = [c for c in preview_cols if c in df_clean.columns]
print(df_clean.tail(5)[available_preview].to_string(index=False))