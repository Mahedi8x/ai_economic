import os
import re

p1 = r"C:\ai_economic\scripts\build_mixed_frequency_monthly_dataset.py"
with open(p1, "r", encoding="utf-8") as f:
    txt = f.read()

old_block = """# Merge on Date (1st of each month)
df_merged = pd.merge(df_market_monthly, df_bb, on='Date', how='inner')"""

new_block = """# Merge on Date (1st of each month)
bb_cols = ['Date', 'Headline_Inflation_pct', 'Food_Inflation_pct', 'USD_BDT_Exchange_Rate', 
           'Foreign_Reserves_USD_Bn', 'Remittance_USD_Bn']
df_bb = df_bb[[c for c in bb_cols if c in df_bb.columns]]
df_merged = pd.merge(df_market_monthly, df_bb, on='Date', how='inner')"""

txt = txt.replace(old_block, new_block)
with open(p1, "w", encoding="utf-8") as f:
    f.write(txt)
print("Updated build_mixed_frequency_monthly_dataset.py")

# Update train_monthly_models_and_validate.py
p2 = r"C:\ai_economic\scripts\train_monthly_models_and_validate.py"
with open(p2, "r", encoding="utf-8") as f:
    txt2 = f.read()

old_fc = "feature_cols = [c for c in df.columns if c not in non_features]"
new_fc = "feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in non_features]"

txt2 = txt2.replace(old_fc, new_fc)
with open(p2, "w", encoding="utf-8") as f:
    f.write(txt2)
print("Updated train_monthly_models_and_validate.py")
