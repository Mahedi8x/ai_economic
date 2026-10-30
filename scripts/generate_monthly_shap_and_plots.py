from pathlib import Path
"""
Generate SHAP Explainability & Publication Plots for Monthly Framework
======================================================================
Generates TreeSHAP summary plot, feature ranking, and crisis alert backtest
figures using the trained mixed-frequency monthly models.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

# Windows console encoding safeguard
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "monthly_macro_food_inflation_dataset.csv")
MODELS_PATH = os.path.join(BASE_DIR, "models", "mixed_frequency_monthly_models.pkl")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 70)
print(" [*] GENERATING SHAP ATTRIBUTION & PUBLICATION FIGURES")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])

with open(MODELS_PATH, "rb") as f:
    models = pickle.load(f)

xgb_t6 = models.get("xgb_Target_Food_Inflation_t6")
if xgb_t6 is None:
    raise ValueError("xgb_Target_Food_Inflation_t6 model not found")

non_features = ['Date', 'Target_Food_Inflation_t1', 'Target_Food_Inflation_t3', 
                'Target_Food_Inflation_t6', 'Ground_Truth_Crisis']
feature_cols = [c for c in df.columns if c not in non_features]

X = df[feature_cols].fillna(0)

# Compute TreeSHAP
print("[1/2] Computing TreeSHAP values for XGBoost model...")
explainer = shap.TreeExplainer(xgb_t6)
shap_values = explainer.shap_values(X)

# Global Feature Importance
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
feat_imp = pd.DataFrame({
    'Feature': feature_cols,
    'Mean_SHAP_Importance': mean_abs_shap
}).sort_values('Mean_SHAP_Importance', ascending=False).reset_index(drop=True)

feat_imp['Cumulative_Pct'] = (feat_imp['Mean_SHAP_Importance'] / feat_imp['Mean_SHAP_Importance'].sum() * 100).cumsum()

top_10 = feat_imp.head(10)
print("\n--- Top 10 Features by SHAP Attribution ---")
print(top_10.to_string(index=False))

# Save SHAP table
feat_imp.to_csv(os.path.join(BASE_DIR, "monthly_shap_feature_importance.csv"), index=False)

# Plot 1: SHAP Bar Plot
plt.figure(figsize=(10, 6), dpi=300)
plt.barh(top_10['Feature'][::-1], top_10['Mean_SHAP_Importance'][::-1], color='#2563EB', edgecolor='#1D4ED8')
plt.xlabel('Mean |SHAP Value| (Impact on Food Inflation Forecast)', fontsize=11, fontweight='bold')
plt.title('Top 10 Drivers of Bangladesh Food Inflation (6-Month Horizon)', fontsize=13, fontweight='bold', pad=15)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
shap_bar_path = os.path.join(FIGURES_DIR, "monthly_shap_feature_importance.png")
plt.savefig(shap_bar_path)
plt.close()
print(f"[2/2] Saved SHAP figure to: {shap_bar_path}")

print("=" * 70)
print(" [+] SHAP Explainability Generation Complete!")
print("=" * 70)
