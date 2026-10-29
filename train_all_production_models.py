"""
train_all_production_models.py
==============================
Retrains all production AI model artifacts on the updated, 
interpolation-free dataset (Point 2 As-Of FFill Fix):
1. best_forecasting_model.pkl    (USD/BDT 30-day forward return)
2. best_food_inflation_model.pkl (Food inflation 30-day forward delta)
3. best_crisis_classifier.pkl    (Macro crisis probability - Realized Outcome label)
4. best_commodity_models.pkl     (6 retail commodities 30-day price forward deltas)

All models are trained with clean, leakage-free feature sets (zero circularity).
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, f1_score, precision_score, recall_score

# Windows console encoding safeguard
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("=" * 75)
print(" [*] RETRAINING PRODUCTION AI MODELS (POINT 2 AS-OF FFILL FIX)")
print("=" * 75)

BASE_DIR = BASE_DIR
dataset_path = os.path.join(BASE_DIR, "daily_macro_master_dataset.csv")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Missing {dataset_path}")

df = pd.read_csv(dataset_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"Loaded Master Dataset: {len(df):,} days ({df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')})")

SPLIT_DATE = '2023-01-01'
HORIZON_30 = 30

# -----------------------------------------------------------------------------
# 1. USD/BDT 30-Day Return Forecaster (best_forecasting_model.pkl)
# -----------------------------------------------------------------------------
print("\n[1/4] Training USD/BDT 30-Day Return Forecasting Model...")

REG_FEATURES = [
    'USD_BDT_Rate', 'Brent_Crude_Oil_USD', 'US_Dollar_Index_DXY', 'Gold_USD_oz',
    'Natural_Gas_USD', 'Wheat_Price_USD', 'US_10Y_Treasury_pct',
    'Headline_Inflation_pct', 'Food_Inflation_pct', 'Foreign_Reserves_USD_Bn',
    'Remittance_USD_Bn', 'Reserves_Months_of_Imports', 'External_Debt_USD_Bn',
    'Short_Term_Debt_USD_Bn', 'USD_BDT_30D_Volatility', 'USD_BDT_YoY_Return_pct',
    'Oil_Shock_YoY_pct', 'DXY_30D_MA', 'Reserves_to_Short_Term_Debt_Ratio'
]

df['Target_USD_Return_30D_pct'] = (
    (df['USD_BDT_Rate'].shift(-HORIZON_30) - df['USD_BDT_Rate']) / df['USD_BDT_Rate']
) * 100

clean_reg = df.dropna(subset=['Target_USD_Return_30D_pct'] + REG_FEATURES).reset_index(drop=True)
tr_reg = clean_reg['Date'] < SPLIT_DATE
te_reg = clean_reg['Date'] >= SPLIT_DATE

X_tr_reg, y_tr_reg = clean_reg.loc[tr_reg, REG_FEATURES], clean_reg.loc[tr_reg, 'Target_USD_Return_30D_pct']
X_te_reg, y_te_reg = clean_reg.loc[te_reg, REG_FEATURES], clean_reg.loc[te_reg, 'Target_USD_Return_30D_pct']

reg_model = XGBRegressor(
    n_estimators=160, learning_rate=0.03, max_depth=4,
    subsample=0.85, colsample_bytree=0.85, random_state=42, verbosity=0
)
reg_model.fit(X_tr_reg, y_tr_reg)

pred_ret_te = reg_model.predict(X_te_reg)
actual_usd_te = clean_reg.loc[te_reg, 'USD_BDT_Rate'].values * (1.0 + y_te_reg.values / 100.0)
pred_usd_te = clean_reg.loc[te_reg, 'USD_BDT_Rate'].values * (1.0 + pred_ret_te / 100.0)
mape_usd = mean_absolute_percentage_error(actual_usd_te, pred_usd_te) * 100
rmse_usd = np.sqrt(mean_squared_error(actual_usd_te, pred_usd_te))

print(f"  + USD/BDT Model Out-of-Sample Test: MAPE = {mape_usd:.2f}%, RMSE = ৳{rmse_usd:.2f}")

joblib.dump(reg_model, os.path.join(BASE_DIR, "best_forecasting_model.pkl"))
print("  + Saved: best_forecasting_model.pkl")

# -----------------------------------------------------------------------------
# 2. Food Inflation 30-Day Delta Forecaster (best_food_inflation_model.pkl)
# -----------------------------------------------------------------------------
print("\n[2/4] Training Food Inflation 30-Day Delta Forecasting Model...")

FOOD_FEATURES = [
    'Wheat_Price_USD', 'Brent_Crude_Oil_USD', 'Natural_Gas_USD', 'Gold_USD_oz',
    'Oil_Shock_YoY_pct', 'USD_BDT_Rate', 'US_Dollar_Index_DXY',
    'Foreign_Reserves_USD_Bn', 'Reserves_Months_of_Imports',
    'USD_BDT_30D_Volatility', 'Food_Inflation_pct', 'Headline_Inflation_pct'
]

df['Target_Food_Inflation_Delta_30D'] = (
    df['Food_Inflation_pct'].shift(-HORIZON_30) - df['Food_Inflation_pct']
)

clean_food = df.dropna(subset=['Target_Food_Inflation_Delta_30D'] + FOOD_FEATURES).reset_index(drop=True)
tr_food = clean_food['Date'] < SPLIT_DATE
te_food = clean_food['Date'] >= SPLIT_DATE

X_tr_food, y_tr_food = clean_food.loc[tr_food, FOOD_FEATURES], clean_food.loc[tr_food, 'Target_Food_Inflation_Delta_30D']
X_te_food, y_te_food = clean_food.loc[te_food, FOOD_FEATURES], clean_food.loc[te_food, 'Target_Food_Inflation_Delta_30D']

food_model = XGBRegressor(
    n_estimators=180, learning_rate=0.03, max_depth=4,
    subsample=0.85, colsample_bytree=0.85, random_state=42, verbosity=0
)
food_model.fit(X_tr_food, y_tr_food)

pred_food_delta_te = food_model.predict(X_te_food)
actual_food_te = clean_food.loc[te_food, 'Food_Inflation_pct'].values + y_te_food.values
pred_food_te = clean_food.loc[te_food, 'Food_Inflation_pct'].values + pred_food_delta_te
rmse_food = np.sqrt(mean_squared_error(actual_food_te, pred_food_te))
mae_food = mean_absolute_error(actual_food_te, pred_food_te)

print(f"  + Food Inflation Out-of-Sample Test: RMSE = {rmse_food:.3f}%, MAE = {mae_food:.3f}%")

joblib.dump(food_model, os.path.join(BASE_DIR, "best_food_inflation_model.pkl"))
print("  + Saved: best_food_inflation_model.pkl")

# -----------------------------------------------------------------------------
# 3. Macro Crisis Classifier (best_crisis_classifier.pkl)
# Realized Outcome Label (Leakage-Free): BDT depr >= 3% OR reserves drop >= 5% in 30D
# -----------------------------------------------------------------------------
print("\n[3/4] Training Macro Crisis Classifier (Realized Outcome Label)...")

df['BDT_Return_30D_Fwd'] = (
    (df['USD_BDT_Rate'].shift(-HORIZON_30) - df['USD_BDT_Rate']) / df['USD_BDT_Rate']
) * 100
df['Reserve_Change_30D_Fwd'] = (
    (df['Foreign_Reserves_USD_Bn'].shift(-HORIZON_30) - df['Foreign_Reserves_USD_Bn']) / df['Foreign_Reserves_USD_Bn']
) * 100

df['Crisis_Label_Realized'] = (
    (df['BDT_Return_30D_Fwd'] >= 3.0) | (df['Reserve_Change_30D_Fwd'] <= -5.0)
).astype(int)

clean_clf = df.dropna(subset=['Crisis_Label_Realized'] + REG_FEATURES).reset_index(drop=True)
tr_clf = clean_clf['Date'] < SPLIT_DATE
te_clf = clean_clf['Date'] >= SPLIT_DATE

X_tr_clf, y_tr_clf = clean_clf.loc[tr_clf, REG_FEATURES], clean_clf.loc[tr_clf, 'Crisis_Label_Realized']
X_te_clf, y_te_clf = clean_clf.loc[te_clf, REG_FEATURES], clean_clf.loc[te_clf, 'Crisis_Label_Realized']

neg_count = (y_tr_clf == 0).sum()
pos_count = (y_tr_clf == 1).sum()
scale_pos = max(1.0, float(neg_count) / max(1.0, float(pos_count)))

clf_model = XGBClassifier(
    n_estimators=180, learning_rate=0.03, max_depth=4,
    subsample=0.85, colsample_bytree=0.85, scale_pos_weight=scale_pos,
    random_state=42, verbosity=0, eval_metric='logloss'
)
clf_model.fit(X_tr_clf, y_tr_clf)

pred_clf_te = clf_model.predict(X_te_clf)
f1_clf = f1_score(y_te_clf, pred_clf_te, zero_division=0)
prec_clf = precision_score(y_te_clf, pred_clf_te, zero_division=0)
rec_clf = recall_score(y_te_clf, pred_clf_te, zero_division=0)

print(f"  + Crisis Classifier Out-of-Sample: Precision = {prec_clf:.3f}, Recall = {rec_clf:.3f}, F1 = {f1_clf:.3f}")

joblib.dump(clf_model, os.path.join(BASE_DIR, "best_crisis_classifier.pkl"))
print("  + Saved: best_crisis_classifier.pkl")

# -----------------------------------------------------------------------------
# 4. Dedicated Commodity AI Forecasters (best_commodity_models.pkl)
# -----------------------------------------------------------------------------
print("\n[4/4] Training Dedicated Retail Commodity Models...")

COMMODITY_TARGETS = {
    'Price_Rice_Coarse_BDT_kg':    ['USD_BDT_Rate', 'Wheat_Price_USD', 'Brent_Crude_Oil_USD', 'Food_Inflation_pct', 'Oil_Shock_YoY_pct'],
    'Price_Lentil_BDT_kg':         ['USD_BDT_Rate', 'Wheat_Price_USD', 'Brent_Crude_Oil_USD', 'Food_Inflation_pct', 'Oil_Shock_YoY_pct'],
    'Price_Soybean_Oil_BDT_liter': ['USD_BDT_Rate', 'Brent_Crude_Oil_USD', 'Natural_Gas_USD', 'Food_Inflation_pct', 'Oil_Shock_YoY_pct'],
    'Price_Flour_BDT_kg':          ['USD_BDT_Rate', 'Wheat_Price_USD', 'Brent_Crude_Oil_USD', 'Food_Inflation_pct', 'Oil_Shock_YoY_pct'],
    'Price_Onion_BDT_kg':          ['USD_BDT_Rate', 'Food_Inflation_pct', 'Brent_Crude_Oil_USD', 'Oil_Shock_YoY_pct'],
    'Price_Sugar_BDT_kg':          ['USD_BDT_Rate', 'Brent_Crude_Oil_USD', 'Wheat_Price_USD', 'Food_Inflation_pct', 'Oil_Shock_YoY_pct'],
}

COMMODITY_NAMES = {
    'Price_Rice_Coarse_BDT_kg':    'মোটা চাল (স্বর্ণা/বিআর-১১)',
    'Price_Lentil_BDT_kg':         'মসুর ডাল (আমদানি)',
    'Price_Soybean_Oil_BDT_liter': 'সয়াবিন তেল (বোতল)',
    'Price_Flour_BDT_kg':          'আটা / ময়দা',
    'Price_Onion_BDT_kg':          'দেশি পেঁয়াজ',
    'Price_Sugar_BDT_kg':          'চিনি (খোলা)'
}

commodity_models = {}

for comm_col, c_feats in COMMODITY_TARGETS.items():
    if comm_col not in df.columns:
        continue
    df_c = df.copy()
    target_delta_col = f"Target_Delta_30D_{comm_col}"
    df_c[target_delta_col] = df_c[comm_col].shift(-HORIZON_30) - df_c[comm_col]
    
    clean_c = df_c.dropna(subset=[target_delta_col] + c_feats + [comm_col]).reset_index(drop=True)
    all_c_feats = c_feats + [comm_col]
    
    first_c_date = clean_c['Date'].min()
    if first_c_date < pd.to_datetime('2020-01-01'):
        c_split_date = pd.to_datetime(SPLIT_DATE)
    else:
        c_split_idx = int(len(clean_c) * 0.75)
        c_split_date = clean_c.iloc[c_split_idx]['Date']
        
    tr_c = clean_c['Date'] < c_split_date
    te_c = clean_c['Date'] >= c_split_date
    
    X_tr_c, y_tr_c = clean_c.loc[tr_c, all_c_feats], clean_c.loc[tr_c, target_delta_col]
    X_te_c, y_te_c = clean_c.loc[te_c, all_c_feats], clean_c.loc[te_c, target_delta_col]
    
    m_comm = XGBRegressor(
        n_estimators=120, learning_rate=0.03, max_depth=4,
        subsample=0.85, colsample_bytree=0.85, random_state=42, verbosity=0
    )
    m_comm.fit(X_tr_c, y_tr_c)
    
    preds_delta = m_comm.predict(X_te_c)
    actual_price = clean_c.loc[te_c, comm_col].values + y_te_c.values
    pred_price = clean_c.loc[te_c, comm_col].values + preds_delta
    mae_c = float(mean_absolute_error(actual_price, pred_price))
    mape_c = float(mean_absolute_percentage_error(actual_price, pred_price) * 100)
    r2_c = float(r2_score(actual_price, pred_price))
    
    commodity_models[comm_col] = {
        'comm_name': COMMODITY_NAMES.get(comm_col, comm_col),
        'model': m_comm,
        'features': all_c_feats,
        'mae': mae_c,
        'metrics': {
            'mae_bdt': mae_c,
            'mape_pct': mape_c,
            'r2_score': r2_c
        }
    }
    print(f"  + {comm_col}: OOS MAE = ৳{mae_c:.2f}/unit | MAPE = {mape_c:.2f}% | R² = {r2_c:.4f}")

joblib.dump(commodity_models, os.path.join(BASE_DIR, "best_commodity_models.pkl"))
print("  + Saved: best_commodity_models.pkl")

print("\n" + "=" * 75)
print(" [+] ALL PRODUCTION AI MODELS SUCCESSFULLY RETRAINED & SAVED!")
print("=" * 75)
