import os
import sys
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

# Windows console encoding safeguard
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("==================================================================")
print(" 🚀 STEP 3: TRAINING DEDICATED COMMODITY AI FORECASTING MODELS")
print("    Targets: 30-Day Forward Retail Price for 6 Key Commodities")
print("==================================================================")

BASE_DIR = r"C:\ai_economic"
dataset_path = os.path.join(BASE_DIR, "daily_macro_master_dataset.csv")

if not os.path.exists(dataset_path):
    print(f"❌ Error: {dataset_path} not found.")
    sys.exit(1)

df = pd.read_csv(dataset_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

COMMODITIES = {
    'Price_Rice_Miniket_BDT_kg': 'চাল (মিনিকেট / নাজিরশাইল)',
    'Price_Rice_Coarse_BDT_kg': 'চাল (মোটা / স্বর্ণা / বিআর-১১)',
    'Price_Lentil_BDT_kg': 'মসুর ডাল (মাঝারি দানা)',
    'Price_Soybean_Oil_BDT_liter': 'সয়াবিন তেল (বোতলজাত)',
    'Price_Flour_BDT_kg': 'আটা / ময়দা (প্যাকেটজাত)',
    'Price_Onion_BDT_kg': 'পেঁয়াজ (দেশি / আমদানিকৃত)',
    'Price_Sugar_BDT_kg': 'চিনি (খোলা / প্যাকেট)'
}

HORIZON = 30
SPLIT_DATE = '2023-01-01'

# Feature set for commodities (Circular stress index removed)
BASE_FEATURES = [
    'USD_BDT_Rate', 'Brent_Crude_Oil_USD', 'Wheat_Price_USD',
    'Food_Inflation_pct', 'Headline_Inflation_pct',
    'Oil_Shock_YoY_pct', 'US_Dollar_Index_DXY', 'Foreign_Reserves_USD_Bn'
]

commodity_models = {}
evaluation_results = []

for comm_col, comm_name in COMMODITIES.items():
    print(f"\n--- Training Dedicated AI Forecaster for: {comm_name} ({comm_col}) ---")
    
    # 30-Day Forward Target Delta
    df_c = df.copy()
    target_col = f"Target_Delta_30D_{comm_col}"
    df_c[target_col] = df_c[comm_col].shift(-HORIZON) - df_c[comm_col]
    
    # Clean NaN values
    df_clean = df_c.dropna(subset=[target_col] + BASE_FEATURES + [comm_col]).reset_index(drop=True)
    
    # Features include base macro indicators + commodity's own current price
    features = BASE_FEATURES + [comm_col]
    
    # Chronological Split
    train_mask = df_clean['Date'] < SPLIT_DATE
    test_mask = df_clean['Date'] >= SPLIT_DATE
    
    X_train = df_clean.loc[train_mask, features]
    y_train = df_clean.loc[train_mask, target_col]
    
    X_test = df_clean.loc[test_mask, features]
    y_test = df_clean.loc[test_mask, target_col]
    
    # Train XGBoost Regressor
    model = XGBRegressor(
        n_estimators=180,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred_delta = model.predict(X_test)
    y_true_level = df_clean.loc[test_mask, comm_col] + y_test
    y_pred_level = df_clean.loc[test_mask, comm_col] + y_pred_delta
    
    mae_delta = mean_absolute_error(y_test, y_pred_delta)
    mae_level = mean_absolute_error(y_true_level, y_pred_level)
    mape_level = mean_absolute_percentage_error(y_true_level, y_pred_level) * 100.0
    r2_level = r2_score(y_true_level, y_pred_level)
    
    # Today's live inference
    latest_row = df.iloc[-1]
    input_today = pd.DataFrame([latest_row[features]])
    today_pred_delta = model.predict(input_today)[0]
    today_curr_price = latest_row[comm_col]
    today_forecast_30d = today_curr_price + today_pred_delta
    
    commodity_models[comm_col] = {
        'model': model,
        'features': features,
        'comm_name': comm_name,
        'metrics': {
            'mae_bdt': round(mae_level, 2),
            'mape_pct': round(mape_level, 2),
            'r2_score': round(r2_level, 4)
        }
    }
    
    status_label = "🔴 রেড অ্যালার্ট" if today_pred_delta >= 4.0 else ("🟡 চাপ বাড়ছে / ঊর্ধ্বমুখী" if today_pred_delta >= 1.5 else "🟢 সহনীয়")
    
    evaluation_results.append({
        'Commodity': comm_name,
        'Column': comm_col,
        'Current_Price': round(today_curr_price, 2),
        'Predicted_30D_Price': round(today_forecast_30d, 2),
        'Delta_BDT': round(today_pred_delta, 2),
        'MAE_BDT': round(mae_level, 2),
        'MAPE_pct': round(mape_level, 2),
        'R2': round(r2_level, 4),
        'Status': status_label
    })
    
    print(f"  ✓ Train Set: {len(X_train)} days | Test Set: {len(X_test)} days (2023-2026)")
    print(f"  ✓ Out-of-Sample MAE: ৳ {mae_level:.2f} | MAPE: {mape_level:.2f}% | R²: {r2_level:.4f}")
    print(f"  🔮 Today: ৳ {today_curr_price:.2f} -> 30-Day Forecast: ৳ {today_forecast_30d:.2f} (Delta: {today_pred_delta:+.2f} ৳) [{status_label}]")

# Save Models Artifact
models_output_path = os.path.join(BASE_DIR, "best_commodity_models.pkl")
joblib.dump(commodity_models, models_output_path)
print(f"\n✓ Saved all commodity models to: {models_output_path}")

# Summary Table
df_eval = pd.DataFrame(evaluation_results)
print("\n" + "=" * 80)
print(" 📊 COMMODITY FORECASTING PERFORMANCE & LIVE 30-DAY INTELLIGENCE BULLETIN")
print("=" * 80)
print(df_eval[['Commodity', 'Current_Price', 'Predicted_30D_Price', 'Delta_BDT', 'MAE_BDT', 'MAPE_pct', 'Status']].to_string(index=False))
print("=" * 80)
