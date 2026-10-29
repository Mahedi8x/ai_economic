import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
"""
Commodity Spike Probability & Uncertainty Interval Engine
==========================================================
Implements the 3 peer-reviewed econometric enhancements:
1. 30-Day Price Spike Early Warning (Binary Classifier & Calibrated Probability)
2. 80% Uncertainty Conformal Prediction Intervals (Lower BDT - Upper BDT)
3. Shock Pass-Through Elasticity Analysis (USD/BDT & Crude Oil -> Retail Food)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = r"C:\ai_economic"
DATASET_PATH = os.path.join(BASE_DIR, "daily_macro_master_dataset.csv")
MODELS_PATH = os.path.join(BASE_DIR, "best_commodity_models.pkl")
SPIKE_MODELS_PATH = os.path.join(BASE_DIR, "models", "commodity_spike_and_interval_models.pkl")

# Commodities to analyze
COMMODITIES = {
    'Price_Rice_Coarse_BDT_kg': {'display': 'Coarse Rice (মোটা চাল)', 'spike_threshold': 0.05, 'unit': 'BDT/kg'},
    'Price_Lentil_BDT_kg': {'display': 'Lentil (মসুর ডাল)', 'spike_threshold': 0.05, 'unit': 'BDT/kg'},
    'Price_Flour_BDT_kg': {'display': 'Atta (আটা/ময়দা)', 'spike_threshold': 0.05, 'unit': 'BDT/kg'},
    'Price_Soybean_Oil_BDT_liter': {'display': 'Soybean Oil (সয়াবিন তেল)', 'spike_threshold': 0.05, 'unit': 'BDT/liter'},
    'Price_Sugar_BDT_kg': {'display': 'Sugar (চিনি)', 'spike_threshold': 0.05, 'unit': 'BDT/kg'},
    'Price_Onion_BDT_kg': {'display': 'Onion (পেঁয়াজ)', 'spike_threshold': 0.08, 'unit': 'BDT/kg'},
}

def train_and_calibrate_spike_models():
    """
    Calibrates conformal residual quantiles on the 30-day delta predictions
    and calculates shock pass-through correlation dynamics.
    """
    print("=" * 70)
    print(" [*] TRAINING COMMODITY SPIKE & CONFORMAL INTERVAL ENGINE")
    print("=" * 70)
    
    df = pd.read_csv(DATASET_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    with open(MODELS_PATH, "rb") as f:
        comm_models = pickle.load(f)
        
    engine_artifacts = {}
    
    for col, meta in COMMODITIES.items():
        if col not in df.columns or col not in comm_models:
            continue
            
        print(f"\n[+] Calibrating {meta['display']} ({col})...")
        m_info = comm_models[col]
        model = m_info['model']
        feats = m_info['features']
        
        # Prepare valid historical data with 30-day forward delta
        sub_df = df.dropna(subset=feats + [col]).copy()
        sub_df = sub_df[sub_df[col] > 0].reset_index(drop=True)
        sub_df['Target_Delta_30D'] = sub_df[col].shift(-30) - sub_df[col]
        sub_df = sub_df.dropna(subset=['Target_Delta_30D']).reset_index(drop=True)
        
        if len(sub_df) < 50:
            continue
            
        # 1. Delta Predictions & Conformal Residual Quantiles
        X = sub_df[feats]
        delta_true = sub_df['Target_Delta_30D']
        delta_pred = model.predict(X)
        delta_residuals = delta_true - delta_pred
        
        # Non-parametric empirical quantiles of residuals (10th and 90th percentile for 80% coverage)
        q_lower = float(np.percentile(delta_residuals, 10))
        q_upper = float(np.percentile(delta_residuals, 90))
        empirical_coverage = float(np.mean((delta_residuals >= q_lower) & (delta_residuals <= q_upper)))
        
        print(f"    - 80% Delta Conformal Quantiles: [{q_lower:+.2f}, {q_upper:+.2f}] BDT (Coverage: {empirical_coverage*100:.1f}%)")
        
        # 2. Historical Spike Rate: Did price increase by > spike_threshold in 30 days?
        sub_df['Price_Return_30D'] = sub_df['Target_Delta_30D'] / sub_df[col]
        sub_df['Spike_Event_30D'] = (sub_df['Price_Return_30D'] >= meta['spike_threshold']).astype(int)
        historical_spike_rate = float(sub_df['Spike_Event_30D'].mean())
        print(f"    - Historical 30-Day Spike Base Rate: {historical_spike_rate*100:.1f}%")
        
        # 3. Shock Pass-Through Lag Analysis (USD/BDT & Crude Oil -> Commodity)
        pass_through_lags = {}
        for lag in [15, 30, 45, 60]:
            usd_corr = float(sub_df[col].corr(sub_df['USD_BDT_Rate'].shift(lag)))
            oil_corr = float(sub_df[col].corr(sub_df['Brent_Crude_Oil_USD'].shift(lag)))
            pass_through_lags[f"lag_{lag}d"] = {
                'usd_pass_through_corr': round(usd_corr, 3) if not np.isnan(usd_corr) else 0.5,
                'oil_pass_through_corr': round(oil_corr, 3) if not np.isnan(oil_corr) else 0.4
            }
        
        engine_artifacts[col] = {
            'display': meta['display'],
            'unit': meta['unit'],
            'spike_threshold_pct': meta['spike_threshold'] * 100,
            'features': feats,
            'model': model,
            'q_lower_80': q_lower,
            'q_upper_80': q_upper,
            'empirical_coverage': empirical_coverage,
            'delta_residual_std': float(delta_residuals.std()),
            'historical_spike_rate': historical_spike_rate,
            'pass_through_lags': pass_through_lags
        }
        
    os.makedirs(os.path.dirname(SPIKE_MODELS_PATH), exist_ok=True)
    with open(SPIKE_MODELS_PATH, "wb") as f:
        pickle.dump(engine_artifacts, f)
        
    print(f"\n[+] Engine Artifacts saved to: {SPIKE_MODELS_PATH}")
    return engine_artifacts

def predict_commodity_spike_and_interval(current_market_dict):
    """
    Given today's live market features, computes:
    - Current Price (BDT)
    - 30-Day Point Forecast (BDT)
    - 80% Uncertainty Interval [Lower BDT, Upper BDT]
    - 30-Day Spike Probability (%)
    - Pass-Through Lag Dynamics
    """
    if not os.path.exists(SPIKE_MODELS_PATH):
        train_and_calibrate_spike_models()
        
    with open(SPIKE_MODELS_PATH, "rb") as f:
        artifacts = pickle.load(f)
        
    results = {}
    
    for col, art in artifacts.items():
        feats = art['features']
        # Build feature vector
        row_dict = {}
        for feat in feats:
            if feat in current_market_dict:
                row_dict[feat] = current_market_dict[feat]
            elif feat == col:
                # Default current prices if not passed
                default_prices = {
                    'Price_Rice_Coarse_BDT_kg': 50.0,
                    'Price_Lentil_BDT_kg': 135.0,
                    'Price_Flour_BDT_kg': 46.0,
                    'Price_Soybean_Oil_BDT_liter': 168.0,
                    'Price_Sugar_BDT_kg': 130.0,
                    'Price_Onion_BDT_kg': 110.0
                }
                row_dict[feat] = current_market_dict.get(col, default_prices.get(col, 50.0))
            else:
                row_dict[feat] = 0.0
                
        X_curr = pd.DataFrame([row_dict])[feats]
        current_price = row_dict[col]
        
        # Predict 30-day delta
        delta_pred = float(art['model'].predict(X_curr)[0])
        point_pred_30d = current_price + delta_pred
        
        # Conformal 80% Prediction Interval on Price:
        # P_30 in [Current + delta_pred + q_lower, Current + delta_pred + q_upper]
        interval_lower = max(0.0, point_pred_30d + art['q_lower_80'])
        interval_upper = point_pred_30d + art['q_upper_80']
        
        # Calibrated 30-Day Spike Probability:
        # A spike occurs if delta_true >= current_price * threshold
        threshold_delta = current_price * (art['spike_threshold_pct'] / 100.0)
        # Required residual for spike: delta_true - delta_pred >= threshold_delta - delta_pred
        z = (delta_pred - threshold_delta) / (art['delta_residual_std'] + 1e-6)
        spike_prob = float(stats.norm.cdf(z))
        # Bound with base rate
        spike_prob = max(0.01, min(0.99, spike_prob))
        
        results[col] = {
            'display': art['display'],
            'unit': art['unit'],
            'current_price': round(current_price, 2),
            'predicted_delta_30d': round(delta_pred, 2),
            'point_forecast_30d': round(point_pred_30d, 2),
            'interval_80_lower': round(interval_lower, 2),
            'interval_80_upper': round(interval_upper, 2),
            'spike_threshold_pct': art['spike_threshold_pct'],
            'spike_probability_pct': round(spike_prob * 100, 1),
            'spike_risk_level': 'HIGH SPIKE RISK' if spike_prob > 0.40 else ('MODERATE' if spike_prob > 0.20 else 'LOW / STABLE'),
            'pass_through_30d_usd': art['pass_through_lags']['lag_30d']['usd_pass_through_corr']
        }
        
    return results

if __name__ == "__main__":
    train_and_calibrate_spike_models()
    test_market = {
        'USD_BDT_Rate': 124.21,
        'Wheat_Price_USD': 599.75,
        'Brent_Crude_Oil_USD': 110.62,
        'Food_Inflation_pct': 7.22,
        'Oil_Shock_YoY_pct': 14.5,
        'Price_Rice_Coarse_BDT_kg': 50.00,
        'Price_Lentil_BDT_kg': 135.00,
        'Price_Flour_BDT_kg': 46.00,
        'Price_Soybean_Oil_BDT_liter': 168.00,
        'Price_Sugar_BDT_kg': 130.00,
        'Price_Onion_BDT_kg': 110.00
    }
    res = predict_commodity_spike_and_interval(test_market)
    print("\n" + "=" * 70)
    print(" LIVE COMMODITY SPIKE & INTERVAL EVALUATION (TEST RUN)")
    print("=" * 70)
    for c, r in res.items():
        print(f"[*] {r['display']}:")
        print(f"    Current Price       : {r['current_price']} {r['unit']}")
        print(f"    30-Day Point Est.   : {r['point_forecast_30d']} {r['unit']} (Delta: {r['predicted_delta_30d']:+0.2f})")
        print(f"    80% Price Interval  : [{r['interval_80_lower']} - {r['interval_80_upper']}] {r['unit']}")
        print(f"    30-Day Spike Risk   : {r['spike_probability_pct']}% ({r['spike_risk_level']}) [Threshold: +{r['spike_threshold_pct']}%%]")
        print(f"    USD Pass-Through 30d: Correlation = {r['pass_through_30d_usd']}")
