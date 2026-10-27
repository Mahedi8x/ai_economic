import urllib.request, json, os, sys

sys.stdout.reconfigure(encoding='utf-8') if sys.platform == 'win32' else None

BASE = r'C:\ai_economic'

# 1. Check GitHub repo
req = urllib.request.Request(
    'https://api.github.com/repos/Mahedi8x/ai_economic/commits?per_page=10',
    headers={'User-Agent': 'py'}
)
commits = json.loads(urllib.request.urlopen(req).read())
print("\n=== GITHUB COMMITS (Latest 10) ===")
for c in commits:
    print(f"  {c['sha'][:7]} - {c['commit']['message'][:70]}")

# 2. Check GitHub files
req2 = urllib.request.Request(
    'https://api.github.com/repos/Mahedi8x/ai_economic/git/trees/main?recursive=1',
    headers={'User-Agent': 'py'}
)
tree = json.loads(urllib.request.urlopen(req2).read())
files = [f for f in tree['tree'] if f['type'] == 'blob']
print(f"\n=== GITHUB FILES (Total: {len(files)}) ===")
for f in sorted(files, key=lambda x: x['path']):
    sz = f.get('size', 0)
    print(f"  {sz:>10} bytes  {f['path']}")

# 3. Check Local Dataset integrity
import pandas as pd, numpy as np
df = pd.read_csv(os.path.join(BASE, 'daily_macro_master_dataset.csv'))
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"\n=== DATASET INTEGRITY CHECK ===")
print(f"  Total rows       : {len(df):,}")
print(f"  Total columns    : {len(df.columns)}")
print(f"  Date range       : {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
print(f"  Missing values   : {df.isnull().sum().sum()} total NaN cells")
print(f"\n  Column-wise NaN counts:")
nans = df.isnull().sum()
nans = nans[nans > 0]
if len(nans) == 0:
    print("  ✓ No missing values found in any column!")
else:
    for col, cnt in nans.items():
        pct = cnt/len(df)*100
        print(f"    {col}: {cnt} NaN ({pct:.1f}%)")

print(f"\n  Key Columns Present:")
expected_cols = [
    'Date', 'USD_BDT_Rate', 'Food_Inflation_pct', 'Headline_Inflation_pct',
    'Brent_Crude_Oil_USD', 'Wheat_Price_USD', 'Natural_Gas_USD', 'Gold_USD_oz',
    'US_Dollar_Index_DXY', 'Foreign_Reserves_USD_Bn', 'Remittance_USD_Bn',
    'Reserves_Months_of_Imports', 'Daily_Economic_Stress_Index',
    'USD_BDT_30D_Volatility', 'Rice_Price_BDT', 'Soybean_Oil_Price_BDT',
    'Lentil_Price_BDT', 'Flour_Price_BDT'
]
for col in expected_cols:
    status = "✓" if col in df.columns else "✗ MISSING"
    print(f"    {status}  {col}")

# 4. Check ALL local model artifacts
print(f"\n=== MODEL ARTIFACTS CHECK ===")
artifacts = [
    'best_forecasting_model.pkl',
    'best_crisis_classifier.pkl',
    'best_food_inflation_model.pkl',
    'best_commodity_models.pkl',
    'best_multi_horizon_models.pkl',
    'best_isolation_forest.pkl',
    'grounded_rag_advisor.pkl',
    'feature_scaler.pkl',
    'target_scaler.pkl',
]
for a in artifacts:
    path = os.path.join(BASE, a)
    if os.path.exists(path):
        sz = os.path.getsize(path) / 1024
        print(f"  ✓ {a:<40} ({sz:.1f} KB)")
    else:
        print(f"  ✗ MISSING: {a}")

# 5. Check Notebooks
print(f"\n=== JUPYTER NOTEBOOKS CHECK ===")
notebooks = [f for f in os.listdir(BASE) if f.endswith('.ipynb')]
for nb in sorted(notebooks):
    sz = os.path.getsize(os.path.join(BASE, nb)) / 1024
    print(f"  ✓ {nb:<55} ({sz:.1f} KB)")

# 6. Check Figures
print(f"\n=== PUBLICATION FIGURES CHECK ===")
figs = [f for f in os.listdir(BASE) if f.endswith('.png')]
for fig in sorted(figs):
    sz = os.path.getsize(os.path.join(BASE, fig)) / 1024
    print(f"  ✓ {fig:<55} ({sz:.1f} KB)")

# 7. Check key scripts
print(f"\n=== KEY PYTHON SCRIPTS CHECK ===")
scripts = [
    'app.py', 'daily_update.py', 'build_daily_master_dataset.py',
    'run_multi_horizon_and_statistical_tests.py',
    'run_anomaly_detection_and_backtesting.py',
    'build_grounded_rag_ai_economist.py',
    'run_all_hypotheses_validation.py',
    'verify_rq1_rq2_independently.py',
    'generate_research_summary_pdf.py'
]
for s in scripts:
    path = os.path.join(BASE, s)
    if os.path.exists(path):
        lines = len(open(path, encoding='utf-8').readlines())
        print(f"  ✓ {s:<50} ({lines} lines)")
    else:
        print(f"  ✗ MISSING: {s}")

# 8. Dataset Statistics Summary
print(f"\n=== DATASET STATISTICS SUMMARY ===")
key_stats = [
    ('USD_BDT_Rate', 'USD/BDT Exchange Rate (৳)'),
    ('Food_Inflation_pct', 'Food Inflation (%)'),
    ('Foreign_Reserves_USD_Bn', 'Forex Reserves ($ Bn)'),
    ('Daily_Economic_Stress_Index', 'Daily Stress Index (0-100)'),
]
for col, label in key_stats:
    if col in df.columns:
        s = df[col].dropna()
        print(f"  {label}")
        print(f"    Range: {s.min():.2f} to {s.max():.2f} | Mean: {s.mean():.2f} | Std: {s.std():.2f}")

print("\n=== FULL AUDIT COMPLETE ===")
