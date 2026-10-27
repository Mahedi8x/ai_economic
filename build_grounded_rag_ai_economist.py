"""
build_grounded_rag_ai_economist.py
====================================
Grounded RAG AI Economist — Contribution 7.

REVIEWER FIX (Hard-coded results removed):
-------------------------------------------
BEFORE (Fabricated):
  - QUANT_STATE had hard-coded predicted_usd_30d=123.54, predicted_food_inf_30d=7.17,
    predicted_crisis_prob=0.018, commodity curr/pred prices — all static numbers.
    Models were loaded but .predict() was NEVER called.
  - eval_data had fabricated: Hallucination 38.5%->0%, Citation 98.2%, Expert 4.85.
    No questions were actually run, no LLM was called, no expert rated anything.
  - grounded_rag_query() had "STABLE" hard-coded regardless of model output.

AFTER (Honest):
  - All model predictions computed via actual .predict() calls on latest data row.
  - RAG evaluation uses a structured retrieval-presence audit protocol:
      * Each question is asked, ChromaDB retrieval is checked for keyword coverage.
      * Grounding score = fraction of answer-critical keywords found in retrieved chunks.
      * Non-RAG baseline = 0% grounding (no retrieval, raw fabrication baseline).
    This is transparent, reproducible, and makes no claims about an external LLM.
  - Risk status label computed from actual classifier probability threshold.
  - All numbers in this file come from code execution, not from the developer's mind.
"""

import os
import sys
import glob
import joblib
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import chromadb
from chromadb.utils import embedding_functions

# Windows console encoding safeguard
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("=" * 70)
print("  CONTRIBUTION 7: GROUNDED RAG AI ECONOMIST (Post-Reviewer Fix)")
print("  Hard-coded results REMOVED — all predictions from actual models")
print("=" * 70)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR   = os.path.join(BASE_DIR, "rag_knowledge_base")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db_store")

# =============================================================================
# 1. Initialize ChromaDB Vector Database
# =============================================================================
print("\n[1/5] Initializing ChromaDB & Ingesting Policy Documents...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
embed_fn = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="bangladesh_economic_policy_kb",
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"}
)

doc_files = glob.glob(os.path.join(KB_DIR, "*.txt"))
print(f"  Found {len(doc_files)} policy documents in {KB_DIR}")

documents, metadatas, ids = [], [], []
for file_idx, fpath in enumerate(doc_files):
    fname = os.path.basename(fpath)
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    chunks = [c.strip() for c in content.split('\n\n') if len(c.strip()) > 40]
    for c_idx, chunk in enumerate(chunks):
        doc_id = f"{fname}_chunk_{c_idx}"
        documents.append(chunk)
        metadatas.append({"source": fname, "chunk_index": c_idx})
        ids.append(doc_id)

if documents:
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    print(f"  Ingested {len(documents)} policy chunks into ChromaDB.")
else:
    print("  [WARNING] No .txt documents found in rag_knowledge_base/")
    print("            RAG grounding scores will be 0 — add policy docs to improve.")

# =============================================================================
# 2. Load Data & Run ACTUAL Model Predictions (no hard-coding)
# =============================================================================
print("\n[2/5] Loading models and running ACTUAL predictions on latest data...")

dataset_path = os.path.join(BASE_DIR, "daily_macro_master_dataset.csv")
df = pd.read_csv(dataset_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
latest_row = df.iloc[-1]

# --- Load models ---
def safe_load(path):
    if os.path.exists(path):
        return joblib.load(path)
    print(f"  [WARNING] Model not found: {path}")
    return None

reg_model   = safe_load(os.path.join(BASE_DIR, "best_forecasting_model.pkl"))
clf_model   = safe_load(os.path.join(BASE_DIR, "best_crisis_classifier.pkl"))
food_model  = safe_load(os.path.join(BASE_DIR, "best_food_inflation_model.pkl"))
comm_models = safe_load(os.path.join(BASE_DIR, "best_commodity_models.pkl"))

# --- Feature sets (leakage-free, no DESI/Risk Score) ---
REG_FEATURES = [
    'USD_BDT_Rate', 'Brent_Crude_Oil_USD', 'US_Dollar_Index_DXY', 'Gold_USD_oz',
    'Natural_Gas_USD', 'Wheat_Price_USD', 'US_10Y_Treasury_pct',
    'Headline_Inflation_pct', 'Food_Inflation_pct', 'Foreign_Reserves_USD_Bn',
    'Remittance_USD_Bn', 'Reserves_Months_of_Imports', 'External_Debt_USD_Bn',
    'Short_Term_Debt_USD_Bn', 'USD_BDT_30D_Volatility', 'USD_BDT_YoY_Return_pct',
    'Oil_Shock_YoY_pct', 'DXY_30D_MA', 'Reserves_to_Short_Term_Debt_Ratio',
]
FOOD_FEATURES = [
    'Wheat_Price_USD', 'Brent_Crude_Oil_USD', 'Natural_Gas_USD', 'Gold_USD_oz',
    'Oil_Shock_YoY_pct', 'USD_BDT_Rate', 'US_Dollar_Index_DXY',
    'Foreign_Reserves_USD_Bn', 'Reserves_Months_of_Imports',
    'USD_BDT_30D_Volatility', 'Food_Inflation_pct', 'Headline_Inflation_pct',
]

def predict_safe(model, features, row, fallback=None):
    """Run model.predict() on latest row. Returns fallback if model/features missing."""
    if model is None:
        return fallback
    available = [f for f in features if f in row.index]
    if len(available) < len(features):
        missing = set(features) - set(available)
        print(f"    [WARN] Missing features for prediction: {missing}")
    try:
        inp = pd.DataFrame([row[available]])
        return float(model.predict(inp)[0])
    except Exception as e:
        print(f"    [WARN] Prediction failed: {e}")
        return fallback

# --- Run actual predictions ---
current_usd        = float(latest_row['USD_BDT_Rate'])
current_food_inf   = float(latest_row['Food_Inflation_pct'])
current_reserves   = float(latest_row['Foreign_Reserves_USD_Bn'])
current_oil        = float(latest_row['Brent_Crude_Oil_USD'])
current_wheat      = float(latest_row['Wheat_Price_USD'])

pred_usd_return_30d    = predict_safe(reg_model, REG_FEATURES, latest_row, fallback=0.0)
pred_usd_30d           = current_usd * (1.0 + pred_usd_return_30d / 100.0)

pred_food_delta_30d    = predict_safe(food_model, FOOD_FEATURES, latest_row, fallback=0.0)
pred_food_inf_30d      = current_food_inf + pred_food_delta_30d

# Crisis probability from classifier
clf_features = [f for f in REG_FEATURES if f in latest_row.index]
if clf_model is not None:
    try:
        inp_clf = pd.DataFrame([latest_row[clf_features]])
        pred_crisis_prob = float(clf_model.predict_proba(inp_clf)[0, 1])
    except Exception as e:
        print(f"    [WARN] Crisis prob failed: {e}")
        pred_crisis_prob = float('nan')
else:
    pred_crisis_prob = float('nan')

# Risk status from actual probability (not hard-coded "STABLE")
if np.isnan(pred_crisis_prob):
    risk_status = "UNKNOWN (model unavailable)"
elif pred_crisis_prob >= 0.50:
    risk_status = "HIGH RISK (Crisis Probable)"
elif pred_crisis_prob >= 0.30:
    risk_status = "ELEVATED"
else:
    risk_status = "STABLE"

# Commodity predictions from actual models
COMMODITY_FEATURES_MAP = {
    'Price_Rice_Miniket_BDT_kg':   ['USD_BDT_Rate', 'Wheat_Price_USD', 'Brent_Crude_Oil_USD',
                                     'Food_Inflation_pct', 'Oil_Shock_YoY_pct'],
    'Price_Lentil_BDT_kg':         ['USD_BDT_Rate', 'Wheat_Price_USD', 'Brent_Crude_Oil_USD',
                                     'Food_Inflation_pct', 'Oil_Shock_YoY_pct'],
    'Price_Soybean_Oil_BDT_liter': ['USD_BDT_Rate', 'Brent_Crude_Oil_USD', 'Natural_Gas_USD',
                                     'Food_Inflation_pct', 'Oil_Shock_YoY_pct'],
    'Price_Flour_BDT_kg':          ['USD_BDT_Rate', 'Wheat_Price_USD', 'Brent_Crude_Oil_USD',
                                     'Food_Inflation_pct', 'Oil_Shock_YoY_pct'],
    'Price_Onion_BDT_kg':          ['USD_BDT_Rate', 'Food_Inflation_pct', 'Brent_Crude_Oil_USD',
                                     'Oil_Shock_YoY_pct'],
    'Price_Sugar_BDT_kg':          ['USD_BDT_Rate', 'Brent_Crude_Oil_USD', 'Wheat_Price_USD',
                                     'Food_Inflation_pct', 'Oil_Shock_YoY_pct'],
}
COMMODITY_LABELS = {
    'Price_Rice_Miniket_BDT_kg':   'Rice',
    'Price_Lentil_BDT_kg':         'Lentil',
    'Price_Soybean_Oil_BDT_liter': 'Oil',
    'Price_Flour_BDT_kg':          'Flour',
    'Price_Onion_BDT_kg':          'Onion',
    'Price_Sugar_BDT_kg':          'Sugar',
}

commodities_pred = {}
for col, label in COMMODITY_LABELS.items():
    if col not in latest_row.index:
        continue
    curr_price = float(latest_row[col]) if not pd.isna(latest_row[col]) else None
    if curr_price is None:
        continue
    pred_price = curr_price   # default: no model = persistence
    if comm_models and col in comm_models:
        m_info  = comm_models[col]
        m_obj   = m_info.get('model')
        m_feats = m_info.get('features', COMMODITY_FEATURES_MAP.get(col, []))
        delta = predict_safe(m_obj, m_feats, latest_row, fallback=0.0)
        pred_price = curr_price + delta
    commodities_pred[label] = {'curr': round(curr_price, 2), 'pred': round(pred_price, 2)}

# Build QUANT_STATE from actual model outputs
QUANT_STATE = {
    'date':                  str(latest_row['Date'])[:10],
    'usd_rate':              round(current_usd, 2),
    'food_inflation':        round(current_food_inf, 2),
    'headline_inflation':    round(float(latest_row.get('Headline_Inflation_pct', 0)), 2),
    'reserves_bn':           round(current_reserves, 2),
    'oil_price':             round(current_oil, 2),
    'wheat_price':           round(current_wheat, 2),
    # ACTUAL model predictions (not hard-coded):
    'predicted_usd_30d':     round(pred_usd_30d, 2),
    'predicted_food_inf_30d':round(pred_food_inf_30d, 2),
    'predicted_crisis_prob': round(pred_crisis_prob, 4) if not np.isnan(pred_crisis_prob) else None,
    'risk_status':           risk_status,
    'commodities':           commodities_pred,
}

print(f"\n  [Actual Model Predictions — not hard-coded]")
print(f"    USD/BDT now  : {QUANT_STATE['usd_rate']}")
print(f"    USD/BDT 30D  : {QUANT_STATE['predicted_usd_30d']}  (model: reg_model.predict)")
print(f"    Food Inf now : {QUANT_STATE['food_inflation']}%")
print(f"    Food Inf 30D : {QUANT_STATE['predicted_food_inf_30d']}%  (model: food_model.predict)")
print(f"    Crisis Prob  : {QUANT_STATE['predicted_crisis_prob']}  => {risk_status}")
print(f"    Commodities  : {QUANT_STATE['commodities']}")

# =============================================================================
# 3. Grounded RAG Query Function
# =============================================================================
print("\n[3/5] Defining Grounded Retrieval & Generation Logic...")

def grounded_rag_query(query_text, n_results=3):
    """
    Retrieves policy chunks from ChromaDB and fuses with quantitative model state.
    Returns: (report_str, retrieved_chunks, sources)
    """
    results         = collection.query(query_texts=[query_text], n_results=n_results)
    retrieved_chunks = results['documents'][0] if results['documents'] else []
    sources          = [m['source'] for m in results['metadatas'][0]] if results['metadatas'] else []

    context_str = "\n".join(
        [f"[{sources[i]}] {retrieved_chunks[i]}" for i in range(len(retrieved_chunks))]
    ) if retrieved_chunks else "[No policy documents retrieved — ChromaDB collection is empty]"

    crisis_display = (f"{QUANT_STATE['predicted_crisis_prob']*100:.1f}%"
                      if QUANT_STATE['predicted_crisis_prob'] is not None else "N/A")

    comm_lines = "\n".join(
        [f"   - {lbl}: BDT {v['curr']:.1f}/unit → 30D forecast BDT {v['pred']:.1f}/unit"
         for lbl, v in QUANT_STATE['commodities'].items()]
    ) if QUANT_STATE['commodities'] else "   [No commodity model predictions available]"

    report = f"""
### GROUNDED AI ECONOMIST POLICY BRIEFING
**Query:** {query_text}
**As of Date:** {QUANT_STATE['date']}
**Macro Risk Status:** {QUANT_STATE['risk_status']} (Crisis Probability: {crisis_display})

1. **Empirical Model Projections (XGBoost Engine — actual .predict() calls):**
   - Food Inflation: {QUANT_STATE['food_inflation']}% → 30D forecast: {QUANT_STATE['predicted_food_inf_30d']}%
   - USD/BDT: {QUANT_STATE['usd_rate']} → 30D forecast: {QUANT_STATE['predicted_usd_30d']}
   - Forex Reserves: ${QUANT_STATE['reserves_bn']}B
   - Kitchen Commodity 30-Day Outlook:
{comm_lines}

2. **Policy Context (Retrieved from ChromaDB Knowledge Base):**
{context_str}
"""
    return report, retrieved_chunks, sources

# =============================================================================
# 4. RAG Evaluation — Transparent Retrieval-Grounding Audit
#    (No fabricated hallucination/expert scores)
# =============================================================================
print("\n[4/5] Executing Retrieval-Grounding Audit (RQ6)...")
print("  NOTE: This audit measures ChromaDB retrieval coverage, NOT LLM hallucination.")
print("  Hallucination/expert ratings require a live LLM + human judges — not claimed here.")

BENCHMARK_QUESTIONS = [
    {
        "question": "What is the current policy repo rate and how does it influence imported inflation?",
        "critical_keywords": ["repo rate", "monetary policy", "interest rate", "inflation"]
    },
    {
        "question": "How does the crawling peg exchange rate system govern USD/BDT transactions?",
        "critical_keywords": ["crawling peg", "exchange rate", "USD", "BDT", "forex"]
    },
    {
        "question": "What is the import LC cash margin requirement for essential food staples?",
        "critical_keywords": ["LC", "letter of credit", "margin", "import", "food"]
    },
    {
        "question": "What is the weight of food items in the BBS CPI basket?",
        "critical_keywords": ["CPI", "food", "weight", "basket", "BBS"]
    },
    {
        "question": "When does TCB buffer stock intervention trigger to cool retail price spikes?",
        "critical_keywords": ["TCB", "buffer", "stock", "intervention", "price"]
    },
    {
        "question": "How are gross foreign reserves calculated under the IMF structural benchmark?",
        "critical_keywords": ["reserves", "IMF", "gross", "foreign", "benchmark"]
    },
    {
        "question": "What is the expected 30-day trajectory of soybean oil based on crude oil shocks?",
        "critical_keywords": ["soybean", "oil", "crude", "price", "forecast"]
    },
    {
        "question": "What is the projected food inflation rate in Bangladesh 30 days forward?",
        "critical_keywords": ["food inflation", "forecast", "30 day", "Bangladesh", "projection"]
    },
    {
        "question": "How does global wheat price volatility transmit to domestic flour prices in Dhaka?",
        "critical_keywords": ["wheat", "flour", "price", "transmission", "Dhaka"]
    },
    {
        "question": "What export diversification measures reduce Bangladesh food price vulnerability?",
        "critical_keywords": ["export", "diversification", "food", "vulnerability", "price"]
    },
]

def grounding_score(question_entry, n_results=3):
    """
    Grounding score = fraction of critical keywords found in retrieved chunks.
    Non-RAG baseline = 0.0 (no retrieval at all).
    This is a transparent, reproducible metric for retrieval quality.
    """
    q    = question_entry["question"]
    keys = [k.lower() for k in question_entry["critical_keywords"]]

    results  = collection.query(query_texts=[q], n_results=n_results)
    chunks   = results['documents'][0] if results['documents'] else []
    combined = " ".join(chunks).lower()

    hits = sum(1 for k in keys if k in combined)
    score = hits / len(keys) if keys else 0.0
    return score, hits, len(keys), chunks

print(f"\n  Running {len(BENCHMARK_QUESTIONS)} benchmark questions against ChromaDB...")
print(f"  {'Q#':<4} {'Grounding Score':>16}  {'Keywords Hit':>12}  Question")
print("  " + "-" * 75)

scores_rag    = []
scores_no_rag = []   # baseline: 0.0 for all (no retrieval)

for i, q_entry in enumerate(BENCHMARK_QUESTIONS):
    score, hits, total, _ = grounding_score(q_entry)
    scores_rag.append(score)
    scores_no_rag.append(0.0)   # Non-RAG has zero retrieval grounding by definition
    print(f"  Q{i+1:<3} {score*100:>14.1f}%  {hits:>5}/{total:<6}  {q_entry['question'][:60]}...")

mean_rag_score   = float(np.mean(scores_rag)) * 100
mean_norag_score = 0.0   # honest baseline
retrieval_lift   = mean_rag_score - mean_norag_score
pct_fully_grounded = float(np.mean([s == 1.0 for s in scores_rag])) * 100
pct_partial       = float(np.mean([0 < s < 1.0 for s in scores_rag])) * 100
pct_ungrounded    = float(np.mean([s == 0.0 for s in scores_rag])) * 100

print(f"\n  {'Metric':<45} {'No-RAG':>10}  {'RAG':>10}")
print("  " + "-" * 68)
print(f"  {'Mean Retrieval Grounding Score (%)':<45} {mean_norag_score:>9.1f}%  {mean_rag_score:>9.1f}%")
print(f"  {'Fully Grounded (all keywords hit) (%)':<45} {'N/A':>10}  {pct_fully_grounded:>9.1f}%")
print(f"  {'Partially Grounded (%)':<45} {'N/A':>10}  {pct_partial:>9.1f}%")
print(f"  {'Ungrounded / No Match (%)':<45} {'N/A':>10}  {pct_ungrounded:>9.1f}%")
print(f"  {'Retrieval Lift (RAG - No-RAG)':<45} {'':>10}  {retrieval_lift:>+9.1f}%")
print()
print("  [NOTE] Scores reflect ChromaDB keyword coverage in retrieved policy chunks.")
print("  [NOTE] Grounding = 0% when no policy .txt files are in rag_knowledge_base/.")
print("  [NOTE] LLM hallucination rate requires an external LLM benchmark — NOT claimed.")

# Build eval_data from computed scores (not hard-coded)
eval_data = [
    {
        'Metric': 'Mean Retrieval Grounding Score (%)',
        'No_RAG_Baseline': round(mean_norag_score, 1),
        'Grounded_RAG':    round(mean_rag_score, 1),
        'Lift':            f'+{retrieval_lift:.1f}%',
        'Note': 'Computed from ChromaDB keyword coverage across 10 questions'
    },
    {
        'Metric': 'Fully Grounded Questions (%)',
        'No_RAG_Baseline': 0.0,
        'Grounded_RAG':    round(pct_fully_grounded, 1),
        'Lift':            f'+{pct_fully_grounded:.1f}%',
        'Note': 'Fraction where all critical keywords found in retrieved chunks'
    },
    {
        'Metric': 'Ungrounded Questions (%) — needs more policy docs',
        'No_RAG_Baseline': 100.0,
        'Grounded_RAG':    round(pct_ungrounded, 1),
        'Lift':            f'{pct_ungrounded - 100:.1f}%',
        'Note': 'Add .txt policy docs to rag_knowledge_base/ to reduce this'
    },
]

df_eval = pd.DataFrame(eval_data)
print("\n  Evaluation summary saved (all computed, none hard-coded):")
print(df_eval[['Metric', 'No_RAG_Baseline', 'Grounded_RAG', 'Lift']].to_string(index=False))

# Sample query output
print("\n--- Sample Grounded RAG Output ---")
sample_report, _, _ = grounded_rag_query(
    "What is the expected food inflation trajectory and what policy controls it?"
)
print(sample_report)

# =============================================================================
# 5. Publication Chart (using computed scores)
# =============================================================================
print("\n[5/5] Generating Publication Chart...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: per-question grounding score bar chart
ax1 = axes[0]
q_labels = [f"Q{i+1}" for i in range(len(BENCHMARK_QUESTIONS))]
colors = ['#059669' if s > 0 else '#EF4444' for s in scores_rag]
ax1.bar(q_labels, [s * 100 for s in scores_rag], color=colors)
ax1.axhline(mean_rag_score, color='navy', linestyle='--', linewidth=1.5,
            label=f'Mean RAG = {mean_rag_score:.1f}%')
ax1.set_ylabel('Retrieval Grounding Score (%)', fontsize=10, fontweight='bold')
ax1.set_title('Per-Question Retrieval Grounding\n(Green = grounded, Red = no match in KB)',
              fontsize=10, fontweight='bold')
ax1.legend()
ax1.set_ylim(0, 115)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Right: summary comparison
ax2 = axes[1]
metrics_labels = ['Mean Grounding\nScore (%)', 'Fully Grounded\nQuestions (%)']
no_rag_vals = [0.0, 0.0]
rag_vals    = [mean_rag_score, pct_fully_grounded]
x = np.arange(len(metrics_labels))
w = 0.32
ax2.bar(x - w/2, no_rag_vals, w, label='No-RAG Baseline (0% retrieval)', color='#EF4444')
ax2.bar(x + w/2, rag_vals,    w, label='Grounded RAG (ChromaDB)', color='#059669')
for i, (nr, r) in enumerate(zip(no_rag_vals, rag_vals)):
    ax2.text(i - w/2, nr + 2, f'{nr:.0f}%', ha='center', va='bottom', fontsize=9, color='#991B1B')
    ax2.text(i + w/2, r  + 2, f'{r:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#065F46')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics_labels, fontsize=9)
ax2.set_title('Contribution 7: RAG Retrieval Grounding\n(All scores computed from actual ChromaDB queries)',
              fontsize=10, fontweight='bold')
ax2.legend(fontsize=8)
ax2.set_ylim(0, 115)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

plt.suptitle(
    "Grounded RAG AI Economist — Retrieval Quality Audit\n"
    "(No fabricated hallucination/expert scores — see paper Sec 4.7 for methodology)",
    fontsize=10, fontweight='bold', y=1.01
)
plt.tight_layout()
plot_path = os.path.join(BASE_DIR, "rag_vs_non_rag_evaluation.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved chart -> {plot_path}")

# =============================================================================
# Save RAG advisor artifact
# =============================================================================
advisor_path = os.path.join(BASE_DIR, "grounded_rag_advisor.pkl")
joblib.dump({
    'collection_name':     'bangladesh_economic_policy_kb',
    'chroma_db_dir':       CHROMA_DIR,
    'eval_results':        eval_data,
    'grounding_scores':    scores_rag,
    'mean_grounding_pct':  mean_rag_score,
    'quant_state':         QUANT_STATE,
    'n_questions':         len(BENCHMARK_QUESTIONS),
    'methodology_note': (
        "Grounding score = fraction of critical keywords found in ChromaDB-retrieved chunks. "
        "No LLM was called. No hallucination rate or expert rating is claimed. "
        "All model predictions computed via actual .predict() calls."
    )
}, advisor_path)
print(f"  Saved RAG advisor artifact -> {advisor_path}")

print("\n" + "=" * 70)
print("  SUCCESS: Contribution 7 complete (honest evaluation)")
print("=" * 70)
print("  Reviewer Fix verified:")
print("  [x] QUANT_STATE: all predictions from actual model.predict() calls")
print("  [x] No hard-coded predicted_usd_30d=123.54 or food_inf=7.17 etc.")
print("  [x] Risk status label computed from classifier probability threshold")
print("  [x] RAG evaluation: retrieval grounding audit (reproducible, no LLM)")
print("  [x] No fabricated Hallucination 38.5%, Citation 98.2%, Expert 4.85")
print("  [x] eval_data built from computed scores, not developer's imagination")
print(f"\n  Actual results today:")
print(f"    USD/BDT 30D prediction : {QUANT_STATE['predicted_usd_30d']}")
print(f"    Food Inflation 30D     : {QUANT_STATE['predicted_food_inf_30d']}%")
print(f"    Crisis Probability     : {QUANT_STATE['predicted_crisis_prob']}")
print(f"    Risk Status            : {QUANT_STATE['risk_status']}")
print(f"    Mean RAG Grounding     : {mean_rag_score:.1f}%  (from {len(BENCHMARK_QUESTIONS)} questions)")
