"""
Clean and polish PAPER_MANUSCRIPT.md for academic journal submission.
"""
import re

with open(r"C:\ai_economic\PAPER_MANUSCRIPT.md", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# Fix common UTF-8 encoding corruptions
text = text.replace('\ufffd', '—')
text = re.sub(r'\?["\']?', '—', text)
text = text.replace('', '—')
text = text.replace('A ·', '·')
text = text.replace('Â·', '·')
text = text.replace('A ', ' ')
text = text.replace('BussiAre', 'Bussière')
text = text.replace('BussiA"re', 'Bussière')

# Ensure author metadata has a clear, formal layout
author_block = """# Explainable Multi-Indicator Machine Learning and Grounded LLM Framework for Food Inflation Forecasting and Economic Early Warning in Bangladesh

> **Author:** Mahedi Hasan  
> **Affiliation:** Department of Computer Science & Engineering / Data Science  
> **Repository:** https://github.com/Mahedi8x/ai_economic  
> **Keywords:** Food Inflation Nowcasting · XGBoost · SHAP Explainability · Conformal Prediction Intervals · Price Spike Early Warning · Isolation Forest · Retrieval-Augmented Generation (RAG) · Bangladesh Macroeconomics · Daily Economic Stress Index (DESI)

---
"""

lines = text.split('\n')
# Find where Abstract starts
abs_idx = 0
for i, l in enumerate(lines):
    if l.strip().startswith('## Abstract'):
        abs_idx = i
        break

if abs_idx > 0:
    polished_text = author_block + '\n'.join(lines[abs_idx:])
else:
    polished_text = text

with open(r"C:\ai_economic\PAPER_MANUSCRIPT.md", "w", encoding="utf-8") as f:
    f.write(polished_text)

print("PAPER_MANUSCRIPT.md polished successfully!")
