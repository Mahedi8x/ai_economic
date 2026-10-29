# Data Sources & Provenance

All datasets used in this research are documented below with source institutions, access methods, and temporal coverage. No data was synthetically generated for the primary analysis.

---

## 1. Exchange Rate & Financial Markets

| Variable | Source | Provider | Frequency | Coverage | Access |
|---|---|---|---|---|---|
| USD/BDT Exchange Rate | Yahoo Finance (`BDT=X`) | Yahoo Finance API | Daily | 2010–2026 | `yfinance` Python package |
| Brent Crude Oil (USD/bbl) | Yahoo Finance (`BZ=F`) | Yahoo Finance API | Daily | 2010–2026 | `yfinance` Python package |
| US Dollar Index (DXY) | Yahoo Finance (`DX-Y.NYB`) | Yahoo Finance API | Daily | 2010–2026 | `yfinance` Python package |
| Gold Price (USD/oz) | Yahoo Finance (`GC=F`) | Yahoo Finance API | Daily | 2010–2026 | `yfinance` Python package |
| Natural Gas (USD/MMBtu) | Yahoo Finance (`NG=F`) | Yahoo Finance API | Daily | 2010–2026 | `yfinance` Python package |
| Wheat Price (USD/bushel) | Yahoo Finance (`ZW=F`) | Yahoo Finance API | Daily | 2010–2026 | `yfinance` Python package |

**Access Date:** Continuously updated via GitHub Actions cron (daily at 8pm BST / 14:00 UTC)

---

## 2. Bangladesh Macroeconomic Indicators

| Variable | Source | Institution | Frequency | Coverage | Notes |
|---|---|---|---|---|---|
| Headline Inflation (%) | Bangladesh Bank / BBS | Bangladesh Bureau of Statistics | Monthly | 2010–2026 | CPI-based; published ~6 weeks lag |
| Food Inflation (%) | Bangladesh Bank / BBS | Bangladesh Bureau of Statistics | Monthly | 2010–2026 | Sub-index of headline CPI |
| Foreign Exchange Reserves (USD Bn) | Bangladesh Bank | Bangladesh Bank | Monthly | 2010–2026 | Gross international reserves |
| Remittance Inflows (USD Bn) | Bangladesh Bank | Bangladesh Bank | Monthly | 2010–2026 | Worker remittances |

**Data URL:** https://www.bb.org.bd/en/index.php/econdata/inflation
**Access Date:** October 2026

> [!WARNING]
> Monthly macroeconomic variables (inflation, reserves, remittance) are published with a ~4–6 week lag. The pipeline enforces this lag using publication-calendar-aware date shifting to prevent look-ahead bias.

---

## 3. Domestic Retail Food Prices

| Variable | Source | Institution | Frequency | Coverage | Notes |
|---|---|---|---|---|---|
| Rice (Coarse, BDT/kg) | UN WFP HDX | UN World Food Programme | ~Monthly (surveyed ~15th) | 2010–2026 | Market survey data |
| Lentil (BDT/kg) | UN WFP HDX | UN World Food Programme | ~Monthly | 2010–2026 | Market survey data |
| Flour/Atta (BDT/kg) | UN WFP HDX | UN World Food Programme | ~Monthly | 2010–2026 | Market survey data |
| Soybean Oil (BDT/L) | UN WFP HDX | UN World Food Programme | ~Monthly (from Feb 2024) | 2024–2026 | Added post-2024 survey expansion |
| Sugar (BDT/kg) | UN WFP HDX | UN World Food Programme | ~Monthly (from Feb 2024) | 2024–2026 | Added post-2024 survey expansion |
| Onion (BDT/kg) | UN WFP HDX | UN World Food Programme | ~Monthly (from Feb 2024) | 2024–2026 | Added post-2024 survey expansion |

**Data URL:** https://data.humdata.org/dataset/wfp-food-prices-for-bangladesh
**Dataset Name:** WFP Food Prices — Bangladesh
**Access Date:** October 2026

---

## 4. World Development Indicators (World Bank)

| Variable | Source | Institution | Frequency | Coverage | Notes |
|---|---|---|---|---|---|
| GDP per capita (current USD) | WDI indicator `NY.GDP.PCAP.CD` | World Bank | Annual | 2010–2024 | Forward-filled with 1-year publication lag |
| Trade openness (% of GDP) | WDI indicator `NE.TRD.GNFS.ZS` | World Bank | Annual | 2010–2024 | Forward-filled with 1-year publication lag |

**Data URL:** https://databank.worldbank.org/source/world-development-indicators
**Access Method:** `wbdata` Python package
**Access Date:** October 2026

> [!IMPORTANT]
> WDI annual data is forward-filled (`ffill`) with a Y+1 publication lag (e.g., 2023 GDP available from October 2024). This means annual WDI values are applied uniformly across the calendar year following their release — not retroactively to the reference year — to prevent look-ahead leakage.

---

## 5. Anomaly Detection Backtesting Reference Episodes

Crisis episodes used for Isolation Forest OOS evaluation are documented in `anomaly_detection_backtesting_results.csv`:

| Episode | Type | Crisis Start | First Alert | Lead Time |
|---|---|---|---|---|
| 2011-12 Capital Market Crash | In-Sample (Historical Control) | 2011-11-01 | 2011-08-28 | 65 days |
| 2020 COVID-19 Supply Chain Freeze | In-Sample (Historical Control) | 2020-03-15 | — | MISSED |
| 2022-23 Ukraine FX Reserve Drain | Out-of-Sample Test | 2022-04-15 | 2022-01-19 | 86 days |
| 2024 Political Disruption & Peak Inflation | Out-of-Sample Test | 2024-06-25 | 2024-04-26 | 60 days |

**OOS Average Lead Time:** (86 + 60) / 2 = 73 days = 2.43 months

---

## 6. RAG Knowledge Base

The Grounded RAG AI Economist uses 4 curated policy documents (stored in `rag_knowledge_base/`):

| File | Source Institution | Document Type |
|---|---|---|
| `bb_mps_2024.txt` | Bangladesh Bank | Monetary Policy Statement 2024 |
| `bbs_cpi_methodology.txt` | Bangladesh Bureau of Statistics | CPI Methodology Note |
| `wfp_bangladesh_food_security.txt` | UN World Food Programme | Food Security Situation Report |
| `world_bank_bangladesh_economic_update.txt` | World Bank | Bangladesh Economic Update |

> [!NOTE]
> RAG retrieval quality was assessed via Mean Reciprocal Rank (MRR = 0.82) on 25 held-out policy queries. The knowledge base currently covers post-2022 policy documents only. Hallucination reduction relative to un-grounded LLM baseline has not been formally measured.

---

## Reproducibility

All data ingestion scripts are in `scripts/` and `build_daily_master_dataset.py`. To reproduce the master dataset from scratch:

```bash
# 1. Fetch WFP food prices
python scripts/fetch_wfp_food_prices.py

# 2. Fetch Yahoo Finance market data
python scripts/fetch_yahoo_finance_daily.py

# 3. Build consolidated master dataset
python build_daily_master_dataset.py

# 4. Train production models
python train_all_production_models.py
```

**Estimated runtime:** ~15 minutes on a standard laptop (CPU-only).
