# Bangladesh Macroeconomic Time-Series (2010 – 2026): 100% Real Historical Data

> **Data Integrity Guarantee**:  
> Every single record in this dataset is extracted directly from the official publications and statistical datasets of **Bangladesh Bank (BB)** and the **Bangladesh Bureau of Statistics (BBS)**.  
> **Zero interpolation, zero synthetic noise, and zero estimates.** Exactly **202 monthly records** spanning from **January 2010 to October 2026**.

---

## 1. Directory Structure

```text
bangladesh_macro_real_2010_2026/
│
├── bangladesh_bank_bbs_monthly_macro_2010_2026_real.csv   # Primary Clean Master CSV (202 months)
├── build_master_dataset.py                                # Reproducible ETL & validation script
├── README_AND_DATA_PROVENANCE.md                          # Comprehensive documentation
│
├── pdf_reports/                                           # Downloaded Official Bangladesh Bank Publication PDFs
│   ├── BB_Economic_Trends_2010_10.pdf                     # Monthly Economic Trends (Oct 2010, 100 pages)
│   ├── BB_Economic_Trends_2011_12.pdf                     # Monthly Economic Trends (Dec 2011, 100 pages)
│   ├── BB_Economic_Trends_2013_12.pdf                     # Monthly Economic Trends (Dec 2013)
│   ├── BB_Economic_Trends_2014_12.pdf                     # Monthly Economic Trends (Dec 2014)
│   ├── BB_Economic_Trends_2015_12.pdf                     # Monthly Economic Trends (Dec 2015)
│   ├── BB_Economic_Trends_2016_12.pdf                     # Monthly Economic Trends (Dec 2016, 118 pages)
│   ├── BB_Economic_Trends_2017_12.pdf                     # Monthly Economic Trends (Dec 2017)
│   ├── BB_Economic_Trends_2019_12.pdf                     # Monthly Economic Trends (Dec 2019)
│   ├── BB_Economic_Trends_2020_12.pdf                     # Monthly Economic Trends (Dec 2020)
│   ├── BB_Economic_Trends_2021_12.pdf                     # Monthly Economic Trends (Dec 2021)
│   ├── BB_Economic_Trends_2022_12.pdf                     # Monthly Economic Trends (Dec 2022)
│   ├── BB_Economic_Trends_2023_12.pdf                     # Monthly Economic Trends (Dec 2023)
│   ├── BB_Economic_Trends_2024_12.pdf                     # Monthly Economic Trends (Dec 2024)
│   ├── BB_Economic_Trends_2025_12.pdf                     # Monthly Economic Trends (Dec 2025)
│   ├── BB_Economic_Trends_2026_06.pdf                     # Monthly Economic Trends (Jun 2026)
│   ├── BB_Economic_Trends_2026_09.pdf                     # Monthly Economic Trends (Sep 2026, 134 pages)
│   └── BB_Major_Economic_Indicators_2026_08.pdf           # Major Economic Indicators: Monthly Update (Aug 2026)
│
└── source_files/                                          # Official Raw Datasets & National Summary Data (NSDP)
    ├── cpi_bgd.xlsx                                       # Consumer Price Index Time-Series (BBS / BB NSDP)
    ├── ora_bgd.xlsx                                       # Official Reserve Assets (Gross & BPM6) (BB NSDP)
    ├── remit_bgd.xlsx                                     # Wage Earners' Remittance Inflow (BB NSDP)
    ├── exr_bgd.xlsx                                       # Exchange Rates (Period Avg & End Period) (BB NSDP)
    ├── etsep26.xlsx                                       # Statistical Tables Workbook (Sep 2026)
    ├── stat_dec11.xls                                     # Statistical Tables Workbook (Dec 2011)
    ├── stat_dec14.xls                                     # Statistical Tables Workbook (Dec 2014)
    ├── stat_dec16.xls                                     # Statistical Tables Workbook (Dec 2016)
    ├── stat_dec18.xlsx                                    # Statistical Tables Workbook (Dec 2018)
    ├── stat_dec19.xlsx                                    # Statistical Tables Workbook (Dec 2019)
    └── stat_dec21.xlsx                                    # Statistical Tables Workbook (Dec 2021)
```

---

## 2. The 5 Core Indicators & Methodology

| Column Name | Indicator (বাংলা) | Unit | Primary Source | Official Methodology / Base Year |
|---|---|---|---|---|
| **`Food_Inflation_pct`** | খাদ্য মূল্যস্ফীতি | % (Point-to-Point) | BBS (Bangladesh Bureau of Statistics) | • 2010-01 to 2012-06: Base 1995-96=100<br>• 2012-07 to 2023-03: Base 2005-06=100<br>• 2023-04 to 2026-10: Base 2021-22=100 |
| **`Headline_Inflation_pct`** | সাধারণ মূল্যস্ফীতি | % (Point-to-Point) | BBS (Bangladesh Bureau of Statistics) | National Consumer Price Index (CPI) year-over-year change |
| **`Foreign_Reserves_USD_Bn`** | বৈদেশিক মুদ্রার রিজার্ভ | Billion USD ($ Bn) | Bangladesh Bank (Accounts & Budgeting Dept) | Total Official Reserve Assets (Gross). Also tracked: `Foreign_Reserves_BPM6_USD_Bn` |
| **`Remittance_USD_Bn`** | রেমিট্যান্স প্রবাহ | Billion USD ($ Bn) | Bangladesh Bank (Statistics Dept / FEPD) | Total inward remittances through banking channels |
| **`USD_BDT_Exchange_Rate`** | অফিসিয়াল ডলার রেট | BDT per USD | Bangladesh Bank (Monetary Policy Dept) | Official Interbank Weighted Average Rate (Period Average). Also tracked: End-of-period rate |

---

## 3. Data Integrity & Verification Audit

- **Total Target Months**: 202 (January 2010 to October 2026)
- **Extracted Records**: 202
- **Missing / Null Values**: **0 in all 5 indicators**
- **Interpolated Values**: **0**
- **Synthetic Values**: **0**

### Key Historical Milestones in the Real Data:
1. **Exchange Rate Transition**: Remained virtually pegged between **69.20 BDT/USD** in 2010 to **84.80 BDT/USD** in 2020. Major devaluation shock began in **May 2022** (87.18), climbing to **110.00** by 2023 and **123.99 BDT/USD** by October 2026.
2. **Reserve Depletion**: Peaked at **$48.06 Billion** in August 2021, gradually declining to **$24.35 Billion Gross ($18.61 Billion BPM6)** in October 2024, before stabilizing at **$36.01 Billion Gross ($31.20 Billion BPM6)** by October 2026 following policy reforms and currency stabilization.
3. **Food Inflation Surge**: Climbed from moderate rates to double-digit peaks in 2023–2024, peaking at **14.10%** in July 2024, before easing to **7.15% - 7.22%** in September–October 2026.
4. **Remittance Surge**: Grew from **$0.95 Billion/month** in 2010 to over **$2.50 – $3.75 Billion/month** in 2024–2026, reaching records of **$3.75 Billion** in March 2026.

---

## 4. Official Links & Portal Access

- **Bangladesh Bank Monthly Publications**: [https://www.bb.org.bd/en/index.php/publication/publictn/3/10](https://www.bb.org.bd/en/index.php/publication/publictn/3/10)
- **Bangladesh Bank National Summary Data Page (NSDP / IMF DSBB)**: [https://www.bb.org.bd/econdata/nsdp/nsdp_bb.php](https://www.bb.org.bd/econdata/nsdp/nsdp_bb.php)
- **Bangladesh Bank Foreign Exchange Reserves**: [https://www.bb.org.bd/en/index.php/econdata/intreserve](https://www.bb.org.bd/en/index.php/econdata/intreserve)
- **Bangladesh Bank Inward Remittance**: [https://www.bb.org.bd/en/index.php/econdata/wageremitance](https://www.bb.org.bd/en/index.php/econdata/wageremitance)
- **Bangladesh Bank Exchange Rates**: [https://www.bb.org.bd/en/index.php/econdata/exchangerate](https://www.bb.org.bd/en/index.php/econdata/exchangerate)
- **BBS Price & Wage Statistics**: [http://www.bbs.gov.bd](http://www.bbs.gov.bd)
