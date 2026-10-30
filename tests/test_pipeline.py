"""
Automated Research Pipeline Test Suite
Validates:
1. Dataset schema integrity, publication lags, and zero synthetic leakages.
2. Production model artifacts serialization, inputs, and inference validity.
3. Grounded RAG knowledge base files and advisor artifact structure.
"""

import os
import unittest
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestDatasetIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_path = os.path.join(BASE_DIR, "daily_macro_master_dataset.csv")
        cls.bday_path = os.path.join(BASE_DIR, "daily_macro_business_days.csv")
        cls.df = pd.read_csv(cls.dataset_path)
        cls.df["Date"] = pd.to_datetime(cls.df["Date"])

    def test_file_exists_and_rows(self):
        self.assertTrue(os.path.exists(self.dataset_path), "Master dataset file missing!")
        # Row count grows daily; require at least 6,000 rows (Jan 2010 – present)
        self.assertGreaterEqual(len(self.df), 6000, f"Expected at least 6,000 rows, got {len(self.df)}")

    def test_column_count_and_schema(self):
        self.assertIn("Daily_Economic_Stress_Index", self.df.columns)
        self.assertIn("Combined_Anomaly_Flag", self.df.columns)
        self.assertGreaterEqual(len(self.df.columns), 41)

    def test_purged_fabricated_series(self):
        self.assertNotIn("Price_Rice_Miniket_BDT_kg", self.df.columns,
                         "CRITICAL: Miniket rice must not be in dataset (lacks daily provenance).")

    def test_date_monotonicity_and_span(self):
        dates = self.df["Date"]
        self.assertTrue(dates.is_monotonic_increasing, "Timeline is not monotonically increasing!")
        self.assertEqual(dates.min().strftime("%Y-%m-%d"), "2010-01-01")
        # Max date covers the benchmark period through at least September 2026
        import datetime
        max_date = dates.max().date()
        self.assertGreaterEqual(max_date, datetime.date(2026, 9, 30),
                                f"Dataset max date {max_date} must reach at least 2026-09-30")

    def test_core_indicators_zero_nulls(self):
        core_continuous = [
            "USD_BDT_Rate", "Brent_Crude_Oil_USD", "US_Dollar_Index_DXY", "Gold_USD_oz",
            "Natural_Gas_USD", "Wheat_Price_USD", "Headline_Inflation_pct", "Food_Inflation_pct",
            "Foreign_Reserves_USD_Bn", "Remittance_USD_Bn", "Price_Rice_Coarse_BDT_kg",
            "Price_Lentil_BDT_kg", "Price_Flour_BDT_kg", "Daily_Economic_Stress_Index"
        ]
        for col in core_continuous:
            self.assertIn(col, self.df.columns)
            null_count = self.df[col].isna().sum()
            self.assertEqual(null_count, 0, f"Core column '{col}' has {null_count} unexpected NaNs!")

    def test_publication_lag_and_availability_flags(self):
        availability_flags = ["Soybean_Oil_Available", "Sugar_Available", "Onion_Available"]
        for flag in availability_flags:
            self.assertIn(flag, self.df.columns)
            pre_2024 = self.df[self.df["Date"] < "2024-02-15"][flag]
            post_2024 = self.df[self.df["Date"] >= "2024-02-15"][flag]
            self.assertEqual(pre_2024.sum(), 0, f"Flag {flag} must be 0 before 2024-02-15!")
            self.assertEqual(post_2024.mean(), 1.0, f"Flag {flag} must be 1 after 2024-02-15!")


class TestModelArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(os.path.join(BASE_DIR, "daily_macro_master_dataset.csv"))
        cls.df["Date"] = pd.to_datetime(cls.df["Date"])
        cls.latest_row = cls.df.iloc[-1:]

    def test_usd_forecaster(self):
        model_path = os.path.join(BASE_DIR, "best_forecasting_model.pkl")
        self.assertTrue(os.path.exists(model_path))
        model = joblib.load(model_path)
        feats = list(getattr(model, "feature_names_in_", []))
        self.assertGreater(len(feats), 0)
        X = self.latest_row[feats]
        pred_delta = float(model.predict(X)[0])
        curr_rate = float(self.latest_row["USD_BDT_Rate"].values[0])
        pred_rate = curr_rate * (1.0 + pred_delta / 100.0)
        self.assertGreater(pred_rate, 50.0)
        self.assertLess(pred_rate, 250.0)

    def test_food_inflation_model(self):
        model_path = os.path.join(BASE_DIR, "best_food_inflation_model.pkl")
        self.assertTrue(os.path.exists(model_path))
        model = joblib.load(model_path)
        feats = list(getattr(model, "feature_names_in_", []))
        self.assertGreater(len(feats), 0)
        X = self.latest_row[feats]
        pred_delta = float(model.predict(X)[0])
        curr_inf = float(self.latest_row["Food_Inflation_pct"].values[0])
        pred_inf = curr_inf + pred_delta
        self.assertGreater(pred_inf, -5.0)
        self.assertLess(pred_inf, 35.0)

    def test_crisis_classifier(self):
        model_path = os.path.join(BASE_DIR, "best_crisis_classifier.pkl")
        self.assertTrue(os.path.exists(model_path))
        model = joblib.load(model_path)
        feats = list(getattr(model, "feature_names_in_", []))
        self.assertGreater(len(feats), 0)
        X = self.latest_row[feats]
        prob = float(model.predict_proba(X)[0, 1])
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_commodity_models(self):
        model_path = os.path.join(BASE_DIR, "best_commodity_models.pkl")
        self.assertTrue(os.path.exists(model_path))
        comm_models = joblib.load(model_path)
        expected_commodities = [
            'Price_Rice_Coarse_BDT_kg', 'Price_Lentil_BDT_kg',
            'Price_Soybean_Oil_BDT_liter', 'Price_Flour_BDT_kg',
            'Price_Onion_BDT_kg', 'Price_Sugar_BDT_kg'
        ]
        self.assertEqual(sorted(list(comm_models.keys())), sorted(expected_commodities))
        for col_name, meta in comm_models.items():
            model = meta["model"]
            feats = meta["features"]
            X = self.latest_row[feats]
            pred_delta = float(model.predict(X)[0])
            self.assertTrue(np.isfinite(pred_delta), f"Non-finite prediction for {col_name}")

    def test_isolation_forest(self):
        model_path = os.path.join(BASE_DIR, "best_isolation_forest.pkl")
        self.assertTrue(os.path.exists(model_path))
        iso_art = joblib.load(model_path)
        self.assertIn("oos_recall", iso_art["metrics"])
        self.assertEqual(iso_art["metrics"]["oos_recall"], 100.0)
        self.assertGreaterEqual(iso_art["metrics"]["oos_avg_lead_months"], 2.0)


class TestGroundedRAG(unittest.TestCase):
    def test_policy_documents_present(self):
        rag_dir = os.path.join(BASE_DIR, "rag_knowledge_base")
        self.assertTrue(os.path.isdir(rag_dir))
        expected_docs = [
            "bb_monetary_policy_statement.txt",
            "bbs_cpi_inflation_report.txt",
            "imf_article_iv_bangladesh_reserves.txt",
            "tcb_open_market_sale_policy.txt"
        ]
        for doc in expected_docs:
            p = os.path.join(rag_dir, doc)
            self.assertTrue(os.path.exists(p), f"Missing policy document: {doc}")
            self.assertGreater(os.path.getsize(p), 500)

    def test_advisor_artifact_structure(self):
        p = os.path.join(BASE_DIR, "grounded_rag_advisor.pkl")
        self.assertTrue(os.path.exists(p))
        art = joblib.load(p)
        self.assertIn("quant_state", art)
        self.assertIn("eval_results", art)
        self.assertEqual(art["mean_grounding_pct"], 82.0)


if __name__ == "__main__":
    unittest.main()
