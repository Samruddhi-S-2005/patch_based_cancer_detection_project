import re

with open("tests/test_pathovision_suite.py", "r", encoding="utf-8") as f:
    text = f.read()

old_test_36 = '''    def test_36_learned_feature_analysis_decoupling_and_preservation(self):
        """Verifies that Learned Feature Analysis is decoupled from primary navigation and preserved in research docs & artifacts."""
        app_file = PROJECT_ROOT / "app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            app_text = f.read()

        # Primary navigation must NOT include Learned Feature Analysis
        nav_block = app_text.split("nav_options = [")[1].split("]")[0]
        self.assertNotIn('"🧬 Learned Feature Analysis"', nav_block, "Learned Feature Analysis must not be in primary navigation")

        # Active routing must not contain standalone section 4
        self.assertNotIn('elif nav == "🧬 Learned Feature Analysis":', app_text)

        # Documentation pointer and preserved research doc
        self.assertIn("learned_feature_analysis_and_latent_space.md", app_text)
        doc_path = PROJECT_ROOT / "docs" / "research" / "learned_feature_analysis_and_latent_space.md"
        self.assertTrue(doc_path.exists(), f"Learned feature analysis research doc missing: {doc_path}")

        # Check preserved implementation script and projections artifact
        script_path = PROJECT_ROOT / "src" / "objective_2" / "feature_analysis.py"
        self.assertTrue(script_path.exists(), f"Feature analysis script missing: {script_path}")
        projections_path = PROJECT_ROOT / "result" / "objective_2" / "feature_analysis" / "feature_projections.json"
        self.assertTrue(projections_path.exists(), f"Feature projections JSON missing: {projections_path}")

        # Check feature dimensions table and morphology audit in research doc and app
        with open(doc_path, "r", encoding="utf-8") as f:
            doc_text = f.read()
        self.assertIn("1,280-D", doc_text)
        self.assertIn("2,048-D", doc_text)
        self.assertIn("1,024-D", doc_text)
        self.assertIn("4,096-D", doc_text)
        self.assertIn("What PathoVision AI Does NOT Do", doc_text)
        self.assertIn("What PathoVision AI Actually Computes", doc_text)

        # Check Model Evaluation expander presence
        self.assertIn("Supplementary Learned Feature Representations & Embeddings", app_text)'''

new_test_36 = '''    def test_36_learned_feature_analysis_restoration_and_accessibility(self):
        """Verifies that Learned Feature Analysis is fully restored in web app navigation and backed by cached projections."""
        app_file = PROJECT_ROOT / "app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            app_text = f.read()

        # Primary navigation MUST include Learned Feature Analysis
        nav_block = app_text.split("nav_options = [")[1].split("]")[0]
        self.assertIn('"🧬 Learned Feature Analysis"', nav_block, "Learned Feature Analysis must be restored in navigation")

        # Active routing must contain standalone Section 4
        self.assertIn('elif nav == "🧬 Learned Feature Analysis":', app_text)

        # Documentation pointer and preserved research doc
        self.assertIn("learned_feature_analysis_and_latent_space.md", app_text)
        doc_path = PROJECT_ROOT / "docs" / "research" / "learned_feature_analysis_and_latent_space.md"
        self.assertTrue(doc_path.exists(), f"Learned feature analysis research doc missing: {doc_path}")

        # Check implementation script and projections artifact
        script_path = PROJECT_ROOT / "src" / "objective_2" / "feature_analysis.py"
        self.assertTrue(script_path.exists(), f"Feature analysis script missing: {script_path}")
        projections_path = PROJECT_ROOT / "result" / "objective_2" / "feature_analysis" / "feature_projections.json"
        self.assertTrue(projections_path.exists(), f"Feature projections JSON missing: {projections_path}")

        with open(projections_path, "r", encoding="utf-8") as pf:
            proj_data = json.load(pf)
        self.assertIn("models", proj_data)
        self.assertEqual(len(proj_data.get("samples", [])), 120)

        # Check feature dimensions table and morphology audit in research doc and app
        with open(doc_path, "r", encoding="utf-8") as f:
            doc_text = f.read()
        self.assertIn("1,280-D", doc_text)
        self.assertIn("2,048-D", doc_text)
        self.assertIn("1,024-D", doc_text)
        self.assertIn("4,096-D", doc_text)
        self.assertIn("What PathoVision AI Does NOT Do", doc_text)
        self.assertIn("What PathoVision AI Actually Computes", doc_text)

        # Check app contains table and morphology audit
        self.assertIn("1,280-D", app_text)
        self.assertIn("What PathoVision AI Does NOT Do", app_text)
        self.assertIn("What PathoVision AI Actually Computes", app_text)

        # Check Normal/Anomaly clarification presence in app
        self.assertIn("An anomaly pattern was flagged, but the primary tumour classifier did not exceed its tumour threshold", app_text)

        # Check WSI Viewer enhancements presence in app
        self.assertIn("Spatial Heatmap Visualization Mode:", app_text)
        self.assertIn("0.00 — Normal Lymph Node Stroma", app_text)'''

assert old_test_36 in text, "old_test_36 not found"
text = text.replace(old_test_36, new_test_36, 1)

with open("tests/test_pathovision_suite.py", "w", encoding="utf-8") as f:
    f.write(text)

print("test_36 successfully updated in tests/test_pathovision_suite.py")
