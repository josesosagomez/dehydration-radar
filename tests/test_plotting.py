"""T-M7-plotting: the Exp A scatter renders headlessly (Agg), and matplotlib coexists with
the scipy <1.17 pin."""


def test_matplotlib_is_pinned_and_scipy_stays_below_1_17():
    import matplotlib
    import scipy

    assert scipy.__version__ < "1.17"
    assert matplotlib.__version__  # importable in the pinned env


def test_headless_agg_scatter_renders_a_png(tmp_path):
    import numpy as np

    from dehyd.eval.exp_a import write_exp_a_reports
    from dehyd.eval.harness import FitRecord, SeedOutcome

    class R:  # a minimal ExpAFoldResult-like object
        def __init__(self, subj, y, pred):
            self.test_subject = subj
            self.test_targets = np.array(y, float)
            self.test_predictions = np.array(pred, float)
            self.seed_outcomes = [SeedOutcome(0, np.array(pred, float), np.array(pred, float), 0.1)]
            self.selected_feature_key = (0, "A", "mag", 0, "off")
            self.selected_family = "ridge"
            self.selected_params = {"alpha": 1.0}
            self.final_fits = [FitRecord("scaler", "outer_train", frozenset({2, 3}), {"m": np.zeros(1)})]

    results = [R(1, [-1.0, -2.0], [-1.1, -1.8]), R(2, [-0.5, -3.0], [-0.6, -2.7])]
    summary = {"conditional_exploratory": True, "subject_balanced_mae": {"point": 0.2}}
    paths = write_exp_a_reports(results, summary, tmp_path, "10ghz")

    assert paths["scatter"].exists()
    assert paths["scatter"].stat().st_size > 0
    assert paths["metrics"].exists() and paths["predictions"].exists()
