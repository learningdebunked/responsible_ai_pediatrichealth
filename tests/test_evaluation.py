"""Tests for evaluation metrics and fairness modules."""

import pytest
import numpy as np
import pandas as pd
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.fairness import FairnessAnalyzer, SelectionBiasAnalyzer


class TestModelEvaluator:
    """Tests for ModelEvaluator."""

    @pytest.fixture
    def evaluator(self):
        return ModelEvaluator(threshold=0.5)

    def test_basic_metrics(self, evaluator):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.3])
        result = evaluator.compute_basic_metrics(y_true, y_pred, y_prob)
        assert 'accuracy' in result
        assert 'precision' in result
        assert 'recall' in result
        assert 'f1' in result
        assert 'auroc' in result
        assert result['accuracy'] == 0.8

    def test_lead_time(self, evaluator):
        y_true = np.array([1, 1, 0])
        y_pred = np.array([1, 1, 0])
        detection_ages = np.array([12, 18, 0])
        diagnosis_ages = np.array([24, 30, 0])
        result = evaluator.compute_early_detection_lead_time(
            y_true, y_pred, detection_ages, diagnosis_ages
        )
        assert 'mean_lead_time_months' in result
        assert result['mean_lead_time_months'] == 12.0

    def test_trajectory_stability_stable(self, evaluator):
        # Perfectly stable: constant risk over time
        risk = np.ones((10, 12)) * 0.3
        result = evaluator.compute_trajectory_stability(risk, window_size=3)
        assert result['mean_volatility'] == 0.0
        assert result['pct_unstable'] == 0.0
        assert result['flip_rate'] == 0.0

    def test_trajectory_stability_volatile(self, evaluator):
        # Volatile: alternating risk
        risk = np.zeros((10, 12))
        risk[:, ::2] = 0.9
        risk[:, 1::2] = 0.1
        result = evaluator.compute_trajectory_stability(risk, window_size=3)
        assert result['mean_volatility'] > 0.0

    def test_trajectory_stability_short_sequence(self, evaluator):
        risk = np.ones((5, 2)) * 0.5
        result = evaluator.compute_trajectory_stability(risk, window_size=3)
        assert result['mean_volatility'] == 0.0  # Too short

    def test_lead_time_with_baseline(self, evaluator):
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0])
        detection_ages = np.array([18, 24, 40, 0, 0])
        diagnosis_ages = np.array([30, 36, 48, 0, 0])
        delay_types = np.array(['language', 'language', 'asd', '', ''])

        result = evaluator.compute_lead_time_with_baseline(
            y_true, y_pred, detection_ages, diagnosis_ages,
            delay_types=delay_types
        )
        assert 'overall' in result
        assert 'language' in result
        assert result['language']['n_detected'] == 2
        assert result['language']['mean_lead_vs_individual'] == 12.0

    def test_lead_time_with_nsch_defaults(self, evaluator):
        y_true = np.array([1, 1])
        y_pred = np.array([1, 1])
        detection_ages = np.array([20, 30])
        diagnosis_ages = np.array([30, 52])
        delay_types = np.array(['language', 'asd'])

        result = evaluator.compute_lead_time_with_baseline(
            y_true, y_pred, detection_ages, diagnosis_ages,
            delay_types=delay_types
        )
        # NSCH defaults: language=30, asd=52
        assert result['language']['nsch_median_diagnosis_months'] == 30
        assert result['asd']['nsch_median_diagnosis_months'] == 52


class TestFairnessAnalyzer:
    """Tests for FairnessAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return FairnessAnalyzer(sensitive_attributes=['group'])

    def test_demographic_parity(self, analyzer):
        y_pred = np.array([1, 0, 1, 0, 1, 1])
        groups = pd.DataFrame({'group': ['A', 'A', 'A', 'B', 'B', 'B']})
        result = analyzer.compute_demographic_parity(y_pred, groups, 'group')
        assert 'group_rates' in result
        assert 'A' in result['group_rates']
        assert 'B' in result['group_rates']

    def test_equalized_odds(self, analyzer):
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 0, 1])
        groups = pd.DataFrame({'group': ['A', 'A', 'A', 'B', 'B', 'B']})
        result = analyzer.compute_equalized_odds(y_true, y_pred, groups, 'group')
        assert 'tpr_gap' in result
        assert 'fpr_gap' in result


class TestSelectionBiasAnalyzer:
    """Tests for SelectionBiasAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return SelectionBiasAnalyzer()

    def test_estimate_coverage_gap(self, analyzer):
        df = pd.DataFrame({
            'geography': ['urban'] * 50 + ['suburban'] * 30 + ['rural'] * 20
        })
        ref = {'urban': 0.31, 'suburban': 0.52, 'rural': 0.17}
        result = analyzer.estimate_coverage_gap(df, 'geography', ref)
        assert 'urban' in result
        assert 'suburban' in result
        assert 'rural' in result
        assert isinstance(result['urban']['coverage_ratio'], float)

    def test_estimate_coverage_gap_missing_attribute(self, analyzer):
        df = pd.DataFrame({'age': [1, 2, 3]})
        with pytest.raises(ValueError, match="not in DataFrame"):
            analyzer.estimate_coverage_gap(df, 'geography')

    def test_model_missing_populations(self, analyzer):
        df = pd.DataFrame({
            'geography': ['urban'] * 50 + ['rural'] * 50,
        })
        y_true = np.array([1] * 25 + [0] * 25 + [1] * 25 + [0] * 25)
        y_pred = np.array([1] * 20 + [0] * 30 + [1] * 10 + [0] * 40)

        result = analyzer.model_missing_populations(
            df, y_true, y_pred, ['geography']
        )
        assert 'geography' in result
        assert 'urban' in result['geography']
        assert 'rural' in result['geography']
        assert 'f1' in result['geography']['urban']

    def test_generate_selection_bias_report(self, analyzer):
        df = pd.DataFrame({
            'geography': ['urban'] * 60 + ['suburban'] * 30 + ['rural'] * 10,
            'income_quintile': np.random.choice([1, 2, 3, 4, 5], 100),
        })
        report = analyzer.generate_selection_bias_report(df)
        assert 'coverage_gaps' in report
        assert 'recommendations' in report
        assert report['n_total'] == 100
