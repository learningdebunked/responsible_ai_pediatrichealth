"""Evaluation metrics for RetailHealth models."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix
)


class ModelEvaluator:
    """Comprehensive model evaluation for developmental screening."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def compute_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             y_prob: np.ndarray) -> Dict[str, float]:
        """Compute basic classification metrics."""
        # Check if we have both classes for AUROC/AUPRC
        n_classes = len(np.unique(y_true))
        
        metrics = {}
        
        # Only compute AUROC/AUPRC if both classes are present
        if n_classes > 1:
            try:
                metrics['auroc'] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics['auroc'] = np.nan
            
            try:
                metrics['auprc'] = average_precision_score(y_true, y_prob)
            except ValueError:
                metrics['auprc'] = np.nan
        else:
            metrics['auroc'] = np.nan
            metrics['auprc'] = np.nan
        
        metrics.update({
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        })
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        })
        
        return metrics
    
    def compute_early_detection_lead_time(self, y_true: np.ndarray, y_pred: np.ndarray,
                                         detection_ages: np.ndarray,
                                         diagnosis_ages: np.ndarray) -> Dict[str, float]:
        """Compute early detection lead time."""
        tp_mask = (y_true == 1) & (y_pred == 1)
        
        if tp_mask.sum() == 0:
            return {'mean_lead_time': 0.0, 'median_lead_time': 0.0, 'n_early_detections': 0}
        
        lead_times = diagnosis_ages[tp_mask] - detection_ages[tp_mask]
        
        return {
            'mean_lead_time': float(np.mean(lead_times)),
            'median_lead_time': float(np.median(lead_times)),
            'std_lead_time': float(np.std(lead_times)),
            'n_early_detections': int(tp_mask.sum())
        }
    
    def evaluate_comprehensive(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_prob: np.ndarray, detection_ages: np.ndarray = None,
                              diagnosis_ages: np.ndarray = None) -> Dict:
        """Comprehensive evaluation with all metrics."""
        results = {'basic_metrics': self.compute_basic_metrics(y_true, y_pred, y_prob)}
        
        if detection_ages is not None and diagnosis_ages is not None:
            results['lead_time'] = self.compute_early_detection_lead_time(
                y_true, y_pred, detection_ages, diagnosis_ages
            )
        
        return results

    def compute_trajectory_stability(
        self,
        risk_scores_over_time: np.ndarray,
        window_size: int = 3
    ) -> Dict[str, float]:
        """Compute stability of risk score trajectories over time.

        Measures how much individual risk predictions fluctuate across
        consecutive time windows. High instability suggests the model
        is fitting noise rather than genuine developmental patterns.

        Args:
            risk_scores_over_time: Array of shape (n_families, n_timepoints)
                containing predicted risk scores at each time step.
            window_size: Number of consecutive time points per window.

        Returns:
            Dict with mean_volatility, median_volatility, pct_unstable
            (fraction of families whose risk changes by >0.2 between
            adjacent windows), and flip_rate (fraction whose binary
            prediction flips between windows).
        """
        n_families, n_times = risk_scores_over_time.shape

        if n_times < window_size * 2:
            return {
                'mean_volatility': 0.0,
                'median_volatility': 0.0,
                'pct_unstable': 0.0,
                'flip_rate': 0.0,
                'n_families': n_families,
                'n_timepoints': n_times,
            }

        # Compute per-family volatility as std of risk over time
        volatilities = np.std(risk_scores_over_time, axis=1)

        # Compute window-to-window changes
        n_windows = n_times // window_size
        window_means = np.array([
            risk_scores_over_time[:, i*window_size:(i+1)*window_size].mean(axis=1)
            for i in range(n_windows)
        ]).T  # shape: (n_families, n_windows)

        # Max change between adjacent windows per family
        if n_windows >= 2:
            window_diffs = np.abs(np.diff(window_means, axis=1))
            max_changes = window_diffs.max(axis=1)
            pct_unstable = float(np.mean(max_changes > 0.2))

            # Flip rate: binary prediction changes
            binary_windows = (window_means > self.threshold).astype(int)
            flips = np.abs(np.diff(binary_windows, axis=1))
            flip_rate = float(np.mean(flips.max(axis=1) > 0))
        else:
            pct_unstable = 0.0
            flip_rate = 0.0

        return {
            'mean_volatility': float(np.mean(volatilities)),
            'median_volatility': float(np.median(volatilities)),
            'pct_unstable': pct_unstable,
            'flip_rate': flip_rate,
            'n_families': n_families,
            'n_timepoints': n_times,
        }

    def compute_lead_time_with_baseline(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        detection_ages: np.ndarray,
        diagnosis_ages: np.ndarray,
        nsch_median_diagnosis_age: Dict[str, float] = None,
        delay_types: np.ndarray = None
    ) -> Dict[str, Dict[str, float]]:
        """Compute lead time compared to NSCH population baseline.

        Instead of only computing lead time vs. individual diagnosis,
        also compares detection age to the NSCH median diagnosis age
        for each condition type. This provides a population-level
        benchmark for how early the model detects relative to typical
        clinical practice.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            detection_ages: Age (months) at which model flagged each case
            diagnosis_ages: Age (months) of actual clinical diagnosis
            nsch_median_diagnosis_age: Dict mapping delay_type to median
                diagnosis age in months from NSCH data. If None, uses
                literature defaults.
            delay_types: Array of delay type strings for each sample.
                Required for per-condition analysis.

        Returns:
            Dict with overall lead time stats and per-condition
            lead time vs. NSCH baseline.
        """
        if nsch_median_diagnosis_age is None:
            # Literature defaults (months)
            nsch_median_diagnosis_age = {
                'language': 30,  # AAP guideline: screened at 18-30 mo
                'motor': 24,
                'asd': 52,       # CDC: median age 4y4m (52 mo)
                'adhd': 84,      # typically diagnosed ~7 years
            }

        # Overall lead time (same as existing method)
        overall = self.compute_early_detection_lead_time(
            y_true, y_pred, detection_ages, diagnosis_ages
        )

        result = {'overall': overall}

        # Per-condition analysis vs NSCH baseline
        if delay_types is not None:
            tp_mask = (y_true == 1) & (y_pred == 1)

            for dtype, nsch_median in nsch_median_diagnosis_age.items():
                condition_mask = tp_mask & (delay_types == dtype)
                n_detected = condition_mask.sum()

                if n_detected == 0:
                    result[dtype] = {
                        'n_detected': 0,
                        'nsch_median_diagnosis_months': nsch_median,
                    }
                    continue

                det_ages = detection_ages[condition_mask]
                diag_ages = diagnosis_ages[condition_mask]

                # Lead time vs individual diagnosis
                lead_vs_individual = diag_ages - det_ages

                # Lead time vs NSCH population median
                lead_vs_nsch = nsch_median - det_ages

                result[dtype] = {
                    'n_detected': int(n_detected),
                    'nsch_median_diagnosis_months': nsch_median,
                    'mean_lead_vs_individual': float(np.mean(lead_vs_individual)),
                    'median_lead_vs_individual': float(np.median(lead_vs_individual)),
                    'mean_lead_vs_nsch': float(np.mean(lead_vs_nsch)),
                    'median_lead_vs_nsch': float(np.median(lead_vs_nsch)),
                    'pct_earlier_than_nsch': float(
                        np.mean(det_ages < nsch_median) * 100
                    ),
                    'mean_detection_age': float(np.mean(det_ages)),
                }

        return result