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