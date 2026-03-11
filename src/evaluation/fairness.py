"""Fairness analysis metrics for demographic equity evaluation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix


class FairnessAnalyzer:
    """Analyzes fairness metrics across demographic groups."""
    
    def __init__(self, sensitive_attributes: List[str] = None):
        """Initialize fairness analyzer.
        
        Args:
            sensitive_attributes: List of sensitive attribute names
                (e.g., ['income_quintile', 'geography', 'ethnicity'])
        """
        if sensitive_attributes is None:
            sensitive_attributes = ['income_quintile', 'geography', 'ethnicity']
        self.sensitive_attributes = sensitive_attributes
    
    def compute_group_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             groups: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute performance metrics for each group.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (binary)
            groups: Group identifiers
            
        Returns:
            Dictionary mapping group names to metrics
        """
        unique_groups = np.unique(groups)
        group_metrics = {}
        
        for group in unique_groups:
            mask = groups == group
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            if len(y_true_group) == 0:
                continue
            
            # Compute confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
            
            # Compute metrics
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value (Precision)
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
            
            group_metrics[str(group)] = {
                'tpr': tpr,
                'fpr': fpr,
                'tnr': tnr,
                'fnr': fnr,
                'ppv': ppv,
                'npv': npv,
                'n_samples': len(y_true_group),
                'n_positive': int(y_true_group.sum()),
                'n_predicted_positive': int(y_pred_group.sum())
            }
        
        return group_metrics
    
    def demographic_parity(self, y_pred: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
        """Compute demographic parity (equal positive prediction rates).
        
        Args:
            y_pred: Predicted labels
            groups: Group identifiers
            
        Returns:
            Dictionary with parity metrics
        """
        unique_groups = np.unique(groups)
        positive_rates = {}
        
        for group in unique_groups:
            mask = groups == group
            positive_rate = y_pred[mask].mean()
            positive_rates[str(group)] = positive_rate
        
        # Compute max difference
        rates = list(positive_rates.values())
        max_diff = max(rates) - min(rates)
        
        return {
            'positive_rates': positive_rates,
            'max_difference': max_diff,
            'ratio': min(rates) / max(rates) if max(rates) > 0 else 0
        }
    
    def equalized_odds(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       groups: np.ndarray) -> Dict[str, float]:
        """Compute equalized odds (equal TPR and FPR across groups).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            groups: Group identifiers
            
        Returns:
            Dictionary with equalized odds metrics
        """
        group_metrics = self.compute_group_metrics(y_true, y_pred, groups)
        
        tprs = [m['tpr'] for m in group_metrics.values()]
        fprs = [m['fpr'] for m in group_metrics.values()]
        
        tpr_gap = max(tprs) - min(tprs)
        fpr_gap = max(fprs) - min(fprs)
        
        return {
            'tpr_gap': tpr_gap,
            'fpr_gap': fpr_gap,
            'max_gap': max(tpr_gap, fpr_gap),
            'group_tprs': {g: m['tpr'] for g, m in group_metrics.items()},
            'group_fprs': {g: m['fpr'] for g, m in group_metrics.items()}
        }
    
    def equal_opportunity(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         groups: np.ndarray) -> Dict[str, float]:
        """Compute equal opportunity (equal TPR across groups).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            groups: Group identifiers
            
        Returns:
            Dictionary with equal opportunity metrics
        """
        group_metrics = self.compute_group_metrics(y_true, y_pred, groups)
        
        tprs = {g: m['tpr'] for g, m in group_metrics.items()}
        tpr_values = list(tprs.values())
        
        return {
            'tpr_gap': max(tpr_values) - min(tpr_values),
            'group_tprs': tprs,
            'min_tpr': min(tpr_values),
            'max_tpr': max(tpr_values)
        }
    
    def predictive_parity(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         groups: np.ndarray) -> Dict[str, float]:
        """Compute predictive parity (equal PPV across groups).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            groups: Group identifiers
            
        Returns:
            Dictionary with predictive parity metrics
        """
        group_metrics = self.compute_group_metrics(y_true, y_pred, groups)
        
        ppvs = {g: m['ppv'] for g, m in group_metrics.items()}
        ppv_values = list(ppvs.values())
        
        return {
            'ppv_gap': max(ppv_values) - min(ppv_values),
            'group_ppvs': ppvs,
            'min_ppv': min(ppv_values),
            'max_ppv': max(ppv_values)
        }
    
    def calibration_by_group(self, y_true: np.ndarray, y_prob: np.ndarray, 
                            groups: np.ndarray, n_bins: int = 10) -> Dict[str, Dict]:
        """Compute calibration metrics by group.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            groups: Group identifiers
            n_bins: Number of probability bins
            
        Returns:
            Dictionary with calibration metrics per group
        """
        unique_groups = np.unique(groups)
        calibration = {}
        
        for group in unique_groups:
            mask = groups == group
            y_true_group = y_true[mask]
            y_prob_group = y_prob[mask]
            
            if len(y_true_group) == 0:
                continue
            
            # Bin predictions
            bins = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(y_prob_group, bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            # Compute calibration per bin
            bin_true_rates = []
            bin_pred_rates = []
            bin_counts = []
            
            for i in range(n_bins):
                bin_mask = bin_indices == i
                if bin_mask.sum() > 0:
                    bin_true_rate = y_true_group[bin_mask].mean()
                    bin_pred_rate = y_prob_group[bin_mask].mean()
                    bin_true_rates.append(bin_true_rate)
                    bin_pred_rates.append(bin_pred_rate)
                    bin_counts.append(bin_mask.sum())
            
            # Expected Calibration Error (ECE)
            ece = 0
            total_samples = len(y_true_group)
            for true_rate, pred_rate, count in zip(bin_true_rates, bin_pred_rates, bin_counts):
                ece += (count / total_samples) * abs(true_rate - pred_rate)
            
            calibration[str(group)] = {
                'ece': ece,
                'bin_true_rates': bin_true_rates,
                'bin_pred_rates': bin_pred_rates,
                'bin_counts': bin_counts
            }
        
        return calibration
    
    def analyze_all(self, y_true: np.ndarray, y_pred: np.ndarray, 
                   y_prob: np.ndarray, data: pd.DataFrame) -> Dict[str, Dict]:
        """Compute all fairness metrics across all sensitive attributes.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (binary)
            y_prob: Predicted probabilities
            data: DataFrame with sensitive attributes
            
        Returns:
            Comprehensive fairness analysis
        """
        results = {}
        
        for attr in self.sensitive_attributes:
            if attr not in data.columns:
                continue
            
            groups = data[attr].values
            
            results[attr] = {
                'group_metrics': self.compute_group_metrics(y_true, y_pred, groups),
                'demographic_parity': self.demographic_parity(y_pred, groups),
                'equalized_odds': self.equalized_odds(y_true, y_pred, groups),
                'equal_opportunity': self.equal_opportunity(y_true, y_pred, groups),
                'predictive_parity': self.predictive_parity(y_true, y_pred, groups),
                'calibration': self.calibration_by_group(y_true, y_prob, groups)
            }
        
        return results
    
    def generate_report(self, fairness_results: Dict[str, Dict]) -> str:
        """Generate human-readable fairness report.
        
        Args:
            fairness_results: Output from analyze_all()
            
        Returns:
            Formatted report string
        """
        report = ["\n" + "="*80]
        report.append("FAIRNESS ANALYSIS REPORT")
        report.append("="*80 + "\n")
        
        for attr, metrics in fairness_results.items():
            report.append(f"\n{attr.upper().replace('_', ' ')}:")
            report.append("-" * 40)
            
            # Equalized Odds
            eo = metrics['equalized_odds']
            report.append(f"\n  Equalized Odds:")
            report.append(f"    TPR Gap: {eo['tpr_gap']:.4f}")
            report.append(f"    FPR Gap: {eo['fpr_gap']:.4f}")
            report.append(f"    Max Gap: {eo['max_gap']:.4f}")
            
            # Equal Opportunity
            eop = metrics['equal_opportunity']
            report.append(f"\n  Equal Opportunity:")
            report.append(f"    TPR Gap: {eop['tpr_gap']:.4f}")
            report.append(f"    Min TPR: {eop['min_tpr']:.4f}")
            report.append(f"    Max TPR: {eop['max_tpr']:.4f}")
            
            # Demographic Parity
            dp = metrics['demographic_parity']
            report.append(f"\n  Demographic Parity:")
            report.append(f"    Max Difference: {dp['max_difference']:.4f}")
            report.append(f"    Min/Max Ratio: {dp['ratio']:.4f}")
            
            # Group-specific metrics
            report.append(f"\n  Group Performance:")
            for group, gm in metrics['group_metrics'].items():
                report.append(f"    {group}:")
                report.append(f"      TPR: {gm['tpr']:.4f}, FPR: {gm['fpr']:.4f}, PPV: {gm['ppv']:.4f}")
                report.append(f"      Samples: {gm['n_samples']}, Positive: {gm['n_positive']}")
        
        report.append("\n" + "="*80)
        return "\n".join(report)


def check_fairness_thresholds(fairness_results: Dict[str, Dict], 
                             tpr_gap_threshold: float = 0.10,
                             fpr_gap_threshold: float = 0.10) -> Tuple[bool, List[str]]:
    """Check if fairness metrics meet specified thresholds.
    
    Args:
        fairness_results: Output from FairnessAnalyzer.analyze_all()
        tpr_gap_threshold: Maximum acceptable TPR gap
        fpr_gap_threshold: Maximum acceptable FPR gap
        
    Returns:
        Tuple of (passes_all_checks, list_of_violations)
    """
    violations = []
    
    for attr, metrics in fairness_results.items():
        eo = metrics['equalized_odds']
        
        if eo['tpr_gap'] > tpr_gap_threshold:
            violations.append(
                f"{attr}: TPR gap {eo['tpr_gap']:.4f} exceeds threshold {tpr_gap_threshold}"
            )
        
        if eo['fpr_gap'] > fpr_gap_threshold:
            violations.append(
                f"{attr}: FPR gap {eo['fpr_gap']:.4f} exceeds threshold {fpr_gap_threshold}"
            )
    
    return len(violations) == 0, violations


class SelectionBiasAnalyzer:
    """Analyze selection bias in the training population.

    Estimates how the population captured by retail purchase data
    differs from the general pediatric population, identifies
    coverage gaps, and models the impact of missing populations
    on model fairness.
    """

    def __init__(self, reference_prevalence: Dict[str, Dict[str, float]] = None):
        """Initialize selection bias analyzer.

        Args:
            reference_prevalence: Reference population prevalence rates
                from NSCH or similar survey, structured as:
                {attribute: {group: prevalence_rate}}
                If None, uses NSCH 2022 defaults.
        """
        if reference_prevalence is None:
            self.reference = {
                'income_quintile': {
                    '1': 0.223, '2': 0.198, '3': 0.172,
                    '4': 0.151, '5': 0.124,
                },
                'geography': {
                    'urban': 0.165, 'suburban': 0.172, 'rural': 0.201,
                },
                'ethnicity': {
                    'white': 0.172, 'black': 0.198, 'hispanic': 0.185,
                    'asian': 0.142, 'other': 0.189,
                },
            }
        else:
            self.reference = reference_prevalence

    def estimate_coverage_gap(
        self,
        observed_df: pd.DataFrame,
        attribute: str,
        reference_distribution: Dict[str, float] = None
    ) -> Dict[str, Dict]:
        """Estimate coverage gap between observed and reference populations.

        Args:
            observed_df: DataFrame with demographic columns
            attribute: Demographic attribute to analyze
            reference_distribution: Expected population proportions per group.
                If None, assumes uniform.

        Returns:
            Dict mapping group to {observed_pct, reference_pct, gap,
            coverage_ratio, underrepresented}
        """
        if attribute not in observed_df.columns:
            raise ValueError(f"Attribute '{attribute}' not in DataFrame")

        observed_counts = observed_df[attribute].value_counts(normalize=True)
        all_groups = set(observed_counts.index)

        if reference_distribution:
            all_groups |= set(reference_distribution.keys())
        else:
            reference_distribution = {
                g: 1.0 / len(all_groups) for g in all_groups
            }

        results = {}
        for group in sorted(all_groups, key=str):
            obs_pct = float(observed_counts.get(group, 0.0))
            ref_pct = reference_distribution.get(str(group), 0.0)

            coverage_ratio = obs_pct / ref_pct if ref_pct > 0 else 0.0

            results[str(group)] = {
                'observed_pct': obs_pct,
                'reference_pct': ref_pct,
                'gap': ref_pct - obs_pct,
                'coverage_ratio': coverage_ratio,
                'underrepresented': coverage_ratio < 0.8,
            }

        return results

    def model_missing_populations(
        self,
        observed_df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        attributes: List[str] = None
    ) -> Dict[str, Dict]:
        """Model which populations are missing and estimate impact.

        For each demographic group, computes:
        - Representation ratio vs reference population
        - Performance metrics for the group
        - Estimated performance degradation due to under-representation

        Args:
            observed_df: DataFrame with demographic columns
            y_true: True labels
            y_pred: Predicted labels
            attributes: Attributes to analyze. Defaults to all reference keys.

        Returns:
            Dict with per-attribute, per-group analysis
        """
        from sklearn.metrics import f1_score, recall_score

        if attributes is None:
            attributes = list(self.reference.keys())

        results = {}

        for attr in attributes:
            if attr not in observed_df.columns:
                continue

            attr_results = {}
            groups = observed_df[attr].unique()

            for group in groups:
                mask = observed_df[attr] == group
                n_group = mask.sum()

                if n_group < 10:
                    continue

                group_y_true = y_true[mask]
                group_y_pred = y_pred[mask]

                group_f1 = f1_score(group_y_true, group_y_pred, zero_division=0)
                group_recall = recall_score(group_y_true, group_y_pred, zero_division=0)

                # Reference prevalence
                ref_prev = self.reference.get(attr, {}).get(str(group), None)

                # Representation ratio
                obs_pct = n_group / len(observed_df)
                # Assume uniform reference if not specified
                ref_pct = 1.0 / len(groups) if ref_prev is None else ref_prev
                rep_ratio = obs_pct / ref_pct if ref_pct > 0 else 0.0

                attr_results[str(group)] = {
                    'n': int(n_group),
                    'observed_pct': float(obs_pct),
                    'f1': float(group_f1),
                    'recall': float(group_recall),
                    'representation_ratio': float(rep_ratio),
                    'underrepresented': rep_ratio < 0.8,
                    'performance_concern': group_f1 < 0.5 or group_recall < 0.5,
                }

            results[attr] = attr_results

        return results

    def generate_selection_bias_report(
        self,
        observed_df: pd.DataFrame,
        y_true: np.ndarray = None,
        y_pred: np.ndarray = None,
        attributes: List[str] = None
    ) -> Dict:
        """Generate comprehensive selection bias report.

        Args:
            observed_df: DataFrame with demographic columns
            y_true: True labels (optional, enables performance analysis)
            y_pred: Predicted labels (optional)
            attributes: Attributes to analyze

        Returns:
            Dict with coverage_gaps, missing_populations, and recommendations
        """
        if attributes is None:
            attributes = [a for a in self.reference.keys()
                          if a in observed_df.columns]

        report = {
            'n_total': len(observed_df),
            'attributes_analyzed': attributes,
            'coverage_gaps': {},
            'recommendations': [],
        }

        # Coverage gaps
        for attr in attributes:
            ref_dist = self.reference.get(attr)
            gaps = self.estimate_coverage_gap(observed_df, attr, ref_dist)
            report['coverage_gaps'][attr] = gaps

            # Check for severe under-representation
            for group, info in gaps.items():
                if info['underrepresented']:
                    report['recommendations'].append(
                        f"Group '{group}' in '{attr}' is underrepresented "
                        f"(coverage ratio: {info['coverage_ratio']:.2f}). "
                        f"Consider targeted data collection or reweighting."
                    )

        # Missing population analysis (if labels available)
        if y_true is not None and y_pred is not None:
            report['missing_populations'] = self.model_missing_populations(
                observed_df, y_true, y_pred, attributes
            )

            # Flag performance concerns
            for attr, groups in report['missing_populations'].items():
                for group, info in groups.items():
                    if info.get('performance_concern'):
                        report['recommendations'].append(
                            f"Performance concern for '{group}' in '{attr}': "
                            f"F1={info['f1']:.3f}, Recall={info['recall']:.3f}. "
                            f"Model may be unreliable for this population."
                        )

        # Print summary
        self._print_report(report)

        return report

    def _print_report(self, report: Dict):
        """Print selection bias report."""
        print(f"\n{'='*60}")
        print("SELECTION BIAS REPORT")
        print(f"{'='*60}")
        print(f"Total observations: {report['n_total']}")

        for attr, gaps in report['coverage_gaps'].items():
            print(f"\n  {attr}:")
            for group, info in sorted(gaps.items()):
                flag = ' ⚠' if info['underrepresented'] else ''
                print(f"    {group:15s}: obs={info['observed_pct']:.1%}, "
                      f"ref={info['reference_pct']:.1%}, "
                      f"ratio={info['coverage_ratio']:.2f}{flag}")

        if report.get('recommendations'):
            print(f"\n  Recommendations:")
            for rec in report['recommendations']:
                print(f"    - {rec}")

        print(f"{'='*60}\n")