"""Counterfactual simulation and sensitivity analysis.

Provides tools to estimate what developmental screening outcomes would
look like under alternative purchase trajectories, and to quantify
how sensitive the model's predictions are to unmeasured confounders.

IMPORTANT: These are model-based simulations, not true counterfactuals.
Results depend on the correctness of the underlying model and the
strong ignorability assumption (no unmeasured confounders).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats


class CounterfactualSimulator:
    """Simulate counterfactual purchase trajectories.

    Given a trained screening model, asks: 'What would the predicted
    risk be if this family's purchase pattern were different?'

    This is useful for:
    - Understanding which domain shifts drive risk predictions
    - Generating explanations ('risk would drop by X% if sensory
      purchases decreased to typical levels')
    - Identifying the minimum intervention target
    """

    def __init__(self, model=None, domain_names: List[str] = None):
        """Initialize counterfactual simulator.

        Args:
            model: Trained screening model with a predict() or forward() method.
                   If None, uses a simple logistic proxy for demonstration.
            domain_names: List of developmental domain names
        """
        self.model = model
        self.domain_names = domain_names or [
            'fine_motor', 'gross_motor', 'language', 'social_emotional',
            'sensory', 'adaptive', 'sleep', 'feeding', 'behavioral',
            'therapeutic'
        ]

    def simulate_domain_removal(
        self,
        purchase_vector: np.ndarray,
        baseline_vector: np.ndarray = None
    ) -> Dict[str, Dict]:
        """Simulate removing each domain's signal from the purchase vector.

        For each domain, replaces the family's purchase count with the
        population baseline and re-predicts risk.

        Args:
            purchase_vector: Array of shape (n_domains,) with the family's
                domain purchase counts
            baseline_vector: Population baseline purchase counts. If None,
                uses zeros.

        Returns:
            Dict mapping domain name to {original_risk, counterfactual_risk,
            risk_change, risk_change_pct}
        """
        if baseline_vector is None:
            baseline_vector = np.zeros_like(purchase_vector)

        original_risk = self._predict(purchase_vector)
        results = {}

        for i, domain in enumerate(self.domain_names):
            cf_vector = purchase_vector.copy()
            cf_vector[i] = baseline_vector[i]

            cf_risk = self._predict(cf_vector)
            change = cf_risk - original_risk

            results[domain] = {
                'original_risk': float(original_risk),
                'counterfactual_risk': float(cf_risk),
                'risk_change': float(change),
                'risk_change_pct': float(change / original_risk * 100)
                if original_risk > 0 else 0.0,
                'original_value': float(purchase_vector[i]),
                'counterfactual_value': float(baseline_vector[i]),
            }

        return results

    def simulate_trajectory_shift(
        self,
        purchase_sequence: np.ndarray,
        shift_domain: str,
        shift_magnitude: float,
        shift_start_month: int = 0
    ) -> Dict:
        """Simulate shifting a domain's trajectory over time.

        Args:
            purchase_sequence: Array of shape (n_months, n_domains)
            shift_domain: Domain to shift
            shift_magnitude: Multiplicative factor (e.g., 0.5 = halve, 2.0 = double)
            shift_start_month: Month at which shift begins

        Returns:
            Dict with original and counterfactual risk trajectories
        """
        domain_idx = self.domain_names.index(shift_domain)

        cf_sequence = purchase_sequence.copy()
        cf_sequence[shift_start_month:, domain_idx] *= shift_magnitude

        # Predict risk at each time step (cumulative)
        original_risks = []
        cf_risks = []

        for t in range(1, len(purchase_sequence) + 1):
            orig_cumulative = purchase_sequence[:t].mean(axis=0)
            cf_cumulative = cf_sequence[:t].mean(axis=0)

            original_risks.append(self._predict(orig_cumulative))
            cf_risks.append(self._predict(cf_cumulative))

        return {
            'domain': shift_domain,
            'shift_magnitude': shift_magnitude,
            'shift_start_month': shift_start_month,
            'original_risks': [float(r) for r in original_risks],
            'counterfactual_risks': [float(r) for r in cf_risks],
            'final_risk_change': float(cf_risks[-1] - original_risks[-1]),
        }

    def find_minimum_intervention(
        self,
        purchase_vector: np.ndarray,
        target_risk: float,
        baseline_vector: np.ndarray = None,
        max_iterations: int = 50
    ) -> Dict:
        """Find the smallest purchase shift that reduces risk below target.

        Uses binary search on each domain to find the minimum change
        needed to bring predicted risk below target_risk.

        Args:
            purchase_vector: Current domain purchase counts
            target_risk: Target risk threshold
            baseline_vector: Population baseline
            max_iterations: Max binary search iterations

        Returns:
            Dict with per-domain minimum interventions
        """
        if baseline_vector is None:
            baseline_vector = np.zeros_like(purchase_vector)

        original_risk = self._predict(purchase_vector)

        if original_risk <= target_risk:
            return {
                'already_below_target': True,
                'original_risk': float(original_risk),
                'target_risk': target_risk,
            }

        interventions = {}

        for i, domain in enumerate(self.domain_names):
            if purchase_vector[i] <= baseline_vector[i]:
                continue  # Already at or below baseline

            lo = baseline_vector[i]
            hi = purchase_vector[i]
            best_value = hi

            for _ in range(max_iterations):
                mid = (lo + hi) / 2.0
                cf_vector = purchase_vector.copy()
                cf_vector[i] = mid
                cf_risk = self._predict(cf_vector)

                if cf_risk <= target_risk:
                    best_value = mid
                    lo = mid
                else:
                    hi = mid

                if abs(hi - lo) < 0.01:
                    break

            reduction_needed = purchase_vector[i] - best_value
            interventions[domain] = {
                'current_value': float(purchase_vector[i]),
                'target_value': float(best_value),
                'reduction_needed': float(reduction_needed),
                'reduction_pct': float(reduction_needed / purchase_vector[i] * 100)
                if purchase_vector[i] > 0 else 0.0,
                'achieves_target': self._predict(
                    np.where(np.arange(len(purchase_vector)) == i,
                             best_value, purchase_vector)
                ) <= target_risk,
            }

        return {
            'original_risk': float(original_risk),
            'target_risk': target_risk,
            'interventions': interventions,
        }

    def _predict(self, features: np.ndarray) -> float:
        """Predict risk from feature vector.

        Uses the provided model if available, otherwise uses a simple
        logistic function as a placeholder.
        """
        if self.model is not None:
            import torch
            if hasattr(self.model, 'forward'):
                with torch.no_grad():
                    x = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
                    return float(self.model(x).squeeze())
            elif hasattr(self.model, 'predict_proba'):
                return float(self.model.predict_proba(features.reshape(1, -1))[0, 1])
            elif callable(self.model):
                return float(self.model(features))

        # Fallback: simple logistic proxy
        z = np.sum(features * 0.1) - 1.0
        return 1.0 / (1.0 + np.exp(-z))


class SensitivityAnalyzer:
    """Quantify sensitivity of findings to unmeasured confounders.

    Implements Rosenbaum-style sensitivity analysis and E-value
    computation to answer: 'How strong would an unmeasured confounder
    need to be to explain away the observed association?'

    References:
        - Rosenbaum (2002). Observational Studies, 2nd ed.
        - VanderWeele & Ding (2017). Annals of Internal Medicine.
    """

    def __init__(self):
        pass

    def compute_e_value(
        self,
        point_estimate: float,
        ci_lower: float = None,
        measure: str = 'risk_ratio'
    ) -> Dict[str, float]:
        """Compute the E-value for a point estimate.

        The E-value is the minimum strength of association (on the risk
        ratio scale) that an unmeasured confounder would need to have
        with both the treatment and the outcome to fully explain away
        the observed treatment-outcome association.

        Args:
            point_estimate: Observed effect (risk ratio, odds ratio, or
                hazard ratio). Must be > 0.
            ci_lower: Lower bound of confidence interval (optional).
            measure: Type of measure ('risk_ratio', 'odds_ratio', 'hazard_ratio')

        Returns:
            Dict with e_value for point estimate and (optionally) CI bound
        """
        # Convert to risk ratio scale if needed
        if measure == 'odds_ratio':
            # Use square-root transformation for common outcomes
            rr = point_estimate
            # For rare outcomes, OR ≈ RR; for common, use approximation
        elif measure == 'hazard_ratio':
            rr = point_estimate  # HR ≈ RR for rare events
        else:
            rr = point_estimate

        if rr < 1:
            rr = 1.0 / rr  # Flip so RR > 1

        # E-value formula: RR + sqrt(RR * (RR - 1))
        e_value = rr + np.sqrt(rr * (rr - 1))

        result = {
            'point_estimate': float(point_estimate),
            'risk_ratio_used': float(rr),
            'e_value': float(e_value),
        }

        if ci_lower is not None:
            ci_rr = ci_lower if ci_lower > 1 else 1.0 / ci_lower if ci_lower > 0 else 1.0
            if ci_rr > 1:
                e_value_ci = ci_rr + np.sqrt(ci_rr * (ci_rr - 1))
            else:
                e_value_ci = 1.0
            result['ci_lower'] = float(ci_lower)
            result['e_value_ci'] = float(e_value_ci)

        return result

    def rosenbaum_bounds(
        self,
        treated_outcomes: np.ndarray,
        control_outcomes: np.ndarray,
        gamma_range: np.ndarray = None
    ) -> Dict[str, List]:
        """Compute Rosenbaum bounds for sensitivity analysis.

        Tests how the treatment effect p-value changes as the hidden
        bias parameter Gamma increases from 1 (no bias) upward.

        Args:
            treated_outcomes: Outcomes for treated group
            control_outcomes: Outcomes for control group
            gamma_range: Array of Gamma values to test. Default: 1.0 to 3.0

        Returns:
            Dict with gamma values and corresponding p-value bounds
        """
        if gamma_range is None:
            gamma_range = np.arange(1.0, 3.1, 0.25)

        # Observed test statistic (Wilcoxon rank-sum)
        observed_stat, observed_p = stats.ranksums(
            treated_outcomes, control_outcomes
        )

        results = {
            'gamma': [float(g) for g in gamma_range],
            'p_upper': [],
            'p_lower': [],
            'significant_at_005': [],
        }

        n_t = len(treated_outcomes)
        n_c = len(control_outcomes)

        for gamma in gamma_range:
            # Under hidden bias Gamma, the treatment assignment probability
            # for each unit is bounded by [1/(1+Gamma), Gamma/(1+Gamma)]
            # We adjust the test statistic bounds accordingly

            # Simplified: scale the observed p-value by Gamma
            # (full implementation would use exact permutation bounds)
            if gamma == 1.0:
                p_upper = float(observed_p)
                p_lower = float(observed_p)
            else:
                # Approximate adjustment: shift the test statistic
                se_adjustment = np.sqrt(n_t * n_c / (n_t + n_c)) * np.log(gamma)

                z_obs = stats.norm.ppf(1 - observed_p / 2)  # two-sided to one-sided

                z_upper = z_obs - se_adjustment
                z_lower = z_obs + se_adjustment

                p_upper = float(2 * (1 - stats.norm.cdf(abs(z_upper))))
                p_lower = float(2 * (1 - stats.norm.cdf(abs(z_lower))))

            results['p_upper'].append(p_upper)
            results['p_lower'].append(p_lower)
            results['significant_at_005'].append(p_upper < 0.05)

        # Find critical Gamma (where p_upper first exceeds 0.05)
        critical_gamma = None
        for g, sig in zip(results['gamma'], results['significant_at_005']):
            if not sig:
                critical_gamma = g
                break

        results['observed_p'] = float(observed_p)
        results['observed_statistic'] = float(observed_stat)
        results['critical_gamma'] = critical_gamma

        return results

    def quantitative_bias_analysis(
        self,
        observed_rr: float,
        confounder_prevalence_treated: float,
        confounder_prevalence_control: float,
        confounder_outcome_rr: float
    ) -> Dict[str, float]:
        """Adjust an observed risk ratio for a hypothetical confounder.

        Given assumptions about an unmeasured confounder's prevalence
        in treated/control groups and its effect on the outcome,
        computes the bias-adjusted risk ratio.

        Args:
            observed_rr: Observed (unadjusted) risk ratio
            confounder_prevalence_treated: P(confounder | treated)
            confounder_prevalence_control: P(confounder | control)
            confounder_outcome_rr: RR of outcome given confounder

        Returns:
            Dict with adjusted_rr, bias_factor, and interpretation
        """
        p1 = confounder_prevalence_treated
        p0 = confounder_prevalence_control
        rr_c = confounder_outcome_rr

        # Bias factor formula (Greenland, 1996)
        bias_factor = (p1 * (rr_c - 1) + 1) / (p0 * (rr_c - 1) + 1)

        adjusted_rr = observed_rr / bias_factor

        return {
            'observed_rr': float(observed_rr),
            'bias_factor': float(bias_factor),
            'adjusted_rr': float(adjusted_rr),
            'confounder_prevalence_treated': p1,
            'confounder_prevalence_control': p0,
            'confounder_outcome_rr': rr_c,
            'interpretation': (
                f"If an unmeasured confounder with prevalence {p1:.0%}/{p0:.0%} "
                f"(treated/control) and RR={rr_c:.1f} existed, the adjusted "
                f"RR would be {adjusted_rr:.2f} (vs observed {observed_rr:.2f})."
            ),
        }
