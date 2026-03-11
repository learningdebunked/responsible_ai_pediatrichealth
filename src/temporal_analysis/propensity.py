"""Propensity score methods for causal inference."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import cdist


class PropensityScoreEstimator:
    """Estimate propensity scores for treatment assignment."""
    
    def __init__(self, method: str = 'logistic'):
        """Initialize estimator.
        
        Args:
            method: Estimation method ('logistic', 'rf')
        """
        self.method = method
        if method == 'logistic':
            self.model = LogisticRegression(max_iter=1000)
        elif method == 'rf':
            self.model = RandomForestClassifier(n_estimators=100)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def fit(self, X: np.ndarray, treatment: np.ndarray) -> 'PropensityScoreEstimator':
        """Fit propensity score model.
        
        Args:
            X: Covariates
            treatment: Treatment indicator
            
        Returns:
            Self
        """
        self.model.fit(X, treatment)
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict propensity scores.
        
        Args:
            X: Covariates
            
        Returns:
            Propensity scores
        """
        return self.model.predict_proba(X)[:, 1]


class PropensityScoreMatching:
    """Propensity score matching for causal inference."""
    
    def __init__(self, caliper: float = 0.1, method: str = 'nearest'):
        """Initialize matcher.
        
        Args:
            caliper: Maximum propensity score difference
            method: Matching method ('nearest', 'optimal')
        """
        self.caliper = caliper
        self.method = method
        self.estimator = PropensityScoreEstimator()
    
    def match(self, X: np.ndarray, treatment: np.ndarray, 
             outcome: np.ndarray) -> Tuple[np.ndarray, float]:
        """Perform propensity score matching.
        
        Args:
            X: Covariates
            treatment: Treatment indicator
            outcome: Outcome variable
            
        Returns:
            Tuple of (matched_indices, ATE)
        """
        # Estimate propensity scores
        self.estimator.fit(X, treatment)
        propensity_scores = self.estimator.predict_proba(X)
        
        # Separate treated and control
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]
        
        treated_ps = propensity_scores[treated_idx]
        control_ps = propensity_scores[control_idx]
        
        # Match
        matched_pairs = []
        
        for i, t_idx in enumerate(treated_idx):
            # Find closest control unit
            distances = np.abs(control_ps - treated_ps[i])
            
            # Apply caliper
            valid_matches = distances < self.caliper
            
            if valid_matches.any():
                # Get closest match
                closest_idx = np.argmin(distances[valid_matches])
                c_idx = control_idx[np.where(valid_matches)[0][closest_idx]]
                matched_pairs.append((t_idx, c_idx))
        
        # Compute ATE
        if len(matched_pairs) > 0:
            treatment_effects = []
            for t_idx, c_idx in matched_pairs:
                effect = outcome[t_idx] - outcome[c_idx]
                treatment_effects.append(effect)
            
            ate = np.mean(treatment_effects)
        else:
            ate = 0.0
        
        return np.array(matched_pairs), ate


class InverseProbabilityWeighting:
    """Inverse probability weighting for causal inference."""
    
    def __init__(self):
        self.estimator = PropensityScoreEstimator()
    
    def estimate_ate(self, X: np.ndarray, treatment: np.ndarray, 
                    outcome: np.ndarray) -> Dict[str, float]:
        """Estimate average treatment effect using IPW.
        
        Args:
            X: Covariates
            treatment: Treatment indicator
            outcome: Outcome variable
            
        Returns:
            Dictionary with ATE and standard error
        """
        # Estimate propensity scores
        self.estimator.fit(X, treatment)
        ps = self.estimator.predict_proba(X)
        
        # Clip propensity scores to avoid extreme weights
        ps = np.clip(ps, 0.01, 0.99)
        
        # Compute IPW weights
        weights = treatment / ps + (1 - treatment) / (1 - ps)
        
        # Weighted outcomes
        weighted_outcomes_treated = (treatment * outcome * weights).sum() / (treatment * weights).sum()
        weighted_outcomes_control = ((1 - treatment) * outcome * weights).sum() / ((1 - treatment) * weights).sum()
        
        ate = weighted_outcomes_treated - weighted_outcomes_control
        
        # Standard error (simplified)
        se = np.std(outcome * weights) / np.sqrt(len(outcome))
        
        return {
            'ate': ate,
            'se': se,
            'ci_lower': ate - 1.96 * se,
            'ci_upper': ate + 1.96 * se
        }


class DoublyRobustEstimator:
    """Doubly robust estimation combining regression and IPW."""
    
    def __init__(self):
        self.ps_estimator = PropensityScoreEstimator()
        self.outcome_model_treated = LogisticRegression()
        self.outcome_model_control = LogisticRegression()
    
    def estimate_ate(self, X: np.ndarray, treatment: np.ndarray, 
                    outcome: np.ndarray) -> float:
        """Estimate ATE using doubly robust method.
        
        Args:
            X: Covariates
            treatment: Treatment indicator
            outcome: Outcome variable
            
        Returns:
            Average treatment effect
        """
        # Estimate propensity scores
        self.ps_estimator.fit(X, treatment)
        ps = np.clip(self.ps_estimator.predict_proba(X), 0.01, 0.99)
        
        # Fit outcome models
        treated_mask = treatment == 1
        control_mask = treatment == 0
        
        if treated_mask.sum() > 0:
            self.outcome_model_treated.fit(X[treated_mask], outcome[treated_mask])
        if control_mask.sum() > 0:
            self.outcome_model_control.fit(X[control_mask], outcome[control_mask])
        
        # Predict potential outcomes
        mu1 = self.outcome_model_treated.predict_proba(X)[:, 1] if hasattr(self.outcome_model_treated, 'classes_') else outcome.mean()
        mu0 = self.outcome_model_control.predict_proba(X)[:, 1] if hasattr(self.outcome_model_control, 'classes_') else outcome.mean()
        
        # Doubly robust estimator
        dr_treated = mu1 + treatment * (outcome - mu1) / ps
        dr_control = mu0 + (1 - treatment) * (outcome - mu0) / (1 - ps)
        
        ate = (dr_treated - dr_control).mean()
        
        return ate