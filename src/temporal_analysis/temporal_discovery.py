"""Temporal association discovery using Granger-style precedence tests.

IMPORTANT: Granger tests establish temporal precedence, NOT causation.
Results should be interpreted as 'X temporally precedes Y' rather than
'X causes Y'. See Granger (1969) and subsequent critiques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
import warnings
warnings.filterwarnings('ignore')


class TemporalAssociationAnalyzer:
    """Temporal association analysis using Granger-style precedence tests.
    
    NOTE: Despite using Granger 'causality' tests internally, this class
    identifies temporal associations (X precedes Y), not true causal
    relationships. Confounding, reverse causation, and selection bias
    cannot be ruled out without experimental or quasi-experimental designs.
    """
    
    def __init__(self, max_lags: int = 3, significance_level: float = 0.05):
        """Initialize analyzer.
        
        Args:
            max_lags: Maximum number of lags to test
            significance_level: Significance level for tests
        """
        self.max_lags = max_lags
        self.significance_level = significance_level
    
    def test_granger_causality(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """Test if X Granger-causes Y.
        
        Args:
            x: Time series X
            y: Time series Y
            
        Returns:
            Dictionary with test results
        """
        # Prepare data
        data = pd.DataFrame({'y': y, 'x': x})
        
        try:
            # Run Granger causality test
            results = grangercausalitytests(data[['y', 'x']], maxlag=self.max_lags, verbose=False)
            
            # Extract p-values for each lag
            p_values = {}
            for lag in range(1, self.max_lags + 1):
                # Use F-test p-value
                p_value = results[lag][0]['ssr_ftest'][1]
                p_values[lag] = p_value
            
            # Determine if significant
            min_p_value = min(p_values.values())
            is_significant = min_p_value < self.significance_level
            
            return {
                'p_values': p_values,
                'min_p_value': min_p_value,
                'is_significant': is_significant,
                'optimal_lag': min(p_values, key=p_values.get)
            }
        
        except Exception as e:
            return {
                'p_values': {},
                'min_p_value': 1.0,
                'is_significant': False,
                'error': str(e)
            }
    
    def analyze_domain_causality(self, transactions: pd.DataFrame, 
                                domains: List[str]) -> pd.DataFrame:
        """Analyze causality between domains.
        
        Args:
            transactions: Transaction data with domain columns
            domains: List of domain names
            
        Returns:
            DataFrame with causality results
        """
        results = []
        
        for i, domain_x in enumerate(domains):
            for j, domain_y in enumerate(domains):
                if i == j:
                    continue
                
                # Extract time series
                x = transactions[domain_x].values
                y = transactions[domain_y].values
                
                # Test causality
                result = self.test_granger_causality(x, y)
                
                results.append({
                    'cause': domain_x,
                    'effect': domain_y,
                    'p_value': result['min_p_value'],
                    'is_significant': result['is_significant'],
                    'optimal_lag': result.get('optimal_lag', None)
                })
        
        return pd.DataFrame(results)


class TemporalCausalDiscovery:
    """Discover temporal causal relationships."""
    
    def __init__(self, max_lags: int = 3):
        self.max_lags = max_lags
        self.granger_analyzer = TemporalAssociationAnalyzer(max_lags=max_lags)
    
    def discover_causal_graph(self, data: pd.DataFrame, 
                             variables: List[str]) -> Dict[str, List[str]]:
        """Discover causal graph structure.
        
        Args:
            data: Time series data
            variables: List of variable names
            
        Returns:
            Dictionary mapping causes to effects
        """
        causal_graph = {var: [] for var in variables}
        
        for cause in variables:
            for effect in variables:
                if cause == effect:
                    continue
                
                result = self.granger_analyzer.test_granger_causality(
                    data[cause].values,
                    data[effect].values
                )
                
                if result['is_significant']:
                    causal_graph[cause].append(effect)
        
        return causal_graph
    
    def analyze_intervention_effect(self, data: pd.DataFrame, 
                                   intervention_var: str,
                                   outcome_var: str,
                                   intervention_time: int) -> Dict:
        """Analyze effect of intervention.
        
        Args:
            data: Time series data
            intervention_var: Variable that was intervened on
            outcome_var: Outcome variable
            intervention_time: Time point of intervention
            
        Returns:
            Dictionary with intervention analysis
        """
        # Split data
        pre_intervention = data[:intervention_time]
        post_intervention = data[intervention_time:]
        
        # Compute means
        pre_mean = pre_intervention[outcome_var].mean()
        post_mean = post_intervention[outcome_var].mean()
        
        # Effect size
        effect = post_mean - pre_mean
        
        return {
            'pre_mean': pre_mean,
            'post_mean': post_mean,
            'effect': effect,
            'relative_change': effect / pre_mean if pre_mean != 0 else 0
        }