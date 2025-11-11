"""Synthetic Data Generator for RetailHealth.

Generates scientifically grounded synthetic retail transaction data
with embedded developmental delay signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import random

from ..taxonomy.developmap import DEVELOPMAP


@dataclass
class FamilyProfile:
    """Profile of a family for simulation."""
    family_id: str
    child_age_months: int
    has_delay: bool
    delay_type: Optional[str]
    delay_onset_month: Optional[int]
    income_quintile: int  # 1-5
    geography: str  # urban, suburban, rural
    ethnicity: str


class SyntheticDataGenerator:
    """Generates synthetic retail transaction data with developmental signals."""
    
    def __init__(self, seed: int = 42):
        """Initialize the generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self.developmap = DEVELOPMAP
        
        # Product catalog by domain
        self._initialize_product_catalog()
    
    def _initialize_product_catalog(self):
        """Create a synthetic product catalog aligned with DevelopMap."""
        self.product_catalog = {}
        
        for domain_name, domain in self.developmap.domains.items():
            products = []
            for i, example in enumerate(domain.example_products):
                products.append({
                    'product_id': f"{domain_name}_{i:03d}",
                    'name': example,
                    'domain': domain_name,
                    'price': np.random.uniform(10, 100)
                })
            
            # Add more generic products
            for i in range(len(domain.example_products), 20):
                keywords = list(domain.keywords)
                name = f"{random.choice(keywords).title()} Product {i}"
                products.append({
                    'product_id': f"{domain_name}_{i:03d}",
                    'name': name,
                    'domain': domain_name,
                    'price': np.random.uniform(5, 150)
                })
            
            self.product_catalog[domain_name] = products
    
    def generate_families(
        self,
        n_families: int,
        age_range: Tuple[int, int] = (0, 72),  # 0-6 years in months
        delay_prevalence: float = 0.15,
        delay_types: Dict[str, float] = None
    ) -> List[FamilyProfile]:
        """Generate family profiles.
        
        Args:
            n_families: Number of families to generate
            age_range: Child age range in months (min, max)
            delay_prevalence: Overall delay prevalence
            delay_types: Dictionary of delay type prevalences
            
        Returns:
            List of family profiles
        """
        if delay_types is None:
            delay_types = {
                'language': 0.06,
                'motor': 0.02,
                'asd': 0.015,
                'adhd': 0.04
            }
        
        families = []
        
        for i in range(n_families):
            # Sample child age
            child_age = np.random.randint(age_range[0], age_range[1] + 1)
            
            # Determine if child has delay
            has_delay = np.random.random() < delay_prevalence
            
            if has_delay:
                # Sample delay type
                delay_type = np.random.choice(
                    list(delay_types.keys()),
                    p=np.array(list(delay_types.values())) / sum(delay_types.values())
                )
                # Delay onset: 6-18 months before current age
                delay_onset = max(0, child_age - np.random.randint(6, 19))
            else:
                delay_type = None
                delay_onset = None
            
            # Demographics
            income_quintile = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.20, 0.30, 0.20, 0.15])
            geography = np.random.choice(['urban', 'suburban', 'rural'], p=[0.35, 0.45, 0.20])
            ethnicity = np.random.choice(
                ['white', 'hispanic', 'black', 'asian', 'other'],
                p=[0.60, 0.18, 0.13, 0.06, 0.03]
            )
            
            family = FamilyProfile(
                family_id=f"family_{i:06d}",
                child_age_months=child_age,
                has_delay=has_delay,
                delay_type=delay_type,
                delay_onset_month=delay_onset,
                income_quintile=income_quintile,
                geography=geography,
                ethnicity=ethnicity
            )
            families.append(family)
        
        return families
    
    def _get_baseline_purchase_rate(self, child_age_months: int, domain: str) -> float:
        """Get baseline monthly purchase rate for a domain given child age.
        
        Args:
            child_age_months: Child age in months
            domain: Developmental domain
            
        Returns:
            Expected number of purchases per month
        """
        # Age-appropriate baseline rates
        age_years = child_age_months / 12.0
        
        # Different domains have different age profiles
        if domain == 'fine_motor':
            return 0.3 + 0.2 * min(age_years, 3)
        elif domain == 'gross_motor':
            return 0.2 + 0.3 * min(age_years, 4)
        elif domain == 'language':
            return 0.4 + 0.3 * min(age_years, 5)
        elif domain == 'social_emotional':
            return 0.2 + 0.2 * min(age_years, 4)
        elif domain in ['sensory', 'adaptive', 'sleep', 'feeding']:
            return 0.1  # Lower baseline
        elif domain == 'behavioral':
            return 0.1 + 0.2 * max(0, age_years - 2)
        elif domain == 'therapeutic':
            return 0.05  # Very low baseline
        else:
            return 0.2
    
    def _get_delay_purchase_multiplier(self, delay_type: str, domain: str, months_since_onset: int) -> float:
        """Get purchase rate multiplier for a delay type and domain.
        
        Args:
            delay_type: Type of developmental delay
            domain: Developmental domain
            months_since_onset: Months since delay onset
            
        Returns:
            Multiplier for purchase rate
        """
        # Gradual increase after onset
        ramp_factor = min(months_since_onset / 6.0, 1.0)
        
        # Domain-specific multipliers for each delay type
        multipliers = {
            'language': {
                'language': 3.0,
                'social_emotional': 1.5,
                'therapeutic': 2.0
            },
            'motor': {
                'fine_motor': 2.5,
                'gross_motor': 2.5,
                'adaptive': 2.0,
                'therapeutic': 1.8
            },
            'asd': {
                'sensory': 4.0,
                'sleep': 3.0,
                'feeding': 2.5,
                'language': 2.0,
                'social_emotional': 2.5,
                'behavioral': 2.0,
                'therapeutic': 2.5
            },
            'adhd': {
                'behavioral': 3.5,
                'attention': 3.0,
                'sensory': 1.5,
                'therapeutic': 2.0
            }
        }
        
        base_multiplier = multipliers.get(delay_type, {}).get(domain, 1.0)
        return 1.0 + (base_multiplier - 1.0) * ramp_factor
    
    def generate_transactions(
        self,
        family: FamilyProfile,
        months_history: int = 24,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """Generate transaction history for a family.
        
        Args:
            family: Family profile
            months_history: Number of months of history to generate
            end_date: End date for transaction history
            
        Returns:
            DataFrame of transactions
        """
        if end_date is None:
            end_date = datetime.now()
        
        transactions = []
        
        # Generate monthly transactions
        for month_offset in range(months_history):
            transaction_date = end_date - timedelta(days=30 * month_offset)
            child_age_at_month = max(0, family.child_age_months - month_offset)
            
            # For each domain, generate purchases
            for domain_name in self.developmap.get_domain_names():
                # Base purchase rate
                base_rate = self._get_baseline_purchase_rate(child_age_at_month, domain_name)
                
                # Apply delay multiplier if applicable
                if family.has_delay and family.delay_onset_month is not None:
                    months_since_onset = child_age_at_month - family.delay_onset_month
                    if months_since_onset >= 0:
                        multiplier = self._get_delay_purchase_multiplier(
                            family.delay_type, domain_name, months_since_onset
                        )
                        base_rate *= multiplier
                
                # Income effect (higher income = more purchases)
                income_factor = 0.5 + 0.2 * family.income_quintile
                base_rate *= income_factor
                
                # Sample number of purchases (Poisson)
                n_purchases = np.random.poisson(base_rate)
                
                # Generate individual purchases
                for _ in range(n_purchases):
                    product = random.choice(self.product_catalog[domain_name])
                    
                    transactions.append({
                        'family_id': family.family_id,
                        'transaction_date': transaction_date,
                        'product_id': product['product_id'],
                        'product_name': product['name'],
                        'domain': domain_name,
                        'price': product['price'] * np.random.uniform(0.8, 1.2),  # Price variation
                        'child_age_months': child_age_at_month
                    })
        
        df = pd.DataFrame(transactions)
        if len(df) > 0:
            df = df.sort_values('transaction_date').reset_index(drop=True)
        
        return df
    
    def generate_dataset(
        self,
        n_families: int = 100000,
        months_history: int = 24,
        **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate complete dataset with families and transactions.
        
        Args:
            n_families: Number of families
            months_history: Months of transaction history
            **kwargs: Additional arguments for generate_families
            
        Returns:
            Tuple of (families_df, transactions_df)
        """
        print(f"Generating {n_families} family profiles...")
        families = self.generate_families(n_families, **kwargs)
        
        # Convert to DataFrame
        families_df = pd.DataFrame([
            {
                'family_id': f.family_id,
                'child_age_months': f.child_age_months,
                'has_delay': f.has_delay,
                'delay_type': f.delay_type,
                'delay_onset_month': f.delay_onset_month,
                'income_quintile': f.income_quintile,
                'geography': f.geography,
                'ethnicity': f.ethnicity
            }
            for f in families
        ])
        
        print(f"Generating transaction histories...")
        all_transactions = []
        
        for i, family in enumerate(families):
            if (i + 1) % 10000 == 0:
                print(f"  Generated transactions for {i + 1}/{n_families} families")
            
            transactions = self.generate_transactions(family, months_history)
            all_transactions.append(transactions)
        
        transactions_df = pd.concat(all_transactions, ignore_index=True)
        
        print(f"Generated {len(transactions_df)} total transactions")
        print(f"Delay prevalence: {families_df['has_delay'].mean():.2%}")
        print(f"Delay types: {families_df[families_df['has_delay']]['delay_type'].value_counts().to_dict()}")
        
        return families_df, transactions_df