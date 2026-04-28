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
import json
import warnings
from pathlib import Path

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
    n_children: int = 1
    household_type: str = 'nuclear'  # 'nuclear', 'single_parent', 'multi_generational', 'shared_custody'


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
        
        # Try loading empirical baseline rates
        self._empirical_rates = None
        self._load_empirical_baseline_rates()
    
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
                'language': 0.05,   # NSCH 2022
                'motor': 0.03,      # NSCH 2022
                'asd': 0.027,       # CDC 2023: 1 in 36
                'adhd': 0.097,      # NSCH 2022
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

            # Household structure (ACS-based)
            n_children = np.random.choice([1, 2, 3], p=[0.40, 0.45, 0.15])
            household_rand = np.random.random()
            if household_rand < 0.23:
                household_type = 'single_parent'
            elif household_rand < 0.27:
                household_type = 'multi_generational'
            elif household_rand < 0.32:
                household_type = 'shared_custody'
            else:
                household_type = 'nuclear'
            
            family = FamilyProfile(
                family_id=f"family_{i:06d}",
                child_age_months=child_age,
                has_delay=has_delay,
                delay_type=delay_type,
                delay_onset_month=delay_onset,
                income_quintile=income_quintile,
                geography=geography,
                ethnicity=ethnicity,
                n_children=n_children,
                household_type=household_type,
            )
            families.append(family)
        
        return families
    
    def _load_empirical_baseline_rates(self):
        """Try to load empirical baseline rates from real data sources.
        
        Priority order:
        1. Instacart rates (most granular transaction data)
        2. Tesco rates (UK grocery data for cross-validation)
        3. Consumer Expenditure Survey rates (US spending data)
        4. Fall back to hardcoded rates with warning
        """
        # Try Instacart first (best source for transaction patterns)
        instacart_path = Path('data/processed/instacart_rates.json')
        if instacart_path.exists():
            with open(instacart_path) as f:
                data = json.load(f)
                self._empirical_rates = data.get('domain_rates_for_synthetic', {})
                self._empirical_source = 'instacart'
                print("Loaded empirical rates from Instacart data")
                return True

        # Try Tesco as backup
        tesco_path = Path('data/processed/tesco_rates.json')
        if tesco_path.exists():
            with open(tesco_path) as f:
                data = json.load(f)
                self._empirical_rates = data.get('domain_rates_for_synthetic', {})
                self._empirical_source = 'tesco'
                print("Loaded empirical rates from Tesco data")
                return True

        # Fall back to CE data
        ce_path = Path('data/processed/ce_baseline_rates.json')
        if ce_path.exists():
            with open(ce_path) as f:
                self._empirical_rates = json.load(f)
                self._empirical_source = 'ce'
                print("Loaded empirical rates from Consumer Expenditure Survey")
                return True

        self._empirical_source = None
        return False

    def _load_empirical_multipliers(self):
        """Try to load empirical delay multipliers from real data.
        
        Uses Kaggle ASD data and PSID-CDS correlations to derive
        multipliers instead of hardcoded values.
        """
        # Try PSID correlation results
        psid_path = Path('data/processed/psid_correlation_results.json')
        if psid_path.exists():
            with open(psid_path) as f:
                self._psid_correlations = json.load(f)
                return True

        # Try Kaggle ASD question importance
        asd_path = Path('data/processed/kaggle_asd_processed.json')
        if asd_path.exists():
            with open(asd_path) as f:
                data = json.load(f)
                self._asd_question_importance = data.get('question_importance', {})
                return True

        return False

    def use_real_data_rates(self, instacart_path: str = None, 
                            tesco_path: str = None,
                            nsch_path: str = None) -> bool:
        """Explicitly load real data rates to eliminate synthetic circularity.
        
        Args:
            instacart_path: Path to Instacart rates JSON
            tesco_path: Path to Tesco rates JSON
            nsch_path: Path to NSCH population stats JSON
            
        Returns:
            True if at least one source loaded successfully
        """
        loaded = False

        if instacart_path and Path(instacart_path).exists():
            with open(instacart_path) as f:
                data = json.load(f)
                self._instacart_rates = data
                self._empirical_rates = data.get('domain_rates_for_synthetic', {})
                self._empirical_source = 'instacart'
                loaded = True
                print(f"Loaded Instacart rates from {instacart_path}")

        if tesco_path and Path(tesco_path).exists():
            with open(tesco_path) as f:
                data = json.load(f)
                self._tesco_rates = data
                if not loaded:
                    self._empirical_rates = data.get('domain_rates_for_synthetic', {})
                    self._empirical_source = 'tesco'
                loaded = True
                print(f"Loaded Tesco rates from {tesco_path}")

        if nsch_path and Path(nsch_path).exists():
            with open(nsch_path) as f:
                self._nsch_stats = json.load(f)
                loaded = True
                print(f"Loaded NSCH population stats from {nsch_path}")

        return loaded

    @property
    def multiplier_noise_sigma(self) -> float:
        """Get the noise sigma for delay multipliers."""
        return getattr(self, '_multiplier_noise_sigma', 0.3)

    @multiplier_noise_sigma.setter
    def multiplier_noise_sigma(self, value: float):
        """Set the noise sigma for delay multipliers."""
        self._multiplier_noise_sigma = value

    def _get_baseline_purchase_rate(self, child_age_months: int, domain: str) -> float:
        """Get baseline monthly purchase rate for a domain given child age.
        
        Tries to use empirical rates from Consumer Expenditure Survey data
        (loaded from data/processed/ce_baseline_rates.json). Falls back to
        hardcoded rates with a warning if empirical data is not available.
        
        Args:
            child_age_months: Child age in months
            domain: Developmental domain
            
        Returns:
            Expected number of purchases per month
        """
        # Try empirical rates first
        if hasattr(self, '_empirical_rates') and self._empirical_rates:
            domain_rates = self._empirical_rates.get(domain)
            if domain_rates:
                # Find closest age key
                age_key = str(int(round(child_age_months / 6.0) * 6))
                age_key = str(min(72, max(0, int(age_key))))
                if age_key in domain_rates:
                    return domain_rates[age_key]

        if not hasattr(self, '_baseline_warning_shown'):
            warnings.warn(
                "WARNING: Using hardcoded baseline rates. "
                "Run scripts/download_public_data.py for empirical rates.",
                stacklevel=2
            )
            self._baseline_warning_shown = True

        # Hardcoded fallback rates
        age_years = child_age_months / 12.0
        
        if domain == 'fine_motor':
            return 0.3 + 0.2 * min(age_years, 3)
        elif domain == 'gross_motor':
            return 0.2 + 0.3 * min(age_years, 4)
        elif domain == 'language':
            return 0.4 + 0.3 * min(age_years, 5)
        elif domain == 'social_emotional':
            return 0.2 + 0.2 * min(age_years, 4)
        elif domain in ['sensory', 'adaptive', 'sleep', 'feeding']:
            return 0.1
        elif domain == 'behavioral':
            return 0.1 + 0.2 * max(0, age_years - 2)
        elif domain == 'therapeutic':
            return 0.05
        else:
            return 0.2
    
    def _get_delay_purchase_multiplier(self, delay_type: str, domain: str, months_since_onset: int) -> float:
        """Get purchase rate multiplier for a delay type and domain.
        
        Tries to use empirical multipliers from PSID-CDS or Kaggle ASD data.
        Falls back to hardcoded values with noise if empirical data unavailable.
        
        Args:
            delay_type: Type of developmental delay
            domain: Developmental domain
            months_since_onset: Months since delay onset
            
        Returns:
            Multiplier for purchase rate
        """
        # Gradual increase after onset
        ramp_factor = min(months_since_onset / 6.0, 1.0)
        
        # Try to get empirical multiplier from PSID correlations
        if hasattr(self, '_psid_correlations') and self._psid_correlations:
            corr_key = f"{delay_type}_{domain}"
            if corr_key in self._psid_correlations:
                corr = self._psid_correlations[corr_key]
                # Convert correlation to multiplier (rough approximation)
                # Higher correlation = higher multiplier
                base_multiplier = 1.0 + abs(corr.get('correlation', 0)) * 3.0
                noisy_multiplier = base_multiplier * np.random.normal(1.0, self.multiplier_noise_sigma)
                noisy_multiplier = np.clip(noisy_multiplier, 0.5, base_multiplier * 1.5)
                return 1.0 + (noisy_multiplier - 1.0) * ramp_factor

        # Try Kaggle ASD question importance for ASD-specific domains
        if delay_type == 'asd' and hasattr(self, '_asd_question_importance'):
            # Map domains to AQ-10 questions
            domain_to_question = {
                'social_emotional': 'q1',
                'language': 'q4',
                'sensory': 'q5',
                'behavioral': 'q7',
            }
            q = domain_to_question.get(domain)
            if q and q in self._asd_question_importance:
                effect = abs(self._asd_question_importance[q].get('effect_size', 0))
                # Convert effect size to multiplier
                base_multiplier = 1.0 + effect * 2.0
                noisy_multiplier = base_multiplier * np.random.normal(1.0, self.multiplier_noise_sigma)
                noisy_multiplier = np.clip(noisy_multiplier, 0.5, base_multiplier * 1.5)
                return 1.0 + (noisy_multiplier - 1.0) * ramp_factor

        # Hardcoded fallback multipliers
        # WARNING: These encode the signal the model will detect (circularity)
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
        # Add Gaussian noise to reduce circularity
        noisy_multiplier = base_multiplier * np.random.normal(1.0, self.multiplier_noise_sigma)
        noisy_multiplier = np.clip(noisy_multiplier, 0.5, base_multiplier * 1.5)
        return 1.0 + (noisy_multiplier - 1.0) * ramp_factor
    
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
        
        # Inject realistic confounds by default
        df = self.generate_confounded_transactions(family, df)
        
        return df
    
    def generate_confounded_transactions(
        self,
        family: FamilyProfile,
        transactions: pd.DataFrame
    ) -> pd.DataFrame:
        """Inject realistic confounds into transaction data.
        
        Adds noise that would be present in real retail data:
        - Sibling purchases (toys bought for older sibling flag younger child)
        - Gift purchases (random products for other children)
        - Temporal misalignment (purchase date != use date)
        - Multi-shopper noise (purchases from different shopper in household)
        
        Args:
            family: Family profile
            transactions: Clean transaction DataFrame
            
        Returns:
            Transaction DataFrame with confounds injected
        """
        if len(transactions) == 0:
            return transactions
        
        confounded = transactions.copy()
        
        # 1. Sibling purchases: if n_children > 1, add purchases for older siblings
        #    that may look like developmental products but aren't for the target child
        if family.n_children > 1:
            n_sibling_purchases = np.random.poisson(family.n_children * 2)
            sibling_rows = []
            for _ in range(n_sibling_purchases):
                # Random domain — siblings may buy sensory/language items regardless
                domain = random.choice(self.developmap.get_domain_names())
                product = random.choice(self.product_catalog[domain])
                # Pick a random existing date
                idx = np.random.randint(len(confounded))
                row = {
                    'family_id': family.family_id,
                    'transaction_date': confounded.iloc[idx]['transaction_date'],
                    'product_id': product['product_id'],
                    'product_name': product['name'],
                    'domain': domain,
                    'price': product['price'] * np.random.uniform(0.8, 1.2),
                    'child_age_months': confounded.iloc[idx]['child_age_months'],
                }
                sibling_rows.append(row)
            if sibling_rows:
                confounded = pd.concat(
                    [confounded, pd.DataFrame(sibling_rows)], ignore_index=True
                )
        
        # 2. Gift purchases: ~5% of transactions are gifts (random domain)
        gift_mask = np.random.random(len(confounded)) < 0.05
        for idx in np.where(gift_mask)[0]:
            random_domain = random.choice(self.developmap.get_domain_names())
            product = random.choice(self.product_catalog[random_domain])
            confounded.at[idx, 'domain'] = random_domain
            confounded.at[idx, 'product_id'] = product['product_id']
            confounded.at[idx, 'product_name'] = product['name']
        
        # 3. Temporal misalignment: shift ~20% of purchase dates by 1-8 weeks
        shift_mask = np.random.random(len(confounded)) < 0.20
        for idx in np.where(shift_mask)[0]:
            lag_days = np.random.randint(7, 57)  # 1-8 weeks
            confounded.at[idx, 'transaction_date'] = (
                confounded.at[idx, 'transaction_date'] - timedelta(days=lag_days)
            )
        
        # 4. Multi-shopper noise: 10% of purchases from a different shopper
        #    (simulated by adding small random noise to purchase patterns)
        shopper_mask = np.random.random(len(confounded)) < 0.10
        for idx in np.where(shopper_mask)[0]:
            # Different shopper might buy slightly different products
            random_domain = random.choice(self.developmap.get_domain_names())
            product = random.choice(self.product_catalog[random_domain])
            confounded.at[idx, 'domain'] = random_domain
            confounded.at[idx, 'product_id'] = product['product_id']
            confounded.at[idx, 'product_name'] = product['name']
        
        # Re-sort by date
        confounded = confounded.sort_values('transaction_date').reset_index(drop=True)
        return confounded
    
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
                'ethnicity': f.ethnicity,
                'n_children': f.n_children,
                'household_type': f.household_type,
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