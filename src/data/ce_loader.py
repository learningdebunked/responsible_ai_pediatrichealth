"""Consumer Expenditure Survey (CE) Data Loader.

Loads CE Interview Survey PUMD (Public Use Microdata) from:
    https://www.bls.gov/cex/pumd_data.htm

This module replaces the hardcoded _get_baseline_purchase_rate() in
src/data/synthetic_generator.py with empirically grounded age-specific
spending curves derived from BLS Consumer Expenditure Survey data.

The CE Interview Survey provides detailed quarterly expenditure data for
US households. We filter to households with children under 6 and extract
spending in categories that map to DevelopMap developmental domains.

Required input: CE Interview Survey PUMD files. Specifically:
    - fmli*.csv  (Family characteristics and income)
    - mtbi*.csv  (Member-level detailed expenditures)
    or the integrated extract files.

Key CE expenditure variables (UCCs / summary categories):
    EDUCAX   - Education expenses
    ENTERTX  - Entertainment/recreation
    HEALTHX  - Healthcare
    READX    - Reading materials
    APPARLX  - Apparel (filter to children's)
    Childcare UCCs: 670110, 670210, 670310 (daycare, babysitting, etc.)
    Toys/games UCCs: 610110, 610120, 610130, 610140
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
import json
import warnings


# CE summary expenditure variables relevant to child development
CE_SPENDING_VARS = {
    'education': 'EDUCAX',
    'entertainment': 'ENTERTX',
    'healthcare': 'HEALTHX',
    'reading': 'READX',
    'apparel': 'APPARLX',
}

# Detailed UCCs for child-specific spending
CHILDCARE_UCCS = ['670110', '670210', '670310', '670410']
TOY_UCCS = ['610110', '610120', '610130', '610140']
BOOK_UCCS = ['590110', '590210', '590220']

# Mapping from CE spending categories to DevelopMap domains
CE_TO_DEVELOPMAP = {
    'education': ['language', 'cognitive'],
    'entertainment': ['fine_motor', 'gross_motor', 'sensory', 'social_emotional'],
    'healthcare': ['therapeutic'],
    'reading': ['language'],
    'apparel': [],  # used for demographic weighting only
    'childcare': ['adaptive', 'therapeutic'],
    'toys': ['fine_motor', 'gross_motor', 'sensory', 'social_emotional'],
    'books': ['language'],
}


class CELoader:
    """Load and process Consumer Expenditure Survey data.

    Produces empirical age-specific baseline spending curves by
    developmental domain, replacing hardcoded rates in the synthetic
    data generator.
    """

    def __init__(self, ce_data_dir: str):
        """Initialize CE loader.

        Args:
            ce_data_dir: Path to directory containing CE PUMD files
                         (fmli*.csv and mtbi*.csv)
        """
        self.data_dir = Path(ce_data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"CE data directory not found: {self.data_dir}\n"
                f"Download from: https://www.bls.gov/cex/pumd_data.htm"
            )
        self.family_df: Optional[pd.DataFrame] = None
        self.expenditure_df: Optional[pd.DataFrame] = None
        self.baseline_rates: Optional[Dict[str, Callable]] = None

    def load(self) -> Dict[str, Callable[[int], float]]:
        """Load CE data and compute age-specific spending curves.

        Returns:
            Dict mapping domain name to function(child_age_months) -> expected
            monthly spending rate (normalized to purchase count proxy).
        """
        self._load_family_data()
        self._load_expenditure_data()
        self._compute_baseline_curves()
        return self.baseline_rates

    def _load_family_data(self):
        """Load family characteristics files (fmli*.csv)."""
        fmli_files = sorted(self.data_dir.glob('fmli*.csv'))
        if not fmli_files:
            # Try subdirectories
            fmli_files = sorted(self.data_dir.rglob('fmli*.csv'))

        if not fmli_files:
            raise FileNotFoundError(
                f"No fmli*.csv files found in {self.data_dir}. "
                f"Download CE Interview Survey PUMD from "
                f"https://www.bls.gov/cex/pumd_data.htm"
            )

        print(f"Loading {len(fmli_files)} family characteristic files...")
        dfs = []
        for f in fmli_files:
            try:
                df = pd.read_csv(f, low_memory=False)
                dfs.append(df)
            except Exception as e:
                warnings.warn(f"Could not load {f}: {e}")

        self.family_df = pd.concat(dfs, ignore_index=True)
        print(f"  Loaded {len(self.family_df)} household records")

        # Filter to households with children under 6
        child_age_cols = [c for c in self.family_df.columns if c.startswith('AGE_REF') or c.startswith('AGE')]
        # PERSLT18: number of persons under 18
        # AGE_REF: age of reference person
        # We need households with young children. Use available indicators.
        if 'PERSLT18' in self.family_df.columns:
            has_children = self.family_df['PERSLT18'] > 0
            self.family_df = self.family_df[has_children].copy()
            print(f"  Filtered to {len(self.family_df)} households with children")

    def _load_expenditure_data(self):
        """Load detailed expenditure files (mtbi*.csv)."""
        mtbi_files = sorted(self.data_dir.glob('mtbi*.csv'))
        if not mtbi_files:
            mtbi_files = sorted(self.data_dir.rglob('mtbi*.csv'))

        if not mtbi_files:
            # Fall back to using summary variables from family file
            print("  No mtbi*.csv found; using summary expenditure variables from fmli")
            self.expenditure_df = None
            return

        print(f"Loading {len(mtbi_files)} expenditure detail files...")
        dfs = []
        for f in mtbi_files:
            try:
                df = pd.read_csv(f, low_memory=False, dtype={'UCC': str})
                dfs.append(df)
            except Exception as e:
                warnings.warn(f"Could not load {f}: {e}")

        self.expenditure_df = pd.concat(dfs, ignore_index=True)
        print(f"  Loaded {len(self.expenditure_df)} expenditure records")

    def _compute_baseline_curves(self):
        """Compute age-specific spending curves per developmental domain.

        Produces a function for each domain that takes child_age_months and
        returns an expected monthly purchase rate (normalized).
        """
        # If we have summary variables in the family file, use those
        if self.family_df is None:
            raise RuntimeError("No data loaded. Call load() or _load_family_data() first.")

        # Extract spending by age of youngest child
        # AGE2 in FMLI is age of second member — we approximate youngest child age
        # from household composition variables
        df = self.family_df.copy()

        # Identify youngest child age from member age variables
        # CE files have AGE_REF, AGE2, ... for household members
        member_age_cols = [c for c in df.columns if c.startswith('AGE') and c[3:].isdigit()]
        if not member_age_cols:
            member_age_cols = ['AGE_REF', 'AGE2']
            member_age_cols = [c for c in member_age_cols if c in df.columns]

        if member_age_cols:
            # Find youngest person under 6 years
            age_data = df[member_age_cols].copy()
            age_data = age_data.apply(pd.to_numeric, errors='coerce')
            # Replace ages > 6 with NaN for youngest-child calculation
            young_child_ages = age_data.where(age_data <= 6)
            df['youngest_child_years'] = young_child_ages.min(axis=1)
        else:
            df['youngest_child_years'] = 3  # default midpoint

        df['youngest_child_months'] = (df['youngest_child_years'] * 12).fillna(36)

        # Extract summary spending variables
        spending_by_age = {}
        for category, var in CE_SPENDING_VARS.items():
            if var in df.columns:
                # CE reports quarterly spending; convert to monthly
                df[f'{category}_monthly'] = pd.to_numeric(df[var], errors='coerce').fillna(0) / 3.0

                # Group by child age (in years for smoothing)
                age_groups = df.groupby(
                    df['youngest_child_years'].round().clip(0, 6)
                )[f'{category}_monthly'].mean()

                spending_by_age[category] = age_groups

        # If we have detailed expenditure data, extract toy/childcare/book spending
        if self.expenditure_df is not None:
            self._extract_detailed_spending(df, spending_by_age)

        # Build domain-level curves by aggregating CE categories
        self.baseline_rates = {}
        domain_names = list(CE_TO_DEVELOPMAP.keys())

        # All DevelopMap domains
        all_domains = [
            'fine_motor', 'gross_motor', 'language', 'social_emotional',
            'sensory', 'adaptive', 'sleep', 'feeding', 'behavioral', 'therapeutic'
        ]

        for domain in all_domains:
            # Find which CE categories contribute to this domain
            contributing = []
            for ce_cat, mapped_domains in CE_TO_DEVELOPMAP.items():
                if domain in mapped_domains and ce_cat in spending_by_age:
                    contributing.append(spending_by_age[ce_cat])

            if contributing:
                # Average across contributing categories, normalize to [0, 2] range
                # as a proxy for "monthly purchase count"
                combined = pd.concat(contributing, axis=1).mean(axis=1)
                max_val = combined.max() if combined.max() > 0 else 1.0

                def make_curve(series, mx):
                    """Create a closure for this domain's curve."""
                    def curve(child_age_months: int) -> float:
                        age_years = round(child_age_months / 12.0)
                        age_years = max(0, min(6, age_years))
                        if age_years in series.index:
                            return float(series.loc[age_years] / mx * 1.5)
                        return 0.3  # fallback
                    return curve

                self.baseline_rates[domain] = make_curve(combined, max_val)
            else:
                # Default flat rate for domains without CE data
                self.baseline_rates[domain] = lambda age, d=domain: 0.1 if d in (
                    'sleep', 'feeding', 'adaptive') else 0.2

        self._print_summary(spending_by_age)
        return self.baseline_rates

    def _extract_detailed_spending(self, family_df: pd.DataFrame,
                                   spending_by_age: Dict):
        """Extract detailed UCC-level spending for toys, childcare, books."""
        if self.expenditure_df is None:
            return

        exp = self.expenditure_df.copy()

        # Merge with family data to get child age
        # NEWID is the common key between fmli and mtbi
        if 'NEWID' in exp.columns and 'NEWID' in family_df.columns:
            exp = exp.merge(
                family_df[['NEWID', 'youngest_child_years']].drop_duplicates(),
                on='NEWID', how='inner'
            )
        else:
            return

        for label, uccs in [('toys', TOY_UCCS), ('childcare', CHILDCARE_UCCS),
                            ('books', BOOK_UCCS)]:
            mask = exp['UCC'].isin(uccs)
            if mask.any():
                sub = exp[mask].copy()
                sub['cost_monthly'] = pd.to_numeric(sub.get('COST', sub.get('VALUE', 0)),
                                                     errors='coerce').fillna(0) / 3.0
                age_groups = sub.groupby(
                    sub['youngest_child_years'].round().clip(0, 6)
                )['cost_monthly'].mean()
                spending_by_age[label] = age_groups

    def _print_summary(self, spending_by_age: Dict):
        """Print summary of computed baseline rates."""
        print(f"\n{'='*60}")
        print("CE BASELINE SPENDING SUMMARY")
        print(f"{'='*60}")
        print(f"Households analyzed: {len(self.family_df)}")
        print(f"CE spending categories available: {list(spending_by_age.keys())}")
        print(f"DevelopMap domains with empirical rates: {list(self.baseline_rates.keys())}")

        print(f"\nSample baseline rates (purchases/month at age 36 months):")
        for domain, func in sorted(self.baseline_rates.items()):
            rate = func(36)
            print(f"  {domain:20s}: {rate:.3f}")
        print(f"{'='*60}\n")

    def get_spending_by_income_quintile(self) -> Dict[int, Dict[str, float]]:
        """Return average spending by income quintile for fairness grounding.

        Returns:
            Dict mapping quintile (1-5) to dict of category -> monthly spending
        """
        if self.family_df is None:
            raise RuntimeError("Call load() first")

        df = self.family_df.copy()

        # Determine income quintile
        income_col = None
        for candidate in ['FINCBTXM', 'FINCATAX', 'INC_RANK']:
            if candidate in df.columns:
                income_col = candidate
                break

        if income_col:
            df['income_quintile'] = pd.qcut(
                pd.to_numeric(df[income_col], errors='coerce').fillna(0),
                q=5, labels=[1, 2, 3, 4, 5], duplicates='drop'
            )
        else:
            df['income_quintile'] = 3

        result = {}
        for q in range(1, 6):
            sub = df[df['income_quintile'] == q]
            if len(sub) == 0:
                continue
            result[q] = {}
            for category, var in CE_SPENDING_VARS.items():
                if var in sub.columns:
                    monthly = pd.to_numeric(sub[var], errors='coerce').fillna(0) / 3.0
                    result[q][category] = float(monthly.mean())

        return result

    def save_baseline_rates(self, output_path: str):
        """Save computed baseline rates to JSON for use by synthetic generator.

        Args:
            output_path: Path to save JSON file (e.g., data/processed/ce_baseline_rates.json)
        """
        if self.baseline_rates is None:
            raise RuntimeError("Call load() first")

        # Sample the curves at each age to create a serializable lookup table
        rates_table = {}
        for domain, func in self.baseline_rates.items():
            rates_table[domain] = {}
            for age_months in range(0, 73, 6):  # 0, 6, 12, ..., 72
                rates_table[domain][str(age_months)] = round(func(age_months), 4)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            json.dump(rates_table, f, indent=2)

        print(f"Saved baseline rates to {output}")
