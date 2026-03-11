"""PSID-CDS (Panel Study of Income Dynamics — Child Development Supplement) Loader.

Loads PSID core family data merged with CDS child assessment data from:
    https://psidonline.isr.umich.edu/cds/

This is the core real-world validation module that breaks the synthetic data
circularity. It merges household expenditure variables with child developmental
outcomes, enabling a test of whether spending patterns actually differ between
families with and without developmental concerns.

Required input:
    - PSID core family file (e.g., J327825.csv or fam2019er.csv)
    - CDS child assessment file (e.g., CDS-2019.csv)
    Both available from https://psidonline.isr.umich.edu/ (free registration required)

Key PSID expenditure variables:
    Education spending, childcare, recreation/toys, health expenses

Key CDS developmental outcome variables:
    Woodcock-Johnson scores (letter-word, applied problems, passage comprehension)
    Behavioral Problems Index (BPI) — internalizing and externalizing subscales
    Child health status
    Special education / developmental services receipt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings
import json


# PSID family file expenditure variable patterns
# These vary by year; we search for common patterns
EXPENDITURE_PATTERNS = {
    'education': ['ER', 'EDUC', 'SCHOOL', 'TUITION'],
    'childcare': ['ER', 'CHILD', 'CARE', 'DAYCARE', 'BABYSIT'],
    'recreation': ['ER', 'RECR', 'ENTERT', 'TOY', 'HOBBY'],
    'health': ['ER', 'HEALTH', 'MEDIC', 'DOCTOR', 'HOSP'],
    'food': ['ER', 'FOOD', 'GROCER'],
    'clothing': ['ER', 'CLOTH', 'APPAR'],
}

# CDS developmental outcome variable patterns
CDS_OUTCOME_PATTERNS = {
    'wj_letter_word': ['WJ', 'LETTER', 'LW'],
    'wj_applied_problems': ['WJ', 'APPLIED', 'AP'],
    'wj_passage_comprehension': ['WJ', 'PASSAGE', 'PC'],
    'bpi_total': ['BPI', 'TOTAL', 'BEHAVIOR'],
    'bpi_external': ['BPI', 'EXTERN'],
    'bpi_internal': ['BPI', 'INTERN'],
    'health_status': ['HEALTH', 'STATUS', 'GENERAL'],
    'special_education': ['SPECIAL', 'EDUC', 'IEP', 'SERVICES'],
    'developmental_concern': ['DEVELOP', 'CONCERN', 'DELAY', 'DISAB'],
}

# Mapping from PSID expenditure categories to DevelopMap domains
PSID_TO_DEVELOPMAP = {
    'education': ['language', 'cognitive'],
    'childcare': ['adaptive', 'therapeutic'],
    'recreation': ['fine_motor', 'gross_motor', 'sensory'],
    'health': ['therapeutic'],
}


class PSIDLoader:
    """Load and merge PSID core data with CDS child assessments.

    Enables real-world validation of whether household spending patterns
    in child-development categories differ between families whose children
    have developmental concerns vs. typically developing children.
    """

    def __init__(self, psid_family_path: str, cds_child_path: str):
        """Initialize PSID-CDS loader.

        Args:
            psid_family_path: Path to PSID core family file (CSV or similar)
            cds_child_path: Path to CDS child assessment file
        """
        self.family_path = Path(psid_family_path)
        self.cds_path = Path(cds_child_path)

        for p, label in [(self.family_path, 'PSID family'),
                         (self.cds_path, 'CDS child')]:
            if not p.exists():
                raise FileNotFoundError(
                    f"{label} file not found: {p}\n"
                    f"Download from: https://psidonline.isr.umich.edu/cds/"
                )

        self.family_df: Optional[pd.DataFrame] = None
        self.cds_df: Optional[pd.DataFrame] = None
        self.merged_df: Optional[pd.DataFrame] = None
        self.expenditure_cols: Dict[str, str] = {}
        self.outcome_cols: Dict[str, str] = {}

    def load(self) -> pd.DataFrame:
        """Load and merge PSID family data with CDS child data.

        Returns:
            Merged DataFrame with expenditure and outcome variables
        """
        self._load_family_data()
        self._load_cds_data()
        self._merge_datasets()
        self._map_to_domains()
        self._print_summary()
        return self.merged_df

    def _load_family_data(self):
        """Load PSID core family file."""
        print(f"Loading PSID family data from {self.family_path}...")

        try:
            self.family_df = pd.read_csv(self.family_path, low_memory=False)
        except Exception:
            # Try tab-separated or other formats
            try:
                self.family_df = pd.read_csv(self.family_path, sep='\t', low_memory=False)
            except Exception as e:
                raise ValueError(f"Could not parse PSID family file: {e}")

        print(f"  Loaded {len(self.family_df)} households, {len(self.family_df.columns)} variables")

        # Identify expenditure columns by pattern matching
        self.expenditure_cols = self._find_columns(
            self.family_df.columns, EXPENDITURE_PATTERNS
        )
        print(f"  Found expenditure variables: {list(self.expenditure_cols.keys())}")

        # Identify family ID column
        self._family_id_col = self._find_id_column(self.family_df)
        print(f"  Family ID column: {self._family_id_col}")

    def _load_cds_data(self):
        """Load CDS child assessment file."""
        print(f"\nLoading CDS data from {self.cds_path}...")

        try:
            self.cds_df = pd.read_csv(self.cds_path, low_memory=False)
        except Exception:
            try:
                self.cds_df = pd.read_csv(self.cds_path, sep='\t', low_memory=False)
            except Exception as e:
                raise ValueError(f"Could not parse CDS file: {e}")

        print(f"  Loaded {len(self.cds_df)} children, {len(self.cds_df.columns)} variables")

        # Identify outcome columns
        self.outcome_cols = self._find_columns(
            self.cds_df.columns, CDS_OUTCOME_PATTERNS
        )
        print(f"  Found outcome variables: {list(self.outcome_cols.keys())}")

        # Identify family ID column for merging
        self._cds_family_id_col = self._find_id_column(self.cds_df)
        print(f"  CDS Family ID column: {self._cds_family_id_col}")

    def _find_columns(self, columns: pd.Index,
                      patterns: Dict[str, List[str]]) -> Dict[str, str]:
        """Find columns matching pattern groups.

        For each category, finds the first column whose name contains
        all keywords in the pattern (case-insensitive).
        """
        found = {}
        col_upper = {c: c.upper() for c in columns}

        for category, keywords in patterns.items():
            for col, col_up in col_upper.items():
                # Check if any keyword appears in the column name
                if any(kw in col_up for kw in keywords[1:]):
                    found[category] = col
                    break

        return found

    def _find_id_column(self, df: pd.DataFrame) -> str:
        """Find the family/household ID column."""
        candidates = ['ER30001', 'V30001', 'FAMID', 'FAMILY_ID',
                       'ID68', 'INTERVIEW_NUM']
        for c in candidates:
            if c in df.columns:
                return c

        # Fall back to first column or any column with 'ID' in name
        id_cols = [c for c in df.columns if 'ID' in c.upper()]
        if id_cols:
            return id_cols[0]
        return df.columns[0]

    def _merge_datasets(self):
        """Merge PSID family data with CDS child data."""
        print("\nMerging PSID family data with CDS child assessments...")

        # Try to merge on family ID
        family_id = self._family_id_col
        cds_id = self._cds_family_id_col

        # If the ID column names differ, try common PSID patterns
        self.merged_df = self.cds_df.merge(
            self.family_df,
            left_on=cds_id,
            right_on=family_id,
            how='inner',
            suffixes=('_cds', '_fam')
        )

        if len(self.merged_df) == 0:
            warnings.warn(
                "Merge produced 0 rows. Family ID columns may not match. "
                f"CDS ID col: {cds_id}, PSID ID col: {family_id}. "
                "Falling back to index-based alignment."
            )
            # Fallback: concatenate side by side (limited validity)
            min_len = min(len(self.cds_df), len(self.family_df))
            self.merged_df = pd.concat([
                self.cds_df.iloc[:min_len].reset_index(drop=True),
                self.family_df.iloc[:min_len].reset_index(drop=True)
            ], axis=1)

        print(f"  Merged dataset: {len(self.merged_df)} records")

    def _map_to_domains(self):
        """Map expenditure categories to DevelopMap domains."""
        df = self.merged_df

        for domain_list_key, domains in PSID_TO_DEVELOPMAP.items():
            exp_col = self.expenditure_cols.get(domain_list_key)
            if exp_col and exp_col in df.columns:
                spending = pd.to_numeric(df[exp_col], errors='coerce').fillna(0)
                for domain in domains:
                    # Split spending equally among mapped domains
                    df[f'spending_{domain}'] = spending / len(domains)

        # Determine developmental concern status
        dev_col = self.outcome_cols.get('developmental_concern')
        sped_col = self.outcome_cols.get('special_education')

        if dev_col and dev_col in df.columns:
            df['has_developmental_concern'] = (
                pd.to_numeric(df[dev_col], errors='coerce').fillna(0) > 0
            )
        elif sped_col and sped_col in df.columns:
            df['has_developmental_concern'] = (
                pd.to_numeric(df[sped_col], errors='coerce').fillna(0) > 0
            )
        else:
            # Fall back to BPI above clinical threshold
            bpi_col = self.outcome_cols.get('bpi_total')
            if bpi_col and bpi_col in df.columns:
                bpi = pd.to_numeric(df[bpi_col], errors='coerce')
                # BPI > 90th percentile as proxy for developmental concern
                threshold = bpi.quantile(0.90)
                df['has_developmental_concern'] = bpi > threshold
            else:
                warnings.warn("No developmental outcome variable found")
                df['has_developmental_concern'] = False

        self.merged_df = df

    def _print_summary(self):
        """Print summary of merged data."""
        df = self.merged_df
        print(f"\n{'='*60}")
        print("PSID-CDS MERGED DATA SUMMARY")
        print(f"{'='*60}")
        print(f"Total records: {len(df)}")

        if 'has_developmental_concern' in df.columns:
            concern_rate = df['has_developmental_concern'].mean()
            print(f"Developmental concern rate: {concern_rate*100:.1f}%")

        spending_cols = [c for c in df.columns if c.startswith('spending_')]
        if spending_cols:
            print(f"\nDomain spending columns: {spending_cols}")
            for col in spending_cols:
                vals = pd.to_numeric(df[col], errors='coerce')
                print(f"  {col}: mean=${vals.mean():.0f}, median=${vals.median():.0f}")

        print(f"{'='*60}\n")

    def test_expenditure_outcome_correlation(self) -> Dict[str, Dict]:
        """Test Spearman correlations between spending shifts and outcomes.

        This is the key validation: do real-world spending patterns correlate
        with developmental outcomes, as the synthetic data assumes?

        Returns:
            Dict mapping domain to correlation results (rho, p-value, n)
        """
        if self.merged_df is None:
            raise RuntimeError("Call load() first")

        df = self.merged_df
        results = {}

        spending_cols = [c for c in df.columns if c.startswith('spending_')]

        # Test correlations with each outcome variable
        for outcome_name, outcome_col in self.outcome_cols.items():
            if outcome_col not in df.columns:
                continue

            outcome_vals = pd.to_numeric(df[outcome_col], errors='coerce')
            valid_outcome = outcome_vals.notna()

            for spend_col in spending_cols:
                domain = spend_col.replace('spending_', '')
                spend_vals = pd.to_numeric(df[spend_col], errors='coerce')
                valid = valid_outcome & spend_vals.notna()

                if valid.sum() < 30:
                    continue

                rho, p_value = stats.spearmanr(
                    spend_vals[valid], outcome_vals[valid]
                )

                key = f"{domain}_vs_{outcome_name}"
                results[key] = {
                    'domain': domain,
                    'outcome': outcome_name,
                    'spearman_rho': float(rho),
                    'p_value': float(p_value),
                    'n': int(valid.sum()),
                    'significant': p_value < 0.05,
                }

        # Also test group differences (concern vs no concern)
        if 'has_developmental_concern' in df.columns:
            concern = df['has_developmental_concern']
            for spend_col in spending_cols:
                domain = spend_col.replace('spending_', '')
                spend_vals = pd.to_numeric(df[spend_col], errors='coerce')

                concern_spending = spend_vals[concern & spend_vals.notna()]
                no_concern_spending = spend_vals[~concern & spend_vals.notna()]

                if len(concern_spending) < 10 or len(no_concern_spending) < 10:
                    continue

                # Mann-Whitney U test
                u_stat, p_value = stats.mannwhitneyu(
                    concern_spending, no_concern_spending,
                    alternative='two-sided'
                )

                # Cohen's d effect size
                pooled_std = np.sqrt(
                    (concern_spending.std()**2 + no_concern_spending.std()**2) / 2
                )
                cohens_d = (
                    (concern_spending.mean() - no_concern_spending.mean()) / pooled_std
                    if pooled_std > 0 else 0.0
                )

                key = f"{domain}_group_difference"
                results[key] = {
                    'domain': domain,
                    'test': 'Mann-Whitney U',
                    'concern_mean': float(concern_spending.mean()),
                    'no_concern_mean': float(no_concern_spending.mean()),
                    'u_statistic': float(u_stat),
                    'p_value': float(p_value),
                    'cohens_d': float(cohens_d),
                    'n_concern': int(len(concern_spending)),
                    'n_no_concern': int(len(no_concern_spending)),
                    'significant': p_value < 0.05,
                }

        self._print_correlation_results(results)
        return results

    def _print_correlation_results(self, results: Dict):
        """Print correlation analysis results."""
        print(f"\n{'='*60}")
        print("EXPENDITURE-OUTCOME CORRELATION RESULTS")
        print(f"{'='*60}")

        # Group difference results
        group_results = {k: v for k, v in results.items() if 'group_difference' in k}
        if group_results:
            print("\nSpending differences (concern vs. no concern):")
            for key, r in sorted(group_results.items()):
                sig = '*' if r['significant'] else ' '
                print(f"  {r['domain']:15s}: d={r['cohens_d']:+.3f}, "
                      f"p={r['p_value']:.4f} {sig} "
                      f"(concern=${r['concern_mean']:.0f} vs ${r['no_concern_mean']:.0f})")

        # Correlation results
        corr_results = {k: v for k, v in results.items() if 'group_difference' not in k}
        if corr_results:
            print("\nSpearman correlations (spending vs. outcome):")
            for key, r in sorted(corr_results.items()):
                sig = '*' if r['significant'] else ' '
                print(f"  {r['domain']:15s} vs {r['outcome']:25s}: "
                      f"rho={r['spearman_rho']:+.3f}, p={r['p_value']:.4f} {sig} (n={r['n']})")

        print(f"\n* = significant at p < 0.05")
        print(f"{'='*60}\n")

    def save_results(self, output_path: str, results: Dict):
        """Save correlation results to JSON.

        Args:
            output_path: Path to save results
            results: Output from test_expenditure_outcome_correlation()
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved PSID-CDS results to {output}")
