"""NSCH (National Survey of Children's Health) Data Loader.

Loads real NSCH microdata from:
    https://www.census.gov/programs-surveys/nsch/data/datasets.html

This module replaces the hardcoded prevalence values in configs/default.yaml
with empirical rates derived from NSCH survey data. The NSCH is a nationally
representative survey that provides data on children's health, including
developmental conditions, demographics, and healthcare access.

Required input: NSCH Topical CSV file (2022 or newer), downloadable from the
Census Bureau website above after agreeing to the data use terms.

Key NSCH variables used:
    SC_K2Q30A  - Developmental delay (ever told)
    SC_K2Q31A  - Intellectual disability
    SC_K2Q32A  - Speech/language disorder
    SC_K2Q33A  - Learning disability
    SC_K2Q34A  - ADD/ADHD
    SC_K2Q35A  - Autism/ASD
    SC_K2Q36A  - Other developmental condition
    SC_AGE_YEARS - Child age in years
    HIESSION   - Household income (poverty level ratio)
    SC_RACE_R  - Child race/ethnicity
    TOTKIDS_R  - Total children in household
    SC_SEX     - Child sex
    AGEPOS4    - Age when first told about condition
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable


# NSCH variable mappings
CONDITION_VARS = {
    'developmental_delay': 'SC_K2Q30A',
    'intellectual_disability': 'SC_K2Q31A',
    'speech_language': 'SC_K2Q32A',
    'learning_disability': 'SC_K2Q33A',
    'adhd': 'SC_K2Q34A',
    'asd': 'SC_K2Q35A',
    'other_developmental': 'SC_K2Q36A',
}

# Map NSCH race codes to labels
RACE_LABELS = {
    1: 'white',
    2: 'black',
    3: 'other',
    4: 'asian',
    5: 'native_american',
    7: 'multiracial',
}

# Map NSCH income ratio to quintiles
# HIESSION: 1=0-99% FPL, 2=100-199%, 3=200-399%, 4=400%+
INCOME_QUINTILE_MAP = {
    1: 1,
    2: 2,
    3: 3,
    4: 5,
}

# Map to geography categories
# SC_METRO_STAT: 1=metro, 2=non-metro/micro, 3=non-metro/non-micro
GEOGRAPHY_MAP = {
    1: 'urban',
    2: 'suburban',
    3: 'rural',
}


class NSCHLoader:
    """Load and process NSCH microdata for developmental delay analysis.

    This loader extracts developmental condition prevalence rates,
    demographic distributions, and diagnosis age distributions from
    the NSCH survey, replacing hardcoded values throughout the codebase.
    """

    def __init__(self, nsch_csv_path: str):
        """Initialize NSCH loader.

        Args:
            nsch_csv_path: Path to the NSCH topical CSV file (2022+).
        """
        self.path = Path(nsch_csv_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"NSCH file not found: {self.path}\n"
                f"Download from: https://www.census.gov/programs-surveys/nsch/data/datasets.html"
            )
        self.raw_df: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Load and process NSCH CSV into a standardized DataFrame.

        Returns:
            DataFrame with columns: family_id, child_age_months, has_delay,
            delay_type, income_quintile, geography, ethnicity, n_children,
            age_at_diagnosis_months
        """
        print(f"Loading NSCH data from {self.path}...")
        self.raw_df = pd.read_csv(self.path, low_memory=False)
        print(f"  Loaded {len(self.raw_df)} records")

        df = self.raw_df.copy()

        # --- Determine delay status ---
        condition_cols = [v for v in CONDITION_VARS.values() if v in df.columns]
        if not condition_cols:
            raise ValueError(
                "No developmental condition columns found in NSCH data. "
                f"Expected columns like: {list(CONDITION_VARS.values())}"
            )

        # NSCH coding: 1 = Yes, 2 = No. Recode to boolean.
        for col in condition_cols:
            df[col] = df[col].apply(lambda x: 1 if x == 1 else 0)

        df['has_delay'] = df[condition_cols].max(axis=1).astype(bool)

        # --- Determine primary delay type ---
        # Priority order for assigning primary type when multiple present
        type_priority = ['asd', 'adhd', 'speech_language', 'developmental_delay',
                         'intellectual_disability', 'learning_disability',
                         'other_developmental']

        def _get_primary_delay(row):
            for dtype in type_priority:
                var = CONDITION_VARS.get(dtype)
                if var and var in row.index and row[var] == 1:
                    # Consolidate into the 4 categories used by the model
                    if dtype in ('speech_language',):
                        return 'language'
                    elif dtype in ('developmental_delay', 'intellectual_disability',
                                   'learning_disability'):
                        return 'motor'  # broad developmental — maps to motor in model
                    elif dtype == 'adhd':
                        return 'adhd'
                    elif dtype == 'asd':
                        return 'asd'
                    else:
                        return 'other'
            return None

        df['delay_type'] = df.apply(_get_primary_delay, axis=1)

        # --- Demographics ---
        # Age
        if 'SC_AGE_YEARS' in df.columns:
            df['child_age_months'] = df['SC_AGE_YEARS'] * 12
        else:
            df['child_age_months'] = np.nan

        # Income quintile
        if 'HIESSION' in df.columns:
            df['income_quintile'] = df['HIESSION'].map(INCOME_QUINTILE_MAP).fillna(3).astype(int)
        elif 'POVLEV4_1722' in df.columns:
            df['income_quintile'] = df['POVLEV4_1722'].map(INCOME_QUINTILE_MAP).fillna(3).astype(int)
        else:
            df['income_quintile'] = 3

        # Geography
        metro_col = None
        for candidate in ['SC_METRO_STAT', 'METRO_YN']:
            if candidate in df.columns:
                metro_col = candidate
                break
        if metro_col:
            df['geography'] = df[metro_col].map(GEOGRAPHY_MAP).fillna('suburban')
        else:
            df['geography'] = 'suburban'

        # Ethnicity
        if 'SC_RACE_R' in df.columns:
            df['ethnicity'] = df['SC_RACE_R'].map(RACE_LABELS).fillna('other')
        else:
            df['ethnicity'] = 'other'

        # Number of children
        if 'TOTKIDS_R' in df.columns:
            df['n_children'] = df['TOTKIDS_R'].clip(upper=5).astype(int)
        else:
            df['n_children'] = 1

        # Age at diagnosis (approximate from NSCH age-first-told variables)
        df['age_at_diagnosis_months'] = self._extract_diagnosis_age(df)

        # --- Build output ---
        df['family_id'] = ['NSCH_' + str(i).zfill(7) for i in range(len(df))]

        output_cols = [
            'family_id', 'child_age_months', 'has_delay', 'delay_type',
            'income_quintile', 'geography', 'ethnicity', 'n_children',
            'age_at_diagnosis_months'
        ]
        self.processed_df = df[output_cols].copy()

        self._print_summary()
        return self.processed_df

    def _extract_diagnosis_age(self, df: pd.DataFrame) -> pd.Series:
        """Extract approximate age-at-diagnosis from NSCH variables.

        NSCH provides age-when-first-told for some conditions via AGEPOS4
        and similar variables. When unavailable, we estimate from current
        age and condition status.
        """
        diag_age = pd.Series(np.nan, index=df.index)

        # Try AGEPOS4 (age first told about developmental delay, in years)
        age_first_told_vars = ['AGEPOS4', 'K2Q35A_1_YEARS']
        for var in age_first_told_vars:
            if var in df.columns:
                mask = diag_age.isna() & df[var].notna() & (df[var] > 0)
                diag_age.loc[mask] = df.loc[mask, var] * 12  # convert years to months

        # For remaining delayed children without diagnosis age, use current age
        # as upper bound (they were diagnosed at some point before current age)
        if 'SC_AGE_YEARS' in df.columns:
            mask = diag_age.isna() & df['has_delay']
            diag_age.loc[mask] = df.loc[mask, 'SC_AGE_YEARS'] * 12

        return diag_age

    def _print_summary(self):
        """Print summary statistics."""
        df = self.processed_df
        n = len(df)
        print(f"\n{'='*60}")
        print("NSCH DATA SUMMARY")
        print(f"{'='*60}")
        print(f"Total records: {n}")
        print(f"Delay prevalence: {df['has_delay'].mean()*100:.1f}%")

        print(f"\nPrevalence by condition:")
        for dtype in ['language', 'motor', 'asd', 'adhd', 'other']:
            count = (df['delay_type'] == dtype).sum()
            pct = count / n * 100
            print(f"  {dtype:20s}: {count:6d} ({pct:.2f}%)")

        print(f"\nPrevalence by income quintile:")
        for q in sorted(df['income_quintile'].unique()):
            sub = df[df['income_quintile'] == q]
            print(f"  Quintile {q}: {sub['has_delay'].mean()*100:.1f}% (n={len(sub)})")

        print(f"\nPrevalence by geography:")
        for geo in sorted(df['geography'].unique()):
            sub = df[df['geography'] == geo]
            print(f"  {geo:10s}: {sub['has_delay'].mean()*100:.1f}% (n={len(sub)})")

        print(f"\nPrevalence by ethnicity:")
        for eth in sorted(df['ethnicity'].unique()):
            sub = df[df['ethnicity'] == eth]
            print(f"  {eth:15s}: {sub['has_delay'].mean()*100:.1f}% (n={len(sub)})")
        print(f"{'='*60}\n")

    def get_diagnosis_age_distribution(self, delay_type: str) -> np.ndarray:
        """Return empirical distribution of age-at-diagnosis for a condition.

        This replaces the paper's single-number "phantom baseline" with a
        real distribution derived from NSCH data.

        Args:
            delay_type: One of 'language', 'motor', 'asd', 'adhd'

        Returns:
            Array of diagnosis ages in months for the given condition
        """
        if self.processed_df is None:
            raise RuntimeError("Call load() first")

        mask = (self.processed_df['delay_type'] == delay_type) & \
               self.processed_df['age_at_diagnosis_months'].notna()
        ages = self.processed_df.loc[mask, 'age_at_diagnosis_months'].values

        if len(ages) == 0:
            print(f"WARNING: No diagnosis age data for {delay_type}")
            return np.array([])

        return ages

    def get_prevalence_by_demographic(self) -> Dict[str, Dict[str, float]]:
        """Return prevalence rates by demographic group.

        Returns:
            Nested dict: {attribute: {group: prevalence_rate}}
        """
        if self.processed_df is None:
            raise RuntimeError("Call load() first")

        df = self.processed_df
        result = {}

        for attr in ['income_quintile', 'geography', 'ethnicity']:
            result[attr] = {}
            for group in df[attr].unique():
                sub = df[df[attr] == group]
                result[attr][str(group)] = float(sub['has_delay'].mean())

        # Overall
        result['overall'] = {
            'any_delay': float(df['has_delay'].mean()),
        }
        for dtype in ['language', 'motor', 'asd', 'adhd']:
            result['overall'][dtype] = float((df['delay_type'] == dtype).mean())

        return result

    def get_delay_prevalence_rates(self) -> Dict[str, float]:
        """Return overall delay prevalence rates for config replacement.

        Returns:
            Dict with keys matching configs/default.yaml structure
        """
        if self.processed_df is None:
            raise RuntimeError("Call load() first")

        df = self.processed_df
        n = len(df)
        return {
            'delay_prevalence': float(df['has_delay'].mean()),
            'language': float((df['delay_type'] == 'language').sum() / n),
            'motor': float((df['delay_type'] == 'motor').sum() / n),
            'asd': float((df['delay_type'] == 'asd').sum() / n),
            'adhd': float((df['delay_type'] == 'adhd').sum() / n),
        }

    def get_population_statistics(self) -> Dict[str, Dict]:
        """Return comprehensive population statistics for validation.

        Returns:
            Dict with demographic distributions, prevalence by group,
            and diagnosis age statistics for real validation metrics.
        """
        if self.processed_df is None:
            raise RuntimeError("Call load() first")

        df = self.processed_df
        n = len(df)

        stats = {
            'sample_size': n,
            'prevalence': self.get_delay_prevalence_rates(),
            'demographics': {},
            'diagnosis_age': {},
            'co_occurrence': {},
        }

        # Demographic distributions
        for attr in ['income_quintile', 'geography', 'ethnicity']:
            dist = df[attr].value_counts(normalize=True).to_dict()
            stats['demographics'][attr] = {str(k): float(v) for k, v in dist.items()}

        # Prevalence by demographic group
        stats['prevalence_by_group'] = self.get_prevalence_by_demographic()

        # Diagnosis age statistics by condition
        for dtype in ['language', 'motor', 'asd', 'adhd']:
            ages = self.get_diagnosis_age_distribution(dtype)
            if len(ages) > 0:
                stats['diagnosis_age'][dtype] = {
                    'mean_months': float(np.mean(ages)),
                    'median_months': float(np.median(ages)),
                    'std_months': float(np.std(ages)),
                    'p25_months': float(np.percentile(ages, 25)),
                    'p75_months': float(np.percentile(ages, 75)),
                    'n': int(len(ages)),
                }

        # Age distribution of sample
        if 'child_age_months' in df.columns:
            ages = df['child_age_months'].dropna()
            stats['age_distribution'] = {
                'mean_months': float(ages.mean()),
                'std_months': float(ages.std()),
                'min_months': float(ages.min()),
                'max_months': float(ages.max()),
            }

        # Gender distribution if available
        if 'SC_SEX' in self.raw_df.columns:
            sex_dist = self.raw_df['SC_SEX'].value_counts(normalize=True)
            stats['demographics']['gender'] = {
                'male': float(sex_dist.get(1, 0)),
                'female': float(sex_dist.get(2, 0)),
            }

            # Prevalence by gender
            for sex_code, sex_label in [(1, 'male'), (2, 'female')]:
                mask = self.raw_df['SC_SEX'] == sex_code
                if mask.sum() > 0:
                    sub_df = df.loc[mask]
                    stats['prevalence_by_group'].setdefault('gender', {})[sex_label] = \
                        float(sub_df['has_delay'].mean())

        return stats

    def get_reference_prevalence_for_validation(self) -> Dict[str, Dict]:
        """Return reference prevalence rates for model validation.

        These rates serve as ground truth for comparing model predictions
        against real population statistics.

        Returns:
            Dict with overall and stratified prevalence rates
        """
        if self.processed_df is None:
            raise RuntimeError("Call load() first")

        df = self.processed_df

        reference = {
            'overall': {
                'any_delay': float(df['has_delay'].mean()),
                'by_type': {},
            },
            'by_income': {},
            'by_geography': {},
            'by_ethnicity': {},
            'confidence_intervals': {},
        }

        # By delay type
        for dtype in ['language', 'motor', 'asd', 'adhd']:
            count = (df['delay_type'] == dtype).sum()
            rate = count / len(df)
            reference['overall']['by_type'][dtype] = {
                'prevalence': float(rate),
                'n': int(count),
            }

        # Stratified prevalence with confidence intervals
        for attr, key in [('income_quintile', 'by_income'),
                          ('geography', 'by_geography'),
                          ('ethnicity', 'by_ethnicity')]:
            for group in df[attr].unique():
                sub = df[df[attr] == group]
                n = len(sub)
                p = sub['has_delay'].mean()
                # Wilson score interval for 95% CI
                z = 1.96
                denom = 1 + z**2 / n
                center = (p + z**2 / (2*n)) / denom
                margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom

                reference[key][str(group)] = {
                    'prevalence': float(p),
                    'n': int(n),
                    'ci_lower': float(max(0, center - margin)),
                    'ci_upper': float(min(1, center + margin)),
                }

        return reference

    def compare_with_model_predictions(self, predictions: pd.DataFrame,
                                        pred_col: str = 'predicted_delay',
                                        group_col: Optional[str] = None) -> Dict[str, Dict]:
        """Compare model predictions against NSCH population statistics.

        Args:
            predictions: DataFrame with model predictions
            pred_col: Column name for predicted delay probability/label
            group_col: Optional column for stratified comparison

        Returns:
            Dict with comparison metrics (bias, calibration, etc.)
        """
        if self.processed_df is None:
            raise RuntimeError("Call load() first")

        reference = self.get_reference_prevalence_for_validation()
        
        comparison = {
            'overall': {},
            'by_group': {},
            'calibration': {},
        }

        # Overall comparison
        pred_rate = predictions[pred_col].mean()
        true_rate = reference['overall']['any_delay']
        comparison['overall'] = {
            'predicted_prevalence': float(pred_rate),
            'nsch_prevalence': float(true_rate),
            'absolute_bias': float(pred_rate - true_rate),
            'relative_bias': float((pred_rate - true_rate) / true_rate) if true_rate > 0 else None,
        }

        # Stratified comparison if group column provided
        if group_col and group_col in predictions.columns:
            for group in predictions[group_col].unique():
                sub = predictions[predictions[group_col] == group]
                pred_rate_g = sub[pred_col].mean()

                # Find matching NSCH reference
                nsch_rate_g = None
                for ref_key in ['by_income', 'by_geography', 'by_ethnicity']:
                    if str(group) in reference.get(ref_key, {}):
                        nsch_rate_g = reference[ref_key][str(group)]['prevalence']
                        break

                comparison['by_group'][str(group)] = {
                    'predicted_prevalence': float(pred_rate_g),
                    'nsch_prevalence': float(nsch_rate_g) if nsch_rate_g else None,
                    'n': int(len(sub)),
                }

        return comparison

    def save_population_statistics(self, output_path: str):
        """Save population statistics to JSON for validation scripts.

        Args:
            output_path: Path to save JSON file
        """
        import json
        from pathlib import Path

        stats = self.get_population_statistics()
        stats['reference_prevalence'] = self.get_reference_prevalence_for_validation()
        stats['metadata'] = {
            'source': 'NSCH (National Survey of Children\'s Health)',
            'url': 'https://www.census.gov/programs-surveys/nsch/data/datasets.html',
            'file': str(self.path),
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Saved NSCH population statistics to {output_path}")
