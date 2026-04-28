"""Kaggle ASD Screening Dataset Loader.

Loads autism screening data from Kaggle ASD datasets:
    - Autism Screening Adult: https://www.kaggle.com/datasets/andrewmvd/autism-screening-on-adults
    - Autism Screening Child: https://www.kaggle.com/datasets/fabdelja/autism-screening-for-toddlers
    - ASD Screening Data: https://www.kaggle.com/datasets/faizunnabi/autism-screening

These datasets contain real diagnostic labels (ASD positive/negative) along with
behavioral screening questionnaire responses (AQ-10, Q-CHAT variants).

We use this to:
1. Provide real ASD diagnostic labels for validation
2. Extract feature importance from screening questionnaires
3. Validate model predictions against clinical screening outcomes
4. Enable real validation metrics instead of synthetic circularity

Required files (download from Kaggle):
    - Autism-Adult-Data.csv or autism_screening_adult.csv
    - Autism-Child-Data.csv or autism_screening_child.csv  
    - Toddler Autism dataset.csv or toddlers_autism.csv

Usage:
    loader = KaggleASDLoader('data/raw/kaggle_asd/')
    loader.load()
    stats = loader.get_prevalence_statistics()
    loader.save_processed('data/processed/kaggle_asd_processed.json')
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Standard column name mappings across different dataset versions
COLUMN_MAPPINGS = {
    # Target variable
    'class/asd': 'asd_label',
    'class': 'asd_label',
    'asd': 'asd_label',
    'autism': 'asd_label',
    'class/asd traits': 'asd_label',
    
    # Demographics
    'age': 'age',
    'age_desc': 'age_group',
    'gender': 'gender',
    'sex': 'gender',
    'ethnicity': 'ethnicity',
    'country_of_res': 'country',
    'country': 'country',
    
    # Family history
    'family_mem_with_asd': 'family_history_asd',
    'austim': 'family_history_asd',  # Typo in original dataset
    'jaundice': 'jaundice_at_birth',
    
    # Screening scores
    'result': 'screening_score',
    'score': 'screening_score',
    'qchat-10-score': 'screening_score',
    
    # AQ-10 questions (adult/child)
    'a1_score': 'q1',
    'a2_score': 'q2',
    'a3_score': 'q3',
    'a4_score': 'q4',
    'a5_score': 'q5',
    'a6_score': 'q6',
    'a7_score': 'q7',
    'a8_score': 'q8',
    'a9_score': 'q9',
    'a10_score': 'q10',
}

# AQ-10 question domains (maps to developmental domains)
AQ10_DOMAINS = {
    'q1': 'social_emotional',  # Social situations
    'q2': 'social_emotional',  # Reading between lines
    'q3': 'behavioral',        # Focusing on details
    'q4': 'language',          # Understanding others
    'q5': 'sensory',           # Noticing patterns
    'q6': 'social_emotional',  # Social chitchat
    'q7': 'behavioral',        # Routines
    'q8': 'social_emotional',  # Imagining characters
    'q9': 'social_emotional',  # Reading faces
    'q10': 'behavioral',       # Making friends
}


class KaggleASDLoader:
    """Load and process Kaggle ASD screening datasets."""

    def __init__(self, data_dir: str):
        """Initialize loader.

        Args:
            data_dir: Directory containing Kaggle ASD CSV files
        """
        self.data_dir = Path(data_dir)
        
        self.adult_data: Optional[pd.DataFrame] = None
        self.child_data: Optional[pd.DataFrame] = None
        self.toddler_data: Optional[pd.DataFrame] = None
        self.combined_data: Optional[pd.DataFrame] = None
        
        self._prevalence_stats: Optional[Dict] = None

    def _find_file(self, patterns: List[str]) -> Optional[Path]:
        """Find a file matching any of the given patterns.

        Args:
            patterns: List of filename patterns to search for

        Returns:
            Path to first matching file, or None
        """
        for pattern in patterns:
            matches = list(self.data_dir.glob(pattern))
            if matches:
                return matches[0]
            # Try case-insensitive
            matches = list(self.data_dir.glob(pattern.lower()))
            if matches:
                return matches[0]
        return None

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across different dataset versions.

        Args:
            df: Raw dataframe

        Returns:
            DataFrame with standardized column names
        """
        df = df.copy()
        df.columns = df.columns.str.lower().str.strip()
        
        # Apply mappings
        rename_map = {}
        for old_name, new_name in COLUMN_MAPPINGS.items():
            if old_name in df.columns:
                rename_map[old_name] = new_name
        
        df = df.rename(columns=rename_map)
        return df

    def _standardize_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize ASD labels to binary 0/1.

        Args:
            df: DataFrame with asd_label column

        Returns:
            DataFrame with standardized binary labels
        """
        if 'asd_label' not in df.columns:
            return df

        df = df.copy()
        label_col = df['asd_label']
        
        # Handle various label formats
        if label_col.dtype == object:
            label_col = label_col.str.lower().str.strip()
            df['asd_label'] = label_col.map({
                'yes': 1, 'no': 0,
                'asd': 1, 'no asd': 0,
                'true': 1, 'false': 0,
                '1': 1, '0': 0,
                'positive': 1, 'negative': 0,
            })
        
        # Convert to int, handling NaN
        df['asd_label'] = pd.to_numeric(df['asd_label'], errors='coerce')
        
        return df

    def _standardize_gender(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize gender to M/F.

        Args:
            df: DataFrame with gender column

        Returns:
            DataFrame with standardized gender
        """
        if 'gender' not in df.columns:
            return df

        df = df.copy()
        gender_col = df['gender'].astype(str).str.lower().str.strip()
        df['gender'] = gender_col.map({
            'm': 'M', 'male': 'M', 'man': 'M', '1': 'M',
            'f': 'F', 'female': 'F', 'woman': 'F', '0': 'F',
        })
        
        return df

    def load_adult_data(self) -> Optional[pd.DataFrame]:
        """Load adult ASD screening dataset.

        Returns:
            Processed adult DataFrame or None
        """
        patterns = [
            'Autism-Adult-Data*.csv',
            'autism*adult*.csv',
            'adult*.csv',
            'ASD_adult*.csv',
        ]
        
        path = self._find_file(patterns)
        if path is None:
            return None

        df = pd.read_csv(path)
        df = self._standardize_columns(df)
        df = self._standardize_labels(df)
        df = self._standardize_gender(df)
        df['age_group'] = 'adult'
        df['dataset'] = 'adult'
        
        print(f"Loaded adult data: {len(df)} records from {path.name}")
        self.adult_data = df
        return df

    def load_child_data(self) -> Optional[pd.DataFrame]:
        """Load child ASD screening dataset.

        Returns:
            Processed child DataFrame or None
        """
        patterns = [
            'Autism-Child-Data*.csv',
            'autism*child*.csv',
            'child*.csv',
            'ASD_child*.csv',
        ]
        
        path = self._find_file(patterns)
        if path is None:
            return None

        df = pd.read_csv(path)
        df = self._standardize_columns(df)
        df = self._standardize_labels(df)
        df = self._standardize_gender(df)
        df['age_group'] = 'child'
        df['dataset'] = 'child'
        
        print(f"Loaded child data: {len(df)} records from {path.name}")
        self.child_data = df
        return df

    def load_toddler_data(self) -> Optional[pd.DataFrame]:
        """Load toddler ASD screening dataset.

        Returns:
            Processed toddler DataFrame or None
        """
        patterns = [
            'Toddler*Autism*.csv',
            'toddler*.csv',
            'autism*toddler*.csv',
            'Q-CHAT*.csv',
            'qchat*.csv',
        ]
        
        path = self._find_file(patterns)
        if path is None:
            return None

        df = pd.read_csv(path)
        df = self._standardize_columns(df)
        df = self._standardize_labels(df)
        df = self._standardize_gender(df)
        df['age_group'] = 'toddler'
        df['dataset'] = 'toddler'
        
        print(f"Loaded toddler data: {len(df)} records from {path.name}")
        self.toddler_data = df
        return df

    def load(self) -> bool:
        """Load all available ASD datasets.

        Returns:
            True if at least one dataset loaded successfully
        """
        self.load_adult_data()
        self.load_child_data()
        self.load_toddler_data()
        
        # Combine available datasets
        dfs = [df for df in [self.adult_data, self.child_data, self.toddler_data] if df is not None]
        
        if not dfs:
            warnings.warn(f"No ASD datasets found in {self.data_dir}")
            return False

        # Find common columns
        common_cols = set(dfs[0].columns)
        for df in dfs[1:]:
            common_cols &= set(df.columns)
        
        # Combine on common columns
        self.combined_data = pd.concat(
            [df[list(common_cols)] for df in dfs],
            ignore_index=True
        )
        
        print(f"\nCombined data: {len(self.combined_data)} total records")
        print(f"  Age groups: {self.combined_data['age_group'].value_counts().to_dict()}")
        
        return True

    def get_prevalence_statistics(self) -> Dict[str, Dict]:
        """Calculate ASD prevalence statistics.

        Returns:
            Dict with prevalence by age group, gender, etc.
        """
        if self.combined_data is None:
            raise ValueError("Data not loaded. Call load() first.")

        df = self.combined_data.dropna(subset=['asd_label'])
        
        stats = {
            'overall': {
                'n_total': int(len(df)),
                'n_asd': int(df['asd_label'].sum()),
                'prevalence': float(df['asd_label'].mean()),
            },
            'by_age_group': {},
            'by_gender': {},
        }
        
        # By age group
        for group in df['age_group'].dropna().unique():
            group_df = df[df['age_group'] == group]
            stats['by_age_group'][group] = {
                'n_total': int(len(group_df)),
                'n_asd': int(group_df['asd_label'].sum()),
                'prevalence': float(group_df['asd_label'].mean()),
            }
        
        # By gender
        if 'gender' in df.columns:
            for gender in df['gender'].dropna().unique():
                gender_df = df[df['gender'] == gender]
                stats['by_gender'][gender] = {
                    'n_total': int(len(gender_df)),
                    'n_asd': int(gender_df['asd_label'].sum()),
                    'prevalence': float(gender_df['asd_label'].mean()),
                }
        
        self._prevalence_stats = stats
        return stats

    def get_screening_score_distribution(self) -> Dict[str, Dict]:
        """Get distribution of screening scores by ASD status.

        Returns:
            Dict with score statistics for ASD+ and ASD- groups
        """
        if self.combined_data is None:
            raise ValueError("Data not loaded. Call load() first.")

        df = self.combined_data.dropna(subset=['asd_label'])
        
        if 'screening_score' not in df.columns:
            return {'error': 'screening_score column not available'}

        df = df.dropna(subset=['screening_score'])
        
        asd_pos = df[df['asd_label'] == 1]['screening_score']
        asd_neg = df[df['asd_label'] == 0]['screening_score']
        
        return {
            'asd_positive': {
                'mean': float(asd_pos.mean()),
                'std': float(asd_pos.std()),
                'median': float(asd_pos.median()),
                'n': int(len(asd_pos)),
            },
            'asd_negative': {
                'mean': float(asd_neg.mean()),
                'std': float(asd_neg.std()),
                'median': float(asd_neg.median()),
                'n': int(len(asd_neg)),
            },
            'effect_size': float((asd_pos.mean() - asd_neg.mean()) / df['screening_score'].std())
            if df['screening_score'].std() > 0 else 0,
        }

    def get_question_importance(self) -> Dict[str, Dict]:
        """Analyze which screening questions best discriminate ASD.

        Returns:
            Dict mapping question to discrimination statistics
        """
        if self.combined_data is None:
            raise ValueError("Data not loaded. Call load() first.")

        df = self.combined_data.dropna(subset=['asd_label'])
        
        question_cols = [c for c in df.columns if c.startswith('q') and c[1:].isdigit()]
        
        if not question_cols:
            return {'error': 'No question columns found'}

        importance = {}
        for q in question_cols:
            if q not in df.columns:
                continue
            
            q_data = df[[q, 'asd_label']].dropna()
            if len(q_data) < 10:
                continue
            
            # Calculate point-biserial correlation
            asd_pos_mean = q_data[q_data['asd_label'] == 1][q].mean()
            asd_neg_mean = q_data[q_data['asd_label'] == 0][q].mean()
            overall_std = q_data[q].std()
            
            effect_size = (asd_pos_mean - asd_neg_mean) / overall_std if overall_std > 0 else 0
            
            importance[q] = {
                'asd_pos_mean': float(asd_pos_mean),
                'asd_neg_mean': float(asd_neg_mean),
                'effect_size': float(effect_size),
                'domain': AQ10_DOMAINS.get(q, 'unknown'),
            }
        
        # Sort by effect size
        importance = dict(sorted(
            importance.items(),
            key=lambda x: abs(x[1]['effect_size']),
            reverse=True
        ))
        
        return importance

    def get_family_history_impact(self) -> Dict[str, float]:
        """Analyze impact of family history on ASD prevalence.

        Returns:
            Dict with family history statistics
        """
        if self.combined_data is None:
            raise ValueError("Data not loaded. Call load() first.")

        df = self.combined_data.dropna(subset=['asd_label'])
        
        if 'family_history_asd' not in df.columns:
            return {'error': 'family_history_asd column not available'}

        # Standardize family history column
        fh = df['family_history_asd'].astype(str).str.lower().str.strip()
        df = df.copy()
        df['family_history'] = fh.map({
            'yes': 1, 'no': 0, '1': 1, '0': 0, 'true': 1, 'false': 0
        })
        
        df = df.dropna(subset=['family_history'])
        
        with_fh = df[df['family_history'] == 1]['asd_label'].mean()
        without_fh = df[df['family_history'] == 0]['asd_label'].mean()
        
        return {
            'prevalence_with_family_history': float(with_fh),
            'prevalence_without_family_history': float(without_fh),
            'relative_risk': float(with_fh / without_fh) if without_fh > 0 else None,
            'n_with_fh': int((df['family_history'] == 1).sum()),
            'n_without_fh': int((df['family_history'] == 0).sum()),
        }

    def get_validation_splits(self, test_size: float = 0.2, 
                               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get train/test splits for validation.

        Args:
            test_size: Fraction for test set
            random_state: Random seed

        Returns:
            Tuple of (train_df, test_df)
        """
        if self.combined_data is None:
            raise ValueError("Data not loaded. Call load() first.")

        df = self.combined_data.dropna(subset=['asd_label'])
        
        # Stratified split
        from sklearn.model_selection import train_test_split
        
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state,
            stratify=df['asd_label']
        )
        
        return train_df, test_df

    def save_processed(self, output_path: str):
        """Save processed data and statistics to JSON.

        Args:
            output_path: Path to save JSON file
        """
        output = {
            'prevalence_statistics': self.get_prevalence_statistics(),
            'screening_score_distribution': self.get_screening_score_distribution(),
            'question_importance': self.get_question_importance(),
            'family_history_impact': self.get_family_history_impact(),
            'metadata': {
                'sources': [
                    'https://www.kaggle.com/datasets/andrewmvd/autism-screening-on-adults',
                    'https://www.kaggle.com/datasets/fabdelja/autism-screening-for-toddlers',
                ],
                'n_total': len(self.combined_data) if self.combined_data is not None else 0,
                'datasets_loaded': {
                    'adult': self.adult_data is not None,
                    'child': self.child_data is not None,
                    'toddler': self.toddler_data is not None,
                },
            },
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Saved Kaggle ASD processed data to {output_path}")

        # Also save combined data as CSV for model training
        if self.combined_data is not None:
            csv_path = Path(output_path).with_suffix('.csv')
            self.combined_data.to_csv(csv_path, index=False)
            print(f"Saved combined data to {csv_path}")


def main():
    """CLI for testing the loader."""
    import argparse

    parser = argparse.ArgumentParser(description='Load Kaggle ASD screening data')
    parser.add_argument('--data_dir', type=str, default='data/raw/kaggle_asd',
                        help='Directory containing Kaggle ASD CSV files')
    parser.add_argument('--output', type=str, default='data/processed/kaggle_asd_processed.json',
                        help='Output path for processed JSON')
    args = parser.parse_args()

    loader = KaggleASDLoader(args.data_dir)
    if loader.load():
        loader.save_processed(args.output)
        
        # Print summary
        stats = loader.get_prevalence_statistics()
        print(f"\nASD Prevalence Summary:")
        print(f"  Overall: {stats['overall']['prevalence']:.1%} ({stats['overall']['n_asd']}/{stats['overall']['n_total']})")
        for group, s in stats['by_age_group'].items():
            print(f"  {group}: {s['prevalence']:.1%} ({s['n_asd']}/{s['n_total']})")
    else:
        print("\nTo download Kaggle ASD data:")
        print("1. Visit https://www.kaggle.com/datasets/andrewmvd/autism-screening-on-adults")
        print("2. Visit https://www.kaggle.com/datasets/fabdelja/autism-screening-for-toddlers")
        print("3. Download and extract CSV files")
        print(f"4. Place them in {args.data_dir}/")


if __name__ == '__main__':
    main()
