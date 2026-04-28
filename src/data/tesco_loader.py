"""Tesco Grocery Dataset Loader.

Loads transaction data from the Tesco Grocery 1.0 dataset:
    https://www.kaggle.com/datasets/tesco/tesco-grocery-1-0

This dataset contains aggregated grocery purchase data from Tesco (UK),
with area-level statistics on product categories and nutritional content.

We use this to:
1. Cross-validate purchase patterns from Instacart (US vs UK)
2. Extract category-level purchase distributions
3. Provide additional empirical rates for synthetic data calibration

Required files (download from Kaggle):
    - area_data.csv: Area-level aggregated purchase data
    - product_data.csv: Product category information (if available)

Usage:
    loader = TescoLoader('data/raw/tesco/')
    loader.load()
    rates = loader.get_category_rates()
    loader.save_rates('data/processed/tesco_rates.json')
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Mapping from Tesco categories to DevelopMap domains
# Tesco uses different category names than Instacart
TESCO_CATEGORY_TO_DOMAIN = {
    'baby': 'adaptive',
    'baby food': 'feeding',
    'baby care': 'adaptive',
    'baby toiletries': 'adaptive',
    'nappies': 'adaptive',
    'baby milk': 'feeding',
    'toddler food': 'feeding',
    'children snacks': 'feeding',
    'vitamins': 'therapeutic',
    'medicines': 'therapeutic',
    'health': 'therapeutic',
    'sleep aids': 'sleep',
    'herbal tea': 'sleep',
}

# Broader category mappings
TESCO_DEPT_TO_CATEGORY = {
    'baby': 'baby_products',
    'health & beauty': 'personal_care',
    'household': 'household',
    'food cupboard': 'food',
    'fresh food': 'food',
    'frozen': 'food',
    'bakery': 'food',
    'drinks': 'food',
    'pet': 'pets',
}


class TescoLoader:
    """Load and analyze Tesco grocery transaction data."""

    def __init__(self, data_dir: str):
        """Initialize loader.

        Args:
            data_dir: Directory containing Tesco CSV files
        """
        self.data_dir = Path(data_dir)
        self.area_data: Optional[pd.DataFrame] = None
        self.product_data: Optional[pd.DataFrame] = None
        self._category_rates: Optional[Dict] = None

    def load(self) -> bool:
        """Load Tesco data files.

        Returns:
            True if data loaded successfully
        """
        # Try different possible file names
        area_files = [
            'area_data.csv',
            'tesco_area_data.csv',
            'grocery_data.csv',
            'tesco_grocery.csv',
        ]

        area_path = None
        for fname in area_files:
            path = self.data_dir / fname
            if path.exists():
                area_path = path
                break

        if area_path is None:
            # Check for any CSV files
            csv_files = list(self.data_dir.glob('*.csv'))
            if csv_files:
                area_path = csv_files[0]
                print(f"Using {area_path.name} as area data")
            else:
                warnings.warn(f"No CSV files found in {self.data_dir}")
                return False

        self.area_data = pd.read_csv(area_path)
        print(f"Loaded Tesco data: {len(self.area_data)} rows, {len(self.area_data.columns)} columns")
        print(f"Columns: {list(self.area_data.columns)[:10]}...")

        # Try to load product data if available
        product_files = ['product_data.csv', 'products.csv', 'tesco_products.csv']
        for fname in product_files:
            path = self.data_dir / fname
            if path.exists():
                self.product_data = pd.read_csv(path)
                print(f"Loaded product data: {len(self.product_data)} products")
                break

        return True

    def get_category_columns(self) -> List[str]:
        """Identify columns that represent product categories.

        Returns:
            List of column names that appear to be category purchase data
        """
        if self.area_data is None:
            return []

        # Look for columns with category-like names
        category_keywords = [
            'baby', 'food', 'drink', 'health', 'household', 'pet',
            'fresh', 'frozen', 'bakery', 'dairy', 'meat', 'fish',
            'fruit', 'vegetable', 'snack', 'confectionery', 'alcohol',
        ]

        category_cols = []
        for col in self.area_data.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in category_keywords):
                # Check if it's numeric
                if pd.api.types.is_numeric_dtype(self.area_data[col]):
                    category_cols.append(col)

        return category_cols

    def get_category_rates(self) -> Dict[str, Dict]:
        """Calculate purchase rate statistics by category.

        Returns:
            Dict mapping category to rate statistics
        """
        if self.area_data is None:
            raise ValueError("Data not loaded. Call load() first.")

        category_cols = self.get_category_columns()

        if not category_cols:
            # If no category columns found, analyze all numeric columns
            numeric_cols = self.area_data.select_dtypes(include=[np.number]).columns
            category_cols = [c for c in numeric_cols if not c.lower().startswith(('id', 'area', 'population'))]

        rates = {}
        for col in category_cols:
            values = self.area_data[col].dropna()
            if len(values) == 0:
                continue

            # Determine domain mapping
            col_lower = col.lower()
            domain = None
            for key, dom in TESCO_CATEGORY_TO_DOMAIN.items():
                if key in col_lower:
                    domain = dom
                    break

            rates[col] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'median': float(values.median()),
                'min': float(values.min()),
                'max': float(values.max()),
                'n_areas': int(len(values)),
                'domain': domain,
            }

        self._category_rates = rates
        return rates

    def get_baby_category_rates(self) -> Dict[str, Dict]:
        """Get rates specifically for baby-related categories.

        Returns:
            Dict with baby category statistics
        """
        all_rates = self.get_category_rates()

        baby_rates = {}
        for col, stats in all_rates.items():
            col_lower = col.lower()
            if 'baby' in col_lower or 'infant' in col_lower or 'toddler' in col_lower:
                baby_rates[col] = stats

        return baby_rates

    def get_area_demographics(self) -> Dict[str, Dict]:
        """Extract area-level demographic information if available.

        Returns:
            Dict with demographic statistics
        """
        if self.area_data is None:
            raise ValueError("Data not loaded. Call load() first.")

        demo_keywords = ['population', 'income', 'age', 'household', 'deprivation', 'imd']
        demo_cols = []

        for col in self.area_data.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in demo_keywords):
                if pd.api.types.is_numeric_dtype(self.area_data[col]):
                    demo_cols.append(col)

        demographics = {}
        for col in demo_cols:
            values = self.area_data[col].dropna()
            if len(values) > 0:
                demographics[col] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'median': float(values.median()),
                }

        return demographics

    def derive_baseline_rates_for_synthetic(self) -> Dict[str, Dict]:
        """Derive baseline rates to calibrate synthetic data.

        Returns:
            Dict mapping domain to empirical rate parameters
        """
        all_rates = self.get_category_rates()

        domain_rates = {}
        for col, stats in all_rates.items():
            domain = stats.get('domain')
            if domain:
                if domain not in domain_rates:
                    domain_rates[domain] = {
                        'mean_rate': [],
                        'std_rate': [],
                        'n_areas': 0,
                    }
                # Normalize to per-household rate if possible
                rate = stats['mean']
                domain_rates[domain]['mean_rate'].append(rate)
                domain_rates[domain]['std_rate'].append(stats['std'])
                domain_rates[domain]['n_areas'] += stats['n_areas']

        # Aggregate
        for domain in domain_rates:
            rates = domain_rates[domain]
            rates['mean_rate'] = float(np.mean(rates['mean_rate']))
            rates['std_rate'] = float(np.mean(rates['std_rate']))
            rates['source'] = 'tesco_area_data'

        return domain_rates

    def compare_with_instacart(self, instacart_rates: Dict) -> Dict[str, Dict]:
        """Compare Tesco rates with Instacart rates.

        Args:
            instacart_rates: Rates from InstacartLoader

        Returns:
            Dict with comparison statistics
        """
        tesco_domain_rates = self.derive_baseline_rates_for_synthetic()
        instacart_domain_rates = instacart_rates.get('domain_rates_for_synthetic', {})

        comparison = {}
        all_domains = set(tesco_domain_rates.keys()) | set(instacart_domain_rates.keys())

        for domain in all_domains:
            tesco_rate = tesco_domain_rates.get(domain, {}).get('mean_rate')
            instacart_rate = instacart_domain_rates.get(domain, {}).get('mean_rate')

            comparison[domain] = {
                'tesco_rate': tesco_rate,
                'instacart_rate': instacart_rate,
                'ratio': tesco_rate / instacart_rate if tesco_rate and instacart_rate else None,
            }

        return comparison

    def save_rates(self, output_path: str):
        """Save derived rates to JSON.

        Args:
            output_path: Path to save JSON file
        """
        output = {
            'category_rates': self.get_category_rates(),
            'baby_rates': self.get_baby_category_rates(),
            'demographics': self.get_area_demographics(),
            'domain_rates_for_synthetic': self.derive_baseline_rates_for_synthetic(),
            'metadata': {
                'source': 'Tesco Grocery 1.0',
                'url': 'https://www.kaggle.com/datasets/tesco/tesco-grocery-1-0',
                'n_areas': len(self.area_data) if self.area_data is not None else 0,
                'columns': list(self.area_data.columns) if self.area_data is not None else [],
            },
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Saved Tesco rates to {output_path}")


def main():
    """CLI for testing the loader."""
    import argparse

    parser = argparse.ArgumentParser(description='Load Tesco grocery data')
    parser.add_argument('--data_dir', type=str, default='data/raw/tesco',
                        help='Directory containing Tesco CSV files')
    parser.add_argument('--output', type=str, default='data/processed/tesco_rates.json',
                        help='Output path for rates JSON')
    args = parser.parse_args()

    loader = TescoLoader(args.data_dir)
    if loader.load():
        loader.save_rates(args.output)
    else:
        print("\nTo download Tesco data:")
        print("1. Visit https://www.kaggle.com/datasets/tesco/tesco-grocery-1-0")
        print("2. Download and extract CSV files")
        print(f"3. Place them in {args.data_dir}/")


if __name__ == '__main__':
    main()
