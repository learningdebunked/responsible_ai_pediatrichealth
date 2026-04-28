"""Instacart Market Basket Analysis Data Loader.

Loads transaction data from the Instacart Market Basket Analysis dataset:
    https://www.kaggle.com/c/instacart-market-basket-analysis

This dataset contains 3M+ grocery orders from 200K+ users, with:
- Order sequences per user (temporal patterns)
- Product details with department/aisle categorization
- Reorder behavior

We use this to:
1. Extract real purchase rate distributions by product category
2. Derive age-appropriate product baskets (baby products, etc.)
3. Replace hardcoded synthetic multipliers with empirical rates

Required files (download from Kaggle):
    - orders.csv: Order metadata (user_id, order_id, order_dow, order_hour_of_day, days_since_prior_order)
    - order_products__prior.csv: Products in prior orders
    - order_products__train.csv: Products in training orders
    - products.csv: Product names and categories
    - departments.csv: Department names
    - aisles.csv: Aisle names

Usage:
    loader = InstacartLoader('data/raw/instacart/')
    loader.load()
    rates = loader.get_category_purchase_rates()
    loader.save_purchase_rates('data/processed/instacart_rates.json')
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Mapping from Instacart aisles to DevelopMap domains
AISLE_TO_DOMAIN = {
    # Baby-related aisles
    'baby food formula': 'feeding',
    'baby bath body care': 'adaptive',
    'baby accessories': 'adaptive',
    'diapers wipes': 'adaptive',
    # Food/feeding related
    'baby food': 'feeding',
    'toddler food': 'feeding',
    'kids snacks': 'feeding',
    # Could indicate sensory preferences
    'specialty cheeses': 'sensory',
    'gluten free': 'sensory',
    'organic': 'sensory',
    # Sleep-related
    'tea': 'sleep',
    'sleep aids': 'sleep',
    # General categories (less specific mapping)
    'vitamins supplements': 'therapeutic',
    'first aid': 'therapeutic',
}

# Mapping from Instacart departments to broader categories
DEPARTMENT_TO_CATEGORY = {
    'babies': 'baby_products',
    'snacks': 'food',
    'beverages': 'food',
    'frozen': 'food',
    'dairy eggs': 'food',
    'produce': 'food',
    'canned goods': 'food',
    'dry goods pasta': 'food',
    'breakfast': 'food',
    'bakery': 'food',
    'meat seafood': 'food',
    'deli': 'food',
    'pantry': 'food',
    'household': 'household',
    'personal care': 'personal_care',
    'pets': 'pets',
    'alcohol': 'alcohol',
    'international': 'food',
    'bulk': 'food',
    'other': 'other',
    'missing': 'other',
}


class InstacartLoader:
    """Load and analyze Instacart transaction data."""

    def __init__(self, data_dir: str, sample_frac: float = 1.0):
        """Initialize loader.

        Args:
            data_dir: Directory containing Instacart CSV files
            sample_frac: Fraction of users to sample (for faster processing)
        """
        self.data_dir = Path(data_dir)
        self.sample_frac = sample_frac

        self.orders: Optional[pd.DataFrame] = None
        self.order_products: Optional[pd.DataFrame] = None
        self.products: Optional[pd.DataFrame] = None
        self.departments: Optional[pd.DataFrame] = None
        self.aisles: Optional[pd.DataFrame] = None

        self._purchase_rates: Optional[Dict] = None
        self._user_patterns: Optional[pd.DataFrame] = None

    def load(self) -> bool:
        """Load all Instacart data files.

        Returns:
            True if all required files loaded successfully
        """
        required_files = {
            'orders': 'orders.csv',
            'products': 'products.csv',
            'departments': 'departments.csv',
            'aisles': 'aisles.csv',
        }

        # At least one of these is needed
        order_product_files = [
            'order_products__prior.csv',
            'order_products__train.csv',
        ]

        # Check required files
        for name, filename in required_files.items():
            path = self.data_dir / filename
            if not path.exists():
                warnings.warn(f"Required file not found: {path}")
                return False

        # Load reference tables
        self.departments = pd.read_csv(self.data_dir / 'departments.csv')
        self.aisles = pd.read_csv(self.data_dir / 'aisles.csv')
        self.products = pd.read_csv(self.data_dir / 'products.csv')

        # Merge product info
        self.products = self.products.merge(
            self.aisles, on='aisle_id', how='left'
        ).merge(
            self.departments, on='department_id', how='left'
        )

        # Load orders
        self.orders = pd.read_csv(self.data_dir / 'orders.csv')

        # Sample users if requested
        if self.sample_frac < 1.0:
            sampled_users = self.orders['user_id'].drop_duplicates().sample(
                frac=self.sample_frac, random_state=42
            )
            self.orders = self.orders[self.orders['user_id'].isin(sampled_users)]

        # Load order products (try prior first, then train)
        order_products_list = []
        for filename in order_product_files:
            path = self.data_dir / filename
            if path.exists():
                df = pd.read_csv(path)
                # Filter to sampled orders
                df = df[df['order_id'].isin(self.orders['order_id'])]
                order_products_list.append(df)

        if not order_products_list:
            warnings.warn("No order_products files found")
            return False

        self.order_products = pd.concat(order_products_list, ignore_index=True)

        # Merge product info into order_products
        self.order_products = self.order_products.merge(
            self.products[['product_id', 'product_name', 'aisle', 'department']],
            on='product_id',
            how='left'
        )

        print(f"Loaded Instacart data:")
        print(f"  Users: {self.orders['user_id'].nunique():,}")
        print(f"  Orders: {len(self.orders):,}")
        print(f"  Order-products: {len(self.order_products):,}")
        print(f"  Products: {len(self.products):,}")
        print(f"  Departments: {len(self.departments)}")
        print(f"  Aisles: {len(self.aisles)}")

        return True

    def get_baby_product_users(self) -> pd.DataFrame:
        """Identify users who purchase baby products.

        Returns:
            DataFrame with user_id and baby product purchase statistics
        """
        if self.order_products is None:
            raise ValueError("Data not loaded. Call load() first.")

        # Filter to baby department
        baby_orders = self.order_products[
            self.order_products['department'] == 'babies'
        ]

        # Aggregate by user
        user_baby_stats = baby_orders.merge(
            self.orders[['order_id', 'user_id']], on='order_id'
        ).groupby('user_id').agg(
            n_baby_orders=('order_id', 'nunique'),
            n_baby_products=('product_id', 'count'),
            unique_baby_products=('product_id', 'nunique'),
        ).reset_index()

        # Add total order count per user
        total_orders = self.orders.groupby('user_id').size().reset_index(name='total_orders')
        user_baby_stats = user_baby_stats.merge(total_orders, on='user_id')

        # Calculate baby product rate
        user_baby_stats['baby_order_rate'] = (
            user_baby_stats['n_baby_orders'] / user_baby_stats['total_orders']
        )

        return user_baby_stats

    def get_category_purchase_rates(self) -> Dict[str, Dict]:
        """Calculate purchase rates by category.

        Returns:
            Dict mapping category to rate statistics
        """
        if self.order_products is None:
            raise ValueError("Data not loaded. Call load() first.")

        # Map departments to categories
        self.order_products['category'] = self.order_products['department'].map(
            DEPARTMENT_TO_CATEGORY
        ).fillna('other')

        # Calculate per-user rates
        user_order_counts = self.orders.groupby('user_id').size()

        category_rates = {}
        for category in self.order_products['category'].unique():
            cat_orders = self.order_products[
                self.order_products['category'] == category
            ].merge(
                self.orders[['order_id', 'user_id']], on='order_id'
            )

            user_cat_orders = cat_orders.groupby('user_id')['order_id'].nunique()

            # Calculate rate per user
            rates = user_cat_orders / user_order_counts
            rates = rates.dropna()

            category_rates[category] = {
                'mean_rate': float(rates.mean()),
                'std_rate': float(rates.std()),
                'median_rate': float(rates.median()),
                'p25_rate': float(rates.quantile(0.25)),
                'p75_rate': float(rates.quantile(0.75)),
                'n_users': int(len(rates)),
            }

        self._purchase_rates = category_rates
        return category_rates

    def get_aisle_purchase_rates(self) -> Dict[str, Dict]:
        """Calculate purchase rates by aisle (finer granularity).

        Returns:
            Dict mapping aisle to rate statistics
        """
        if self.order_products is None:
            raise ValueError("Data not loaded. Call load() first.")

        user_order_counts = self.orders.groupby('user_id').size()

        aisle_rates = {}
        for aisle in self.order_products['aisle'].dropna().unique():
            aisle_orders = self.order_products[
                self.order_products['aisle'] == aisle
            ].merge(
                self.orders[['order_id', 'user_id']], on='order_id'
            )

            user_aisle_orders = aisle_orders.groupby('user_id')['order_id'].nunique()
            rates = user_aisle_orders / user_order_counts
            rates = rates.dropna()

            if len(rates) < 10:  # Skip rare aisles
                continue

            aisle_rates[aisle] = {
                'mean_rate': float(rates.mean()),
                'std_rate': float(rates.std()),
                'median_rate': float(rates.median()),
                'n_users': int(len(rates)),
                'domain': AISLE_TO_DOMAIN.get(aisle.lower(), None),
            }

        return aisle_rates

    def get_baby_aisle_rates(self) -> Dict[str, Dict]:
        """Get purchase rates specifically for baby-related aisles.

        Returns:
            Dict mapping baby aisle to rate statistics among baby-product users
        """
        if self.order_products is None:
            raise ValueError("Data not loaded. Call load() first.")

        # Get users who buy baby products
        baby_users = self.get_baby_product_users()
        baby_user_ids = set(baby_users['user_id'])

        # Filter to baby department aisles
        baby_aisles = self.order_products[
            self.order_products['department'] == 'babies'
        ]['aisle'].unique()

        # Calculate rates among baby-product users
        baby_user_orders = self.orders[
            self.orders['user_id'].isin(baby_user_ids)
        ]
        user_order_counts = baby_user_orders.groupby('user_id').size()

        aisle_rates = {}
        for aisle in baby_aisles:
            aisle_orders = self.order_products[
                (self.order_products['aisle'] == aisle)
            ].merge(
                baby_user_orders[['order_id', 'user_id']], on='order_id'
            )

            user_aisle_orders = aisle_orders.groupby('user_id')['order_id'].nunique()
            rates = user_aisle_orders / user_order_counts
            rates = rates.dropna()

            aisle_rates[aisle] = {
                'mean_rate': float(rates.mean()),
                'std_rate': float(rates.std()),
                'median_rate': float(rates.median()),
                'n_users': int(len(rates)),
                'domain': AISLE_TO_DOMAIN.get(aisle.lower(), 'adaptive'),
            }

        return aisle_rates

    def get_temporal_patterns(self) -> Dict[str, Dict]:
        """Analyze temporal purchase patterns.

        Returns:
            Dict with day-of-week and hour-of-day patterns
        """
        if self.orders is None:
            raise ValueError("Data not loaded. Call load() first.")

        # Day of week patterns
        dow_counts = self.orders['order_dow'].value_counts().sort_index()
        dow_total = dow_counts.sum()

        # Hour of day patterns
        hod_counts = self.orders['order_hour_of_day'].value_counts().sort_index()
        hod_total = hod_counts.sum()

        # Days since prior order distribution
        days_since = self.orders['days_since_prior_order'].dropna()

        return {
            'day_of_week': {
                'distribution': (dow_counts / dow_total).to_dict(),
                'peak_day': int(dow_counts.idxmax()),
            },
            'hour_of_day': {
                'distribution': (hod_counts / hod_total).to_dict(),
                'peak_hour': int(hod_counts.idxmax()),
            },
            'days_between_orders': {
                'mean': float(days_since.mean()),
                'median': float(days_since.median()),
                'std': float(days_since.std()),
            },
        }

    def get_reorder_rates(self) -> Dict[str, float]:
        """Calculate product reorder rates.

        Returns:
            Dict with reorder statistics
        """
        if self.order_products is None:
            raise ValueError("Data not loaded. Call load() first.")

        if 'reordered' not in self.order_products.columns:
            return {'reorder_rate': None, 'note': 'reordered column not available'}

        reorder_rate = self.order_products['reordered'].mean()

        # By department
        dept_reorder = self.order_products.groupby('department')['reordered'].mean()

        return {
            'overall_reorder_rate': float(reorder_rate),
            'by_department': dept_reorder.to_dict(),
        }

    def derive_baseline_rates_for_synthetic(self) -> Dict[str, Dict]:
        """Derive baseline purchase rates to replace hardcoded synthetic values.

        Returns:
            Dict mapping domain to empirical rate parameters
        """
        baby_rates = self.get_baby_aisle_rates()
        category_rates = self.get_category_purchase_rates()

        # Map to DevelopMap domains
        domain_rates = {}

        for aisle, stats in baby_rates.items():
            domain = stats.get('domain')
            if domain:
                if domain not in domain_rates:
                    domain_rates[domain] = {
                        'mean_rate': [],
                        'std_rate': [],
                        'n_users': 0,
                    }
                domain_rates[domain]['mean_rate'].append(stats['mean_rate'])
                domain_rates[domain]['std_rate'].append(stats['std_rate'])
                domain_rates[domain]['n_users'] += stats['n_users']

        # Aggregate
        for domain in domain_rates:
            rates = domain_rates[domain]
            rates['mean_rate'] = float(np.mean(rates['mean_rate']))
            rates['std_rate'] = float(np.mean(rates['std_rate']))
            rates['source'] = 'instacart_baby_aisles'

        # Add general category rates for domains not covered
        if 'food' in category_rates:
            if 'feeding' not in domain_rates:
                domain_rates['feeding'] = {
                    'mean_rate': category_rates['food']['mean_rate'] * 0.1,  # Subset
                    'std_rate': category_rates['food']['std_rate'] * 0.1,
                    'n_users': category_rates['food']['n_users'],
                    'source': 'instacart_food_category_scaled',
                }

        return domain_rates

    def save_purchase_rates(self, output_path: str):
        """Save derived purchase rates to JSON.

        Args:
            output_path: Path to save JSON file
        """
        output = {
            'category_rates': self.get_category_purchase_rates(),
            'baby_aisle_rates': self.get_baby_aisle_rates(),
            'temporal_patterns': self.get_temporal_patterns(),
            'reorder_rates': self.get_reorder_rates(),
            'domain_rates_for_synthetic': self.derive_baseline_rates_for_synthetic(),
            'metadata': {
                'source': 'Instacart Market Basket Analysis',
                'url': 'https://www.kaggle.com/c/instacart-market-basket-analysis',
                'n_users': int(self.orders['user_id'].nunique()) if self.orders is not None else 0,
                'n_orders': len(self.orders) if self.orders is not None else 0,
                'sample_frac': self.sample_frac,
            },
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Saved Instacart rates to {output_path}")


def main():
    """CLI for testing the loader."""
    import argparse

    parser = argparse.ArgumentParser(description='Load Instacart data')
    parser.add_argument('--data_dir', type=str, default='data/raw/instacart',
                        help='Directory containing Instacart CSV files')
    parser.add_argument('--output', type=str, default='data/processed/instacart_rates.json',
                        help='Output path for rates JSON')
    parser.add_argument('--sample', type=float, default=0.1,
                        help='Fraction of users to sample (default 0.1 for speed)')
    args = parser.parse_args()

    loader = InstacartLoader(args.data_dir, sample_frac=args.sample)
    if loader.load():
        loader.save_purchase_rates(args.output)
    else:
        print("\nTo download Instacart data:")
        print("1. Visit https://www.kaggle.com/c/instacart-market-basket-analysis/data")
        print("2. Download and extract all CSV files")
        print(f"3. Place them in {args.data_dir}/")


if __name__ == '__main__':
    main()
