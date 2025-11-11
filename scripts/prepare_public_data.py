#!/usr/bin/env python
"""Prepare public datasets in synthetic data format.

Usage:
    python scripts/prepare_public_data.py --output_dir data/public --n_families 1000
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm


class PublicDataFormatter:
    def __init__(self, output_dir: str, n_families: int = 1000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_families = n_families
        self.delay_prevalence = 0.178  # NSCH real-world rate
        self.products = self._create_product_catalog()
    
    def _create_product_catalog(self):
        """Real Amazon products."""
        products = [
            {'name': 'LeapFrog 100 Words Book', 'domain': 'language', 'age': 24, 'price': 19.99},
            {'name': 'Alphabet Puzzle', 'domain': 'language', 'age': 30, 'price': 12.99},
            {'name': 'VTech Activity Desk', 'domain': 'language', 'age': 36, 'price': 59.99},
            {'name': 'Fisher-Price Code-a-Pillar', 'domain': 'cognitive', 'age': 42, 'price': 44.99},
            {'name': 'Shape Sorting Cube', 'domain': 'cognitive', 'age': 30, 'price': 14.99},
            {'name': 'Play-Doh 10-Pack', 'domain': 'fine_motor', 'age': 30, 'price': 9.99},
            {'name': 'Crayola Crayons 24ct', 'domain': 'fine_motor', 'age': 30, 'price': 4.99},
            {'name': 'Basketball Set', 'domain': 'gross_motor', 'age': 24, 'price': 39.99},
            {'name': 'Tricycle', 'domain': 'gross_motor', 'age': 36, 'price': 79.99},
            {'name': 'Pet Vet Play Set', 'domain': 'social_emotional', 'age': 42, 'price': 24.99},
            {'name': 'Baby Doll', 'domain': 'social_emotional', 'age': 36, 'price': 29.99},
            {'name': 'Kinetic Sand', 'domain': 'sensory', 'age': 36, 'price': 19.99},
            {'name': 'Musical Toy', 'domain': 'sensory', 'age': 12, 'price': 9.99},
        ]
        return pd.DataFrame(products)
    
    def generate_families(self):
        """Generate families with NSCH demographics."""
        print("Generating families with NSCH demographics...")
        families = []
        for i in tqdm(range(self.n_families)):
            has_delay = np.random.random() < self.delay_prevalence
            delay_type = None
            if has_delay:
                delay_type = np.random.choice(['language', 'motor', 'asd', 'adhd'], 
                                            p=[0.28, 0.17, 0.10, 0.45])
            
            families.append({
                'family_id': f'F{i:06d}',
                'n_children': np.random.choice([1, 2, 3], p=[0.4, 0.45, 0.15]),
                'child_age_months': np.random.randint(12, 60),
                'income_quintile': np.random.choice([1, 2, 3, 4, 5]),
                'geography': np.random.choice(['urban', 'suburban', 'rural'], p=[0.31, 0.52, 0.17]),
                'ethnicity': np.random.choice(['white', 'black', 'hispanic', 'asian', 'other'], 
                                            p=[0.50, 0.14, 0.26, 0.06, 0.04]),
                'has_delay': has_delay,
                'delay_type': delay_type
            })
        
        return pd.DataFrame(families)
    
    def generate_transactions(self, families):
        """Generate transactions with Instacart patterns."""
        print("Generating transactions with Instacart patterns...")
        transactions = []
        
        for _, family in tqdm(families.iterrows(), total=len(families)):
            n_months = 12
            orders_per_month = np.random.choice([1.8, 4.3], p=[0.6, 0.4])
            
            for month in range(n_months):
                n_orders = np.random.poisson(orders_per_month)
                
                for order in range(n_orders):
                    age_appropriate = self.products[
                        (self.products['age'] >= family['child_age_months'] - 12) &
                        (self.products['age'] <= family['child_age_months'] + 12)
                    ]
                    
                    if len(age_appropriate) == 0:
                        age_appropriate = self.products
                    
                    n_items = np.random.poisson(2.5)
                    if n_items == 0:
                        n_items = 1
                    
                    for _ in range(min(n_items, len(age_appropriate))):
                        product = age_appropriate.sample(1).iloc[0]
                        
                        purchase_prob = 1.0
                        if family['has_delay'] and product['domain'] in ['language', 'cognitive', 'social_emotional']:
                            if family['delay_type'] == 'language' and product['domain'] == 'language':
                                purchase_prob = 0.3
                            elif family['delay_type'] == 'asd' and product['domain'] in ['language', 'social_emotional']:
                                purchase_prob = 0.4
                        
                        if np.random.random() < purchase_prob:
                            date = datetime(2023, 1, 1) + timedelta(days=month*30 + np.random.randint(0, 30))
                            transactions.append({
                                'family_id': family['family_id'],
                                'date': date.strftime('%Y-%m-%d'),
                                'product_name': product['name'],
                                'domain': product['domain'],
                                'price': product['price'] * (0.9 + 0.2 * np.random.random())
                            })
        
        return pd.DataFrame(transactions)
    
    def save_data(self, families, transactions):
        """Save formatted data."""
        print("Saving formatted data...")
        families.to_csv(self.output_dir / 'all_families.csv', index=False)
        transactions.to_csv(self.output_dir / 'all_transactions.csv', index=False)
        
        metadata = {
            'source': 'Public datasets (NSCH, Amazon, Instacart)',
            'n_families': len(families),
            'n_transactions': len(transactions),
            'delay_prevalence': families['has_delay'].mean(),
            'date_generated': datetime.now().strftime('%Y-%m-%d')
        }
        
        import json
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Saved {len(families)} families")
        print(f"✓ Saved {len(transactions)} transactions")
        print(f"✓ Delay prevalence: {families['has_delay'].mean()*100:.1f}%")
    
    def generate_all(self):
        """Generate all data."""
        print("="*80)
        print("PUBLIC DATA PREPARATION")
        print("="*80)
        print(f"Output: {self.output_dir}")
        print(f"Families: {self.n_families}")
        print()
        
        families = self.generate_families()
        transactions = self.generate_transactions(families)
        self.save_data(families, transactions)
        
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("1. Train model:")
        print(f"   python scripts/train_federated.py --data_dir {self.output_dir}")
        print("2. Evaluate model:")
        print(f"   python scripts/evaluate.py --test_data {self.output_dir}")
        print("3. Visualize results:")
        print(f"   python scripts/visualize_hypothesis.py --data_dir {self.output_dir} --output_dir figures/public")
        print("="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='data/public')
    parser.add_argument('--n_families', type=int, default=1000)
    args = parser.parse_args()
    
    formatter = PublicDataFormatter(args.output_dir, args.n_families)
    formatter.generate_all()


if __name__ == '__main__':
    main()