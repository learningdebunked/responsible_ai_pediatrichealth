#!/usr/bin/env python
"""Download and prepare public datasets for RetailHealth.

This script does NOT generate any synthetic data. It:
  1. Prints download instructions with exact URLs for each dataset
  2. Verifies that required files exist in data/raw/
  3. Calls each real data loader to parse and validate
  4. Saves processed outputs to data/processed/
  5. Prints a summary report

Usage:
    # First, manually download datasets into data/raw/ (see instructions below)
    python scripts/download_public_data.py --raw_dir data/raw --output_dir data/processed
"""

import argparse
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Dataset download instructions ────────────────────────────────────────────

DATASETS = {
    'nsch': {
        'name': 'National Survey of Children\'s Health (NSCH) 2022',
        'url': 'https://www.census.gov/programs-surveys/nsch/data/datasets.html',
        'files': ['nsch_topical.csv'],
        'instructions': (
            '1. Visit the URL above\n'
            '2. Select the most recent year (2022 or newer)\n'
            '3. Download the "Topical" CSV data file\n'
            '4. Place the CSV in data/raw/ as nsch_topical.csv'
        ),
    },
    'ce': {
        'name': 'Consumer Expenditure Survey (CE) Interview PUMD',
        'url': 'https://www.bls.gov/cex/pumd_data.htm',
        'files': ['ce/'],  # directory with fmli*.csv and mtbi*.csv
        'instructions': (
            '1. Visit the URL above\n'
            '2. Download the Interview Survey PUMD (most recent year)\n'
            '3. Extract the archive\n'
            '4. Place the extracted folder in data/raw/ce/\n'
            '   (should contain fmli*.csv and optionally mtbi*.csv)'
        ),
    },
    'amazon': {
        'name': 'Amazon Reviews 2023 Product Metadata',
        'url': 'https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023',
        'files': ['amazon/'],  # directory with meta_*.jsonl files
        'instructions': (
            '1. Visit the URL above\n'
            '2. Download metadata for Baby_Products and Toys_and_Games\n'
            '3. Place JSONL files in data/raw/amazon/\n'
            '   (e.g., meta_Baby_Products.jsonl, meta_Toys_and_Games.jsonl)'
        ),
    },
    'psid': {
        'name': 'PSID-CDS (Panel Study of Income Dynamics — Child Development Supplement)',
        'url': 'https://psidonline.isr.umich.edu/cds/',
        'files': ['psid_family.csv', 'psid_cds.csv'],
        'instructions': (
            '1. Register (free) at https://simba.isr.umich.edu/\n'
            '2. Visit the CDS page at the URL above\n'
            '3. Download CDS-2019/2020 child assessment file\n'
            '4. Download PSID core 2019 family file\n'
            '5. Place in data/raw/ as psid_family.csv and psid_cds.csv'
        ),
    },
}


def print_download_instructions():
    """Print download instructions for all required datasets."""
    print('\n' + '=' * 80)
    print('DATASET DOWNLOAD INSTRUCTIONS')
    print('=' * 80)
    print('\nThe following datasets must be manually downloaded and placed in data/raw/')
    print('before running this script.\n')

    for key, ds in DATASETS.items():
        print(f'── {ds["name"]} ──')
        print(f'   URL:   {ds["url"]}')
        print(f'   Files: {", ".join(ds["files"])}')
        print(f'   Steps:')
        for line in ds['instructions'].split('\n'):
            print(f'   {line}')
        print()

    print('=' * 80)


def check_data_availability(raw_dir: Path) -> dict:
    """Check which datasets are available in raw_dir.

    Returns:
        Dict mapping dataset key to (available: bool, path_or_msg: str)
    """
    status = {}

    # NSCH
    nsch_path = raw_dir / 'nsch_topical.csv'
    nsch_alt = list(raw_dir.glob('nsch*.csv'))
    if nsch_path.exists():
        status['nsch'] = (True, str(nsch_path))
    elif nsch_alt:
        status['nsch'] = (True, str(nsch_alt[0]))
    else:
        status['nsch'] = (False, 'nsch_topical.csv not found')

    # CE
    ce_dir = raw_dir / 'ce'
    fmli_files = list(ce_dir.glob('fmli*.csv')) if ce_dir.exists() else []
    if fmli_files:
        status['ce'] = (True, str(ce_dir))
    else:
        status['ce'] = (False, 'ce/ directory with fmli*.csv not found')

    # Amazon
    amazon_dir = raw_dir / 'amazon'
    meta_files = list(amazon_dir.glob('meta_*.jsonl*')) if amazon_dir.exists() else []
    if meta_files:
        status['amazon'] = (True, str(amazon_dir))
    else:
        status['amazon'] = (False, 'amazon/ directory with meta_*.jsonl not found')

    # PSID
    psid_family = raw_dir / 'psid_family.csv'
    psid_cds = raw_dir / 'psid_cds.csv'
    psid_fam_alt = list(raw_dir.glob('psid*fam*.csv')) + list(raw_dir.glob('fam*er*.csv'))
    psid_cds_alt = list(raw_dir.glob('psid*cds*.csv')) + list(raw_dir.glob('CDS*.csv'))

    fam_path = psid_family if psid_family.exists() else (psid_fam_alt[0] if psid_fam_alt else None)
    cds_path = psid_cds if psid_cds.exists() else (psid_cds_alt[0] if psid_cds_alt else None)

    if fam_path and cds_path:
        status['psid'] = (True, f'{fam_path}, {cds_path}')
    else:
        missing = []
        if not fam_path:
            missing.append('psid_family.csv')
        if not cds_path:
            missing.append('psid_cds.csv')
        status['psid'] = (False, f'Missing: {", ".join(missing)}')

    return status


def process_nsch(raw_dir: Path, output_dir: Path, status: dict):
    """Load and process NSCH data."""
    if not status['nsch'][0]:
        print(f'\n⚠ Skipping NSCH: {status["nsch"][1]}')
        return

    from src.data.nsch_loader import NSCHLoader

    nsch_path = status['nsch'][1]
    loader = NSCHLoader(nsch_path)
    df = loader.load()
    df.to_csv(output_dir / 'nsch_processed.csv', index=False)

    # Save prevalence rates for config update
    rates = loader.get_delay_prevalence_rates()
    with open(output_dir / 'nsch_prevalence_rates.json', 'w') as f:
        json.dump(rates, f, indent=2)

    # Save demographic prevalence
    demo = loader.get_prevalence_by_demographic()
    with open(output_dir / 'nsch_demographic_prevalence.json', 'w') as f:
        json.dump(demo, f, indent=2)

    # Save diagnosis age distributions
    for delay_type in ['language', 'motor', 'asd', 'adhd']:
        ages = loader.get_diagnosis_age_distribution(delay_type)
        if len(ages) > 0:
            pd.DataFrame({'diagnosis_age_months': ages}).to_csv(
                output_dir / f'nsch_diagnosis_age_{delay_type}.csv', index=False
            )

    print(f'✓ NSCH processed → {output_dir}/nsch_*.csv/json')


def process_ce(raw_dir: Path, output_dir: Path, status: dict):
    """Load and process Consumer Expenditure Survey data."""
    if not status['ce'][0]:
        print(f'\n⚠ Skipping CE: {status["ce"][1]}')
        return

    from src.data.ce_loader import CELoader

    ce_dir = status['ce'][1]
    loader = CELoader(ce_dir)
    loader.load()
    loader.save_baseline_rates(str(output_dir / 'ce_baseline_rates.json'))

    # Save income quintile spending
    income_spending = loader.get_spending_by_income_quintile()
    with open(output_dir / 'ce_spending_by_income.json', 'w') as f:
        json.dump({str(k): v for k, v in income_spending.items()}, f, indent=2)

    print(f'✓ CE processed → {output_dir}/ce_*.json')


def process_amazon(raw_dir: Path, output_dir: Path, status: dict):
    """Load and classify Amazon product data."""
    if not status['amazon'][0]:
        print(f'\n⚠ Skipping Amazon: {status["amazon"][1]}')
        return

    from src.data.amazon_product_loader import AmazonProductLoader

    amazon_dir = status['amazon'][1]
    loader = AmazonProductLoader(amazon_dir)
    report = loader.load_and_classify()
    loader.save_report(str(output_dir / 'amazon_classification_report.json'), report)

    print(f'✓ Amazon processed → {output_dir}/amazon_*.json')


def process_psid(raw_dir: Path, output_dir: Path, status: dict):
    """Load and process PSID-CDS data."""
    if not status['psid'][0]:
        print(f'\n⚠ Skipping PSID-CDS: {status["psid"][1]}')
        return

    from src.data.psid_loader import PSIDLoader

    paths = status['psid'][1].split(', ')
    fam_path, cds_path = paths[0], paths[1]
    loader = PSIDLoader(fam_path, cds_path)
    loader.load()

    # Run correlation analysis
    results = loader.test_expenditure_outcome_correlation()
    loader.save_results(str(output_dir / 'psid_correlation_results.json'), results)

    print(f'✓ PSID-CDS processed → {output_dir}/psid_*.json')


def print_summary(status: dict, output_dir: Path):
    """Print processing summary."""
    print('\n' + '=' * 80)
    print('PROCESSING SUMMARY')
    print('=' * 80)

    for key, (available, msg) in status.items():
        symbol = '✓' if available else '✗'
        name = DATASETS[key]['name']
        print(f'  {symbol} {name}: {"processed" if available else msg}')

    n_available = sum(1 for avail, _ in status.values() if avail)
    n_total = len(status)
    print(f'\n  {n_available}/{n_total} datasets processed')

    if n_available < n_total:
        print('\n  To process remaining datasets, download them and re-run this script.')
        print('  Run with --help-downloads for detailed instructions.')

    print(f'\n  Processed data saved to: {output_dir}/')
    print('=' * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Process public datasets for RetailHealth validation'
    )
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                        help='Directory containing raw downloaded datasets')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--help-downloads', action='store_true',
                        help='Print download instructions and exit')
    args = parser.parse_args()

    if args.help_downloads:
        print_download_instructions()
        return

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        print(f'Created {raw_dir}/ — download datasets there first.')
        print_download_instructions()
        return

    print('\n' + '=' * 80)
    print('RETAILHEALTH PUBLIC DATA PROCESSING PIPELINE')
    print('=' * 80)

    # Check data availability
    status = check_data_availability(raw_dir)
    print('\nData availability:')
    for key, (available, msg) in status.items():
        symbol = '✓' if available else '✗'
        print(f'  {symbol} {DATASETS[key]["name"]}: {msg if not available else "found"}')

    if not any(avail for avail, _ in status.values()):
        print('\n⚠ No datasets found. Download them first:')
        print_download_instructions()
        return

    # Process available datasets
    process_nsch(raw_dir, output_dir, status)
    process_ce(raw_dir, output_dir, status)
    process_amazon(raw_dir, output_dir, status)
    process_psid(raw_dir, output_dir, status)

    print_summary(status, output_dir)


if __name__ == '__main__':
    main()