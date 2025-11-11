# RetailHealth Deployment Guide

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -e .

# 2. Generate synthetic data (small dataset for testing)
python scripts/generate_synthetic_data.py --n_families 1000 --output_dir data/synthetic

# 3. Verify installation
python -c "from src.taxonomy.developmap import DEVELOPMAP; print('✓ Installation successful')"
```

## System Requirements

- **Python**: 3.8+
- **RAM**: 16GB minimum
- **Storage**: 50GB
- **GPU**: Optional (10x speedup)

## Data Generation

### Small Dataset (Testing)
```bash
python scripts/generate_synthetic_data.py --n_families 1000 --output_dir data/synthetic/small
```

### Full Dataset (Research)
```bash
python scripts/generate_synthetic_data.py --n_families 100000 --output_dir data/synthetic/full
```

## Deployment Phases

### Phase 1: Pilot (Months 1-6)
- Partner with 1-2 retailers
- 10,000 opt-in families
- IRB approval required

### Phase 2: Federation (Months 7-12)
- Expand to 5-10 retailers
- Enable federated learning

### Phase 3: Clinical Integration (Year 2)
- EHR API integration
- Pediatrician training

### Phase 4: Scale (Year 3+)
- National rollout
- 50M+ children

## Troubleshooting

### Out of Memory
```bash
python scripts/train_federated.py --batch_size 16
```

### Slow Training
```bash
python scripts/train_federated.py --mixed_precision
```

## Contact

- **Email**: kapilporeddy@researchmail.org
- **Issues**: https://github.com/username/retailhealth-framework/issues