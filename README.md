# RetailHealth: Privacy-Preserving Developmental Screening Framework

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

RetailHealth is a privacy-preserving framework for embedding developmental screening into e-commerce platforms via federated learning. The system enables multi-retailer collaboration without sharing customer data, with the goal of earlier detection of childhood developmental delays.

> **Honest status**: This is a research prototype. We have integrated **real data loaders** (Instacart, Tesco, Kaggle ASD, NSCH) to eliminate synthetic circularity. The synthetic generator now uses empirical purchase rates when available. Real validation using Kaggle ASD diagnostic labels is now possible. See [Disclaimer](#disclaimer).

### Key Features

- **Privacy-First**: Formal (ε, δ)-differential privacy with gradient inversion defense
- **Federated Learning**: Multi-retailer collaboration without data sharing
- **DevelopMap Taxonomy**: Product classification across 10 developmental domains with clinical citations (ASQ-3, M-CHAT-R/F, SPM, BASC-3, etc.)
- **Temporal Analysis**: Granger-style precedence tests (explicitly *not* causal claims)
- **Real Data Loaders**: NSCH, Consumer Expenditure Survey, Amazon Reviews 2023, PSID-CDS, Instacart, Tesco, Kaggle ASD
- **Fairness & Selection Bias**: `SelectionBiasAnalyzer` for coverage gap estimation
- **Governance**: COPPA-aware consent management and hash-chained audit logs

## Quick Start

### Installation

```bash
git clone https://github.com/username/retailhealth-framework.git
cd retailhealth-framework
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Download Real Data (Recommended)

```bash
# Print download instructions for all required datasets
python scripts/download_public_data.py --help-downloads

# After downloading datasets into data/raw/, process them
python scripts/download_public_data.py --raw_dir data/raw --output_dir data/processed
```

### Generate Synthetic Data (Fallback)

```bash
python scripts/generate_synthetic_data.py --n_families 100000 --output_dir data/synthetic
```

### Run Ablation Study

```bash
python scripts/run_ablation_study.py --n_families 5000 --output_dir results/ablation
```

### Validate Taxonomy

```bash
python scripts/validate_taxonomy.py --output_dir results/taxonomy
```

### Run Tests

```bash
pytest tests/ -v
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Retail Platform Layer                       │
│         (Shopify / Magento / Custom APIs)                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Governance Layer (NEW)                          │
│    (Consent Manager, Audit Logger, COPPA Compliance)       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Privacy Layer                             │
│  (DP Noise, Secure Aggregation, Gradient Inversion Defense)│
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Feature Layer                              │
│  (DevelopMap, Temporal Encoder, Age Baselines)              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Federated Learning Layer                        │
│    (FedAvg/FedProx + Rényi DP Accounting)                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│          Temporal Analysis Layer (renamed from causal)       │
│  (Granger Precedence, Counterfactuals, Sensitivity)        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Evaluation Layer                                │
│  (Fairness, Selection Bias, Trajectory Stability)          │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
retailhealth-framework/
├── README.md
├── CHANGELOG.md
├── LICENSE
├── requirements.txt
├── setup.py
├── configs/
│   └── default.yaml             # NSCH-updated prevalence rates
├── src/
│   ├── taxonomy/                # DevelopMap with clinical citations
│   │   ├── developmap.py        # 10 domains + validate_against_asq3()
│   │   └── classifier.py       # ProductClassifier + calibrate()
│   ├── data/
│   │   ├── synthetic_generator.py  # Uses real data rates when available
│   │   ├── nsch_loader.py       # NSCH microdata + population stats
│   │   ├── ce_loader.py         # Consumer Expenditure Survey loader
│   │   ├── amazon_product_loader.py  # Amazon Reviews 2023 loader
│   │   ├── psid_loader.py       # PSID-CDS loader
│   │   ├── instacart_loader.py  # Instacart transaction patterns
│   │   ├── tesco_loader.py      # Tesco UK grocery data
│   │   └── kaggle_asd_loader.py # Kaggle ASD diagnostic labels
│   ├── privacy/
│   │   ├── dp_mechanisms.py     # DP + GradientInversionDefense
│   │   └── secure_aggregation.py  # Dropped-client handling + integrity
│   ├── features/                # Temporal encoding
│   ├── federated/               # FL client, server, models
│   ├── temporal_analysis/       # Renamed from causal/
│   │   ├── temporal_discovery.py  # TemporalAssociationAnalyzer
│   │   ├── propensity.py        # Propensity score methods
│   │   └── counterfactual.py    # Counterfactual + sensitivity analysis
│   ├── evaluation/
│   │   ├── metrics.py           # + trajectory stability, NSCH lead time
│   │   └── fairness.py          # + SelectionBiasAnalyzer
│   ├── governance/              # NEW
│   │   ├── consent.py           # COPPA-aware consent manager
│   │   └── audit.py             # Hash-chained audit logger
│   └── utils/
├── scripts/
│   ├── download_public_data.py  # Real data processing pipeline
│   ├── run_ablation_study.py    # 5-experiment ablation
│   ├── validate_taxonomy.py     # ASQ-3 + Amazon validation
│   └── validate_with_real_data.py  # Validation using real datasets
├── tests/
│   ├── test_taxonomy.py
│   ├── test_synthetic.py
│   ├── test_privacy.py
│   ├── test_evaluation.py
│   └── test_governance.py
├── notebooks/
├── data/
│   ├── raw/                     # Downloaded public datasets
│   ├── processed/               # Processed outputs
│   └── synthetic/               # Generated synthetic data
└── docs/
```

## DevelopMap Taxonomy

Universal product classification across 10 developmental domains, each with specific clinical tool citations:

| Domain | Clinical Alignment |
|--------|-------------------|
| Fine Motor | ASQ-3 Fine Motor (Squires & Bricker, 2009); PDMS-2 (Folio & Fewell, 2000) |
| Gross Motor | ASQ-3 Gross Motor; Bayley-III Motor Scale (Bayley, 2006) |
| Language | ASQ-3 Communication; M-CHAT-R/F Items 5,6 (Robins et al., 2014) |
| Social-Emotional | ASQ-3 Personal-Social; M-CHAT-R/F; ITSEA (Carter & Briggs-Gowan, 2006) |
| Sensory | SPM (Parham & Ecker, 2007); Sensory Profile-2 (Dunn, 2014) |
| Adaptive | ABAS-3 (Harrison & Oakland, 2015); Vineland-3 (Sparrow et al., 2016) |
| Sleep | CSHQ (Owens et al., 2000); BISQ (Sadeh, 2004) |
| Feeding | PediEAT (Thoyre et al., 2014); BAMBI (Lukens & Linscheid, 2008) |
| Behavioral | BASC-3 (Reynolds & Kamphaus, 2015); BRIEF-P (Gioia et al., 2003) |
| Therapeutic | IDEA Part C; AAP guidelines (Lipkin et al., 2020) |

## Real Data Sources

| Dataset | Purpose | URL |
|---------|---------|-----|
| NSCH 2022 | Delay prevalence, demographics, population statistics | census.gov/programs-surveys/nsch |
| CE PUMD | Age-specific spending curves | bls.gov/cex/pumd_data.htm |
| Amazon Reviews 2023 | Product classification validation | huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 |
| PSID-CDS | Expenditure-outcome correlations | psidonline.isr.umich.edu/cds/ |
| **Instacart** | Real grocery transaction patterns (3M+ orders) | kaggle.com/c/instacart-market-basket-analysis |
| **Tesco** | UK grocery basket data for cross-validation | kaggle.com/datasets/tesco/tesco-grocery-1-0 |
| **Kaggle ASD** | Real ASD diagnostic labels for validation | kaggle.com/datasets/andrewmvd/autism-screening-on-adults |

## Privacy Guarantees

- **Differential Privacy**: Gaussian mechanism + gradient clipping + Rényi DP accounting
- **Gradient Inversion Defense**: Top-k compression, SVD perturbation, InstaHide mixing
- **Secure Aggregation**: Masked updates with dropped-client handling + integrity verification

## Ethical Considerations

- **Non-Diagnostic**: Screening prompts only, not diagnoses
- **Consent**: COPPA-aware, granular, revocable consent management
- **Audit Trail**: Hash-chained, tamper-evident logging of all model lifecycle events
- **Fairness**: Selection bias analysis, coverage gap estimation, per-group metrics
- **Governance**: IRB review required; prohibition on monetizing health signals

## Citation

```bibtex
@article{poreddy2025retailhealth,
  title={Privacy-Preserving Developmental Screening Through Retail Transaction
         Analytics: A Federated Learning Framework for E-Commerce Platforms},
  author={Poreddy, Kapil},
  journal={arXiv preprint},
  year={2025}
}
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## Contact

- **Author**: Kapil Poreddy
- **Email**: poreddykapil@ieee.org
- **Repository**: https://github.com/learningdebunked/responsible_ai_pediatrichealth

## Acknowledgments

This work builds on research in federated learning, differential privacy, developmental psychology, and retail analytics. We thank the open-source community for tools including PyTorch, Flower, and Opacus.

## Disclaimer

**IMPORTANT**: This is a research prototype. Key limitations:

1. **Circularity mitigation in progress**: The synthetic generator now loads empirical rates from Instacart/Tesco when available, reducing reliance on hardcoded multipliers. Run `scripts/run_ablation_study.py` to quantify remaining circularity.
2. **Real validation now possible**: Kaggle ASD datasets provide real diagnostic labels. Run `scripts/validate_with_real_data.py` to validate against real data instead of synthetic.
3. **Temporal ≠ causal**: The `temporal_analysis` module identifies temporal precedence (Granger-style), not causation. Unmeasured confounders cannot be ruled out.
4. **Not clinically validated**: Do not use for actual medical screening without prospective clinical studies, IRB approval, and regulatory compliance (FDA, FTC, state privacy laws).
5. **Selection bias**: Retail purchase data systematically under-represents low-income and rural populations. The `SelectionBiasAnalyzer` quantifies but does not eliminate this.
6. **Dataset access**: Some datasets (Instacart, Tesco, Kaggle ASD) require Kaggle accounts. NSCH and PSID-CDS require accepting data use agreements.
