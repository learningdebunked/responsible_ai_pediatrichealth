# PrenatalSentinel

**A Privacy-Preserving Federated Learning Framework for Autism Risk Prediction from Multimodal Prenatal Purchase Signals**

> Kapil Kumar Reddy Poreddy · IEEE Member · poreddykapil@ieee.org
> Submitted to *IEEE Journal of Biomedical and Health Informatics (JBHI)*, June 2025

---

## Overview

PrenatalSentinel integrates three publicly available prenatal signals — grocery food purchases, OTC supplement purchases, and USDA Food Desert / SES data — into a federated learning pipeline with (ε,δ)-differential privacy to screen for autism risk **before birth**, without exposing individual health records.

| Signal | Dataset | Features |
|--------|---------|----------|
| Grocery purchases | Instacart 2017 (Kaggle) | folate, omega-3, iron, vitD, processed food index, dietary diversity |
| OTC/supplement purchases | Instacart Aisles 47 & 11 | supplement adherence, medication risk |
| SES / Food Desert | USDA FARA 2019 + NSCH 2020–22 | food desert score, SES quintile, urban flag |

**Key Results (N=5,000 synthetic cohort):**
- FL + DP (ε=1): **AUC = 0.685** — outperforms centralized LR baseline (0.672)
- Flat privacy-utility curve across ε ∈ {1, 5, 10, ∞}
- Equity gap: high food desert AUC = 0.607 vs. low food desert = 0.628 (3.3%)
- Best signal config: F+M+E (food + OTC + SES) → AUC = 0.650

---

## Repository Structure

```
prenatal_sentinel/
├── paper/
│   ├── PrenatalSentinel_IEEE_JBHI.pdf    ← Compiled submission-ready PDF (8 pages)
│   ├── ieee_autism_multimodal_paper.tex  ← Full LaTeX source (IEEEtran)
│   └── references.bib                   ← BibTeX bibliography (25 entries)
├── figures/
│   ├── prenatal_sentinel_architecture.png ← Fig 1: System architecture
│   ├── fig2_privacy_utility.png           ← Fig 2: Privacy-utility tradeoff
│   ├── fig3_equity_audit.png              ← Fig 3: Equity audit by SES quintile
│   ├── fig4_ablation.png                  ← Fig 4: Signal ablation study
│   └── fig5_roc_curves.png               ← Fig 5: ROC curves
├── data/
│   ├── prenatal_sentinel_dataset.csv     ← Synthetic cohort (5,000 × 15 cols)
│   └── results.json                     ← All model metrics, ROC data, ablation
├── code/
│   ├── pipeline.py                      ← Full ML pipeline (data gen + training)
│   └── generate_figures.py             ← Figure generation script
└── README.md                           ← This file
```

---

## Quick Start

```bash
# Install dependencies
pip install numpy pandas scikit-learn matplotlib

# Run the full pipeline (generates dataset + trains all models)
python code/pipeline.py

# Generate all figures from results.json
python code/generate_figures.py
```

---

## Citation

```bibtex
@article{poreddy2025prenatalsentinel,
  author  = {Poreddy, Kapil Kumar Reddy},
  title   = {PrenatalSentinel: A Privacy-Preserving Federated Learning Framework
             for Autism Risk Prediction from Multimodal Prenatal Purchase Signals},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2025},
  note    = {Submitted}
}
```

---

## License

Code: MIT License. See [LICENSE](../LICENSE).
Data: Synthetic only. Source datasets (Instacart, USDA FARA, NSCH) are public domain / CC0.
