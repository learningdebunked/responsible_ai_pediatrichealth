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
- FL + DP (ε=1): **AUC = 0.686, F1 = 0.203** — outperforms centralized LR baseline (AUC 0.672, F1 0.201) with class-weighted federated gradients (fixes prior F1=0.0 collapse at threshold 0.5)
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

## Changelog

**2026-07-09 — Ajit Kumar Sahu **

- **Fixed federated learning class-imbalance bug**: the FL model previously predicted the majority class for every case at the standard 0.5 threshold (F1 = Precision = Recall = 0.0 at every privacy budget ε), despite reporting a competitive AUC. Added per-client class-weighted gradients (`code/pipeline.py`) so the FL model now correctly recalls 71% of ASD cases. Result: AUC 0.685 → **0.686**, F1 0.0 → **0.203** at all ε ∈ {1, 5, 10, ∞}.
- **Fixed ROC curve truncation bug** in Fig. 2 and Fig. 5: only the first 100 raw threshold points were saved instead of being resampled across the full FPR range, so curves never reached (1,1). Replaced with a fixed-grid interpolation (`roc_grid()` in `code/pipeline.py`); Fig. 5 now shows full, correctly-terminating ROC curves.
- Fixed hardcoded sandbox file paths (`/home/sandbox/...`) in `code/pipeline.py` and `code/generate_figures.py` so the pipeline runs locally/anywhere.
- Regenerated `figures/fig2_privacy_utility.png` and `figures/fig5_roc_curves.png` to reflect the above fixes. `figures/fig3_equity_audit.png` and `figures/fig4_ablation.png` were intentionally left unchanged (matching the currently-submitted PDF) since their minor numeric drift traced to scikit-learn version differences, not a real bug.
- Fixed an internal inconsistency in the paper: Introduction Contribution C5 stated a "10.2% AUC gap" while the Abstract, Results, Discussion, and Conclusion all correctly state **3.3%**.
- **Citation integrity pass** on `paper/ieee_autism_multimodal_paper.tex` / `paper/references.bib`: verified every citation against real sources via web research.
  - Corrected 3 citations that had fabricated/placeholder DOIs or arXiv IDs (`alshammari2024explainable`, `mohammadifar2023federated`) or were entirely unverifiable (`islam2025multimodal`, replaced with a verified real paper by Singh & Rahman 2025).
  - Removed one unverifiable citation (`amebleh2025privacy`) with no evidence the paper or authors exist.
  - Fixed a misattributed statistic: the "$461 billion" annual ASD cost figure was cited to Buescher et al. 2014 (which reports per-patient lifetime costs, not a national annual figure); added the correct source (Leigh & Du, 2015).
  - Corrected a folic-acid/ASD risk-reduction claim (`guo2020prenatal` → `liu2021folicacid`) to cite the real underlying meta-analysis with accurate effect sizes.
  - Corrected the journal/title metadata for `panjwani2021food` to match the real paper.
- Fixed Table I (baseline classifiers) to match Fig. 5 / `data/results.json` (Gradient Boosting AUC 0.634 → 0.636; Precision values corrected) — both stemmed from the same re-run needed for the ROC curve fix.
- Updated Table II (FL results) with the corrected AUC/F1 values and a new F1 column; updated the Abstract, Discussion, and Conclusion to match.
- Added a **Declarations** section to the paper (Conflict of Interest, Funding, Data Availability, Author Contributions, Generative AI Usage Disclosure) — previously missing, required for IEEE JBHI submission.
- **Not yet resolved**: the self-citation `poreddy2024foodmedicine` could not be independently verified and needs author confirmation. The paper has not been recompiled to PDF in this environment (no LaTeX toolchain available) — recompile before submitting.

---

## License

Code: MIT License. See [LICENSE](../LICENSE).
Data: Synthetic only. Source datasets (Instacart, USDA FARA, NSCH) are public domain / CC0.
