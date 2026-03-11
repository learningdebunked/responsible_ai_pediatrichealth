# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - Synthetic Data Overhaul

### Phase 0: Cleanup
- Deleted 0-byte placeholder files (`docs/api_reference.md`, `docs/ethical_guidelines.md`)
- Updated `.gitignore` with `.env`, `data/public/*`, `logs/`, `.ipynb_checkpoints/`
- Created `.gitkeep` files for empty tracked directories

### Phase 1: Real Data Loaders
- **Added** `src/data/nsch_loader.py` -- NSCH 2022 microdata loader for delay prevalence and demographics
- **Added** `src/data/ce_loader.py` -- Consumer Expenditure Survey loader for age-specific spending curves
- **Added** `src/data/amazon_product_loader.py` -- Amazon Reviews 2023 metadata loader for product classification validation
- **Added** `src/data/psid_loader.py` -- PSID-CDS loader for expenditure-outcome correlations
- **Rewritten** `scripts/download_public_data.py` -- prints manual download instructions, verifies raw files, calls loaders, saves processed outputs

### Phase 2: Fix Synthetic Generator
- **Updated** `configs/default.yaml` -- NSCH/CDC empirical prevalence rates (delay=0.178, language=0.05, motor=0.03, asd=0.027, adhd=0.097) and household structure parameters
- **Extended** `FamilyProfile` dataclass with `n_children` and `household_type` fields
- **Added** empirical baseline purchase rate fallback from CE processed data
- **Added** Gaussian noise (sigma=0.15) to delay purchase multipliers to reduce circularity
- **Added** `generate_confounded_transactions()` -- sibling purchases, gifts, temporal misalignment, multi-shopper noise
- **Added** `scripts/run_ablation_study.py` -- 5 ablation experiments measuring impact of synthetic assumptions on model AUROC

### Phase 3: Fix Taxonomy
- **Updated** all 10 `clinical_alignment` fields in `DevelopMap` with specific citations (ASQ-3, M-CHAT-R/F, SPM, BASC-3, ABAS-3, Vineland-3, CSHQ, PediEAT, BRIEF-P, etc.)
- **Added** `validate_against_asq3()` method to `DevelopMap`
- **Added** `calibrate()` method to `ProductClassifier` -- grid search over threshold and keyword weight to maximize macro F1
- **Added** `scripts/validate_taxonomy.py` -- ASQ-3 alignment, citation validation, keyword coverage, Amazon product classification

### Phase 4: Rename Causal to Temporal Analysis
- **Renamed** `src/causal/` to `src/temporal_analysis/`
- **Renamed** `GrangerCausalityAnalyzer` to `TemporalAssociationAnalyzer` with updated docstring clarifying temporal precedence vs. causation
- **Updated** `configs/default.yaml` section from `causal:` to `temporal_analysis:`
- **Added** `src/temporal_analysis/counterfactual.py` -- `CounterfactualSimulator` (domain removal, trajectory shift, minimum intervention) and `SensitivityAnalyzer` (E-values, Rosenbaum bounds, quantitative bias analysis)

### Phase 5: Fix Evaluation & Fairness
- **Added** `SelectionBiasAnalyzer` to `src/evaluation/fairness.py` -- coverage gap estimation, missing population modeling, selection bias reporting with NSCH reference prevalence
- **Added** `compute_trajectory_stability()` to `ModelEvaluator` -- volatility, instability rate, flip rate across time windows
- **Added** `compute_lead_time_with_baseline()` to `ModelEvaluator` -- per-condition lead time vs. NSCH median diagnosis age

### Phase 6: Fix Privacy
- **Added** `GradientInversionDefense` to `src/privacy/dp_mechanisms.py` -- gradient compression (top-k), SVD perturbation (Soteria-style), InstaHide mixing, reconstruction risk estimation
- **Fixed** `aggregate_masked_updates()` in `SecureAggregator` -- handles dropped clients with weight renormalization and warning
- **Added** `verify_aggregation_integrity()` to `SecureAggregator` -- compares plaintext weighted average to secure aggregation output

### Phase 7: Add Governance Module
- **Added** `src/governance/consent.py` -- `ConsentManager` with granular scopes (data collection, model training, result sharing, research, third-party), COPPA compliance validation, auto-expiry, append-only audit trail
- **Added** `src/governance/audit.py` -- `AuditLogger` with hash-chained entries (training runs, inferences, fairness evaluations, data access, alerts), integrity verification, JSONL persistence

### Phase 8: Add Tests
- **Added** `tests/test_taxonomy.py` -- 12 tests for DevelopMap and ProductClassifier
- **Added** `tests/test_synthetic.py` -- 12 tests for FamilyProfile and SyntheticDataGenerator
- **Added** `tests/test_privacy.py` -- 18 tests for DPMechanism, RenyiDPAccountant, GradientInversionDefense, SecureAggregator, FedAvg, FedProx
- **Added** `tests/test_evaluation.py` -- 10 tests for ModelEvaluator, FairnessAnalyzer, SelectionBiasAnalyzer
- **Added** `tests/test_governance.py` -- 18 tests for ConsentManager and AuditLogger

### Phase 9: Update Docs
- **Rewritten** `README.md` -- honest status disclaimer, updated architecture diagram, project structure, DevelopMap citation table, real data sources table, updated privacy/ethics sections
- **Updated** `requirements.txt` -- renamed "Causal Inference" comment to "Temporal Analysis"
- **Added** `CHANGELOG.md`
