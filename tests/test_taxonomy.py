"""Tests for DevelopMap taxonomy and ProductClassifier."""

import pytest
import numpy as np
from src.taxonomy.developmap import DevelopMap, DEVELOPMAP, DevelopmentalDomain
from src.taxonomy.classifier import ProductClassifier


class TestDevelopMap:
    """Tests for DevelopMap taxonomy."""

    def test_all_domains_present(self):
        dm = DevelopMap()
        expected = {
            'fine_motor', 'gross_motor', 'language', 'social_emotional',
            'sensory', 'adaptive', 'sleep', 'feeding', 'behavioral',
            'therapeutic'
        }
        assert set(dm.get_domain_names()) == expected

    def test_domain_has_required_fields(self):
        dm = DevelopMap()
        for name, domain in dm.domains.items():
            assert isinstance(domain, DevelopmentalDomain), f"{name} is not DevelopmentalDomain"
            assert len(domain.name) > 0, f"{name} missing name"
            assert len(domain.description) > 0, f"{name} missing description"
            assert len(domain.keywords) > 0, f"{name} has no keywords"
            assert len(domain.example_products) > 0, f"{name} has no example products"
            assert len(domain.clinical_alignment) > 0, f"{name} missing clinical_alignment"

    def test_clinical_citations_have_references(self):
        """Phase 3: Every domain should have a specific citation with year."""
        dm = DevelopMap()
        for name, domain in dm.domains.items():
            citation = domain.clinical_alignment
            has_year = any(c.isdigit() for c in citation)
            has_parens = '(' in citation and ')' in citation
            assert has_year, f"{name} citation missing year: {citation}"
            assert has_parens, f"{name} citation missing author ref: {citation}"

    def test_no_empty_keyword_sets(self):
        dm = DevelopMap()
        for name, domain in dm.domains.items():
            assert len(domain.keywords) >= 5, (
                f"{name} has too few keywords ({len(domain.keywords)})"
            )

    def test_get_domain_returns_correct_type(self):
        dm = DevelopMap()
        domain = dm.get_domain('language')
        assert isinstance(domain, DevelopmentalDomain)
        assert domain.name == "Language Development"

    def test_get_domain_returns_none_for_invalid(self):
        dm = DevelopMap()
        assert dm.get_domain('nonexistent') is None

    def test_get_all_keywords_structure(self):
        dm = DevelopMap()
        kw = dm.get_all_keywords()
        assert isinstance(kw, dict)
        for name, keywords in kw.items():
            assert isinstance(keywords, set)

    def test_validate_against_asq3(self):
        dm = DevelopMap()
        results = dm.validate_against_asq3()
        assert isinstance(results, dict)
        # At least 9/10 should map (therapeutic is the outlier)
        n_mapped = sum(1 for v in results.values() if v['has_asq3_mapping'])
        assert n_mapped >= 9, f"Only {n_mapped}/10 domains mapped to ASQ-3"

    def test_global_instance_exists(self):
        assert DEVELOPMAP is not None
        assert isinstance(DEVELOPMAP, DevelopMap)


class TestProductClassifier:
    """Tests for ProductClassifier."""

    @pytest.fixture
    def classifier(self):
        return ProductClassifier(use_embeddings=False)

    def test_classify_returns_dict(self, classifier):
        result = classifier.classify("wooden puzzle for toddlers")
        assert isinstance(result, dict)

    def test_classify_fine_motor_product(self, classifier):
        result = classifier.classify("wooden puzzle blocks for fine motor skills")
        assert 'fine_motor' in result
        assert result['fine_motor'] > 0

    def test_classify_language_product(self, classifier):
        result = classifier.classify("board books for baby vocabulary learning")
        assert 'language' in result
        assert result['language'] > 0

    def test_classify_gross_motor_product(self, classifier):
        result = classifier.classify("balance bike outdoor sports scooter")
        assert 'gross_motor' in result

    def test_classify_sensory_product(self, classifier):
        result = classifier.classify("fidget toys sensory weighted blanket")
        assert 'sensory' in result

    def test_classify_empty_string(self, classifier):
        result = classifier.classify("")
        assert isinstance(result, dict)

    def test_classify_batch(self, classifier):
        products = [
            "wooden puzzles",
            "board books",
            "balance bike",
        ]
        results = classifier.classify_batch(products)
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)

    def test_get_primary_domain(self, classifier):
        result = classifier.get_primary_domain("crayons markers art supplies fine motor")
        assert result is not None
        assert len(result) == 2  # (domain_name, score)

    def test_get_domain_vector(self, classifier):
        vec = classifier.get_domain_vector("sensory fidget toy")
        assert isinstance(vec, np.ndarray)
        assert len(vec) == len(DEVELOPMAP.get_domain_names())

    def test_calibrate_returns_valid_result(self, classifier):
        labeled = [
            ("wooden puzzle blocks", "fine_motor"),
            ("board books for reading", "language"),
            ("soccer ball outdoor", "gross_motor"),
            ("fidget spinner sensory", "sensory"),
            ("visual schedule routine", "behavioral"),
        ]
        result = classifier.calibrate(labeled, n_steps=3)
        assert 'best_threshold' in result
        assert 'best_keyword_weight' in result
        assert 'best_f1' in result
        assert result['best_f1'] >= 0.0
