"""Tests for governance module (consent and audit)."""

import pytest
import json
import tempfile
from pathlib import Path
from src.governance.consent import (
    ConsentManager, ConsentScope, ConsentAction, ConsentRecord
)
from src.governance.audit import AuditLogger, AuditEventType


class TestConsentRecord:
    """Tests for ConsentRecord."""

    def test_hash_computed(self):
        record = ConsentRecord(
            family_id="fam_001",
            scope=ConsentScope.DATA_COLLECTION,
            action=ConsentAction.GRANT,
            timestamp="2024-01-01T00:00:00",
            guardian_id="guardian_001",
            child_age_months=24,
        )
        assert len(record.record_hash) == 16

    def test_hash_deterministic(self):
        kwargs = dict(
            family_id="fam_001",
            scope=ConsentScope.DATA_COLLECTION,
            action=ConsentAction.GRANT,
            timestamp="2024-01-01T00:00:00",
            guardian_id="guardian_001",
            child_age_months=24,
        )
        r1 = ConsentRecord(**kwargs)
        r2 = ConsentRecord(**kwargs)
        assert r1.record_hash == r2.record_hash


class TestConsentManager:
    """Tests for ConsentManager."""

    @pytest.fixture
    def manager(self):
        return ConsentManager(default_expiry_days=365)

    def test_grant_consent(self, manager):
        record = manager.grant_consent(
            "fam_001", ConsentScope.DATA_COLLECTION,
            "guardian_001", 24
        )
        assert record.action == ConsentAction.GRANT
        assert manager.check_consent("fam_001", ConsentScope.DATA_COLLECTION)

    def test_revoke_consent(self, manager):
        manager.grant_consent(
            "fam_001", ConsentScope.DATA_COLLECTION,
            "guardian_001", 24
        )
        manager.revoke_consent(
            "fam_001", ConsentScope.DATA_COLLECTION,
            "guardian_001", 24
        )
        assert not manager.check_consent("fam_001", ConsentScope.DATA_COLLECTION)

    def test_check_consent_not_granted(self, manager):
        assert not manager.check_consent("fam_001", ConsentScope.MODEL_TRAINING)

    def test_check_all_required_consent(self, manager):
        manager.grant_consent(
            "fam_001", ConsentScope.DATA_COLLECTION,
            "guardian_001", 24
        )
        # Missing MODEL_TRAINING
        all_ok, missing = manager.check_all_required_consent("fam_001")
        assert not all_ok
        assert ConsentScope.MODEL_TRAINING in missing

    def test_check_all_required_consent_pass(self, manager):
        for scope in [ConsentScope.DATA_COLLECTION, ConsentScope.MODEL_TRAINING]:
            manager.grant_consent("fam_001", scope, "guardian_001", 24)
        all_ok, missing = manager.check_all_required_consent("fam_001")
        assert all_ok
        assert len(missing) == 0

    def test_consent_history(self, manager):
        manager.grant_consent(
            "fam_001", ConsentScope.DATA_COLLECTION,
            "guardian_001", 24
        )
        manager.revoke_consent(
            "fam_001", ConsentScope.DATA_COLLECTION,
            "guardian_001", 24
        )
        history = manager.get_consent_history("fam_001")
        assert len(history) == 2
        assert history[0]['action'] == 'grant'
        assert history[1]['action'] == 'revoke'

    def test_coppa_compliance_under_13(self, manager):
        # No consent granted for child under 13
        result = manager.validate_coppa_compliance("fam_001", 60)
        assert result['coppa_applicable']
        assert not result['compliant']

    def test_coppa_compliance_with_consent(self, manager):
        manager.grant_consent(
            "fam_001", ConsentScope.DATA_COLLECTION,
            "guardian_001", 60
        )
        result = manager.validate_coppa_compliance("fam_001", 60)
        assert result['compliant']

    def test_invalid_child_age(self, manager):
        with pytest.raises(ValueError):
            manager.grant_consent(
                "fam_001", ConsentScope.DATA_COLLECTION,
                "guardian_001", 300  # > 216 months
            )

    def test_export_audit_trail(self, manager):
        manager.grant_consent(
            "fam_001", ConsentScope.DATA_COLLECTION,
            "guardian_001", 24
        )
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        manager.export_audit_trail(path)
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1
        Path(path).unlink()

    def test_summary(self, manager):
        manager.grant_consent("fam_001", ConsentScope.DATA_COLLECTION, "g1", 24)
        manager.grant_consent("fam_002", ConsentScope.DATA_COLLECTION, "g2", 36)
        summary = manager.get_summary()
        assert summary['n_families'] == 2
        assert summary['n_records'] == 2


class TestAuditLogger:
    """Tests for AuditLogger."""

    @pytest.fixture
    def logger(self):
        return AuditLogger()

    def test_log_entry(self, logger):
        entry = logger.log(AuditEventType.DATA_ACCESS, "system", {"file": "test.csv"})
        assert entry.event_type == AuditEventType.DATA_ACCESS
        assert len(entry.entry_hash) == 32

    def test_hash_chain(self, logger):
        e1 = logger.log(AuditEventType.DATA_ACCESS, "sys", {})
        e2 = logger.log(AuditEventType.TRAINING_START, "sys", {})
        assert e2.previous_hash == e1.entry_hash

    def test_verify_integrity_valid(self, logger):
        logger.log(AuditEventType.DATA_ACCESS, "sys", {})
        logger.log(AuditEventType.TRAINING_END, "sys", {})
        result = logger.verify_integrity()
        assert result['is_valid']
        assert result['n_entries'] == 2

    def test_verify_integrity_tampered(self, logger):
        logger.log(AuditEventType.DATA_ACCESS, "sys", {})
        logger.log(AuditEventType.TRAINING_END, "sys", {})
        # Tamper with first entry
        logger._entries[0].entry_hash = "tampered"
        result = logger.verify_integrity()
        assert not result['is_valid']

    def test_log_training_run(self, logger):
        entry = logger.log_training_run(
            actor="researcher",
            model_name="screening_v2",
            hyperparams={'lr': 0.001, 'epochs': 50},
            data_version="abc123",
            metrics={'auroc': 0.85},
            privacy_budget={'epsilon': 1.0, 'delta': 1e-5},
        )
        assert entry.details['model_name'] == 'screening_v2'
        assert entry.details['metrics']['auroc'] == 0.85

    def test_log_inference(self, logger):
        entry = logger.log_inference(
            actor="api_v1",
            input_hash="sha256_abc",
            prediction=1,
            confidence=0.87,
            model_version="v2.1",
        )
        assert entry.details['confidence'] == 0.87

    def test_log_fairness_evaluation(self, logger):
        entry = logger.log_fairness_evaluation(
            actor="evaluator",
            evaluation_results={'demographic_parity': {'gap': 0.05}},
            passed=True,
        )
        assert entry.details['passed']

    def test_get_entries_filter_by_type(self, logger):
        logger.log(AuditEventType.DATA_ACCESS, "sys", {})
        logger.log(AuditEventType.TRAINING_END, "sys", {})
        logger.log(AuditEventType.DATA_ACCESS, "sys", {})
        results = logger.get_entries(event_type=AuditEventType.DATA_ACCESS)
        assert len(results) == 2

    def test_get_entries_filter_by_actor(self, logger):
        logger.log(AuditEventType.DATA_ACCESS, "alice", {})
        logger.log(AuditEventType.DATA_ACCESS, "bob", {})
        results = logger.get_entries(actor="alice")
        assert len(results) == 1

    def test_get_summary(self, logger):
        logger.log(AuditEventType.DATA_ACCESS, "sys", {})
        logger.log(AuditEventType.ALERT, "sys", {'severity': 'high', 'message': 'test', 'context': {}})
        summary = logger.get_summary()
        assert summary['n_entries'] == 2
        assert summary['integrity_valid']

    def test_persistence_to_file(self):
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            path = f.name

        # Write
        logger1 = AuditLogger(log_path=path)
        logger1.log(AuditEventType.DATA_ACCESS, "sys", {"key": "val"})
        logger1.log(AuditEventType.TRAINING_END, "sys", {})

        # Read back
        logger2 = AuditLogger(log_path=path)
        assert len(logger2._entries) == 2
        assert logger2.verify_integrity()['is_valid']

        Path(path).unlink()

    def test_export(self, logger):
        logger.log(AuditEventType.DATA_ACCESS, "sys", {})
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        logger.export(path)
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1
        Path(path).unlink()
