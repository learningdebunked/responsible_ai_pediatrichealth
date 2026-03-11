"""Audit trail and model governance for RetailHealth.

Provides tamper-evident logging of all model lifecycle events:
- Training runs (hyperparameters, data versions, metrics)
- Inference requests (input hashes, predictions, confidence)
- Fairness evaluations (per-group metrics, bias flags)
- Data access events (who accessed what, when)
- Model deployments and rollbacks

All entries are append-only and hash-chained for integrity.
"""

import json
import hashlib
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


class AuditEventType(Enum):
    """Types of auditable events."""
    TRAINING_START = "training_start"
    TRAINING_END = "training_end"
    INFERENCE = "inference"
    FAIRNESS_EVAL = "fairness_eval"
    DATA_ACCESS = "data_access"
    MODEL_DEPLOY = "model_deploy"
    MODEL_ROLLBACK = "model_rollback"
    CONSENT_CHANGE = "consent_change"
    PRIVACY_BUDGET = "privacy_budget"
    ALERT = "alert"


@dataclass
class AuditEntry:
    """Single entry in the audit log."""
    event_type: AuditEventType
    timestamp: str
    actor: str
    details: Dict[str, Any]
    entry_hash: str = ""
    previous_hash: str = ""

    def __post_init__(self):
        if not self.entry_hash:
            self.entry_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = (
            f"{self.event_type.value}|{self.timestamp}|{self.actor}|"
            f"{json.dumps(self.details, sort_keys=True)}|{self.previous_hash}"
        )
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def to_dict(self) -> Dict:
        return {
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'actor': self.actor,
            'details': self.details,
            'entry_hash': self.entry_hash,
            'previous_hash': self.previous_hash,
        }


class AuditLogger:
    """Append-only, hash-chained audit logger.

    Each entry's hash includes the previous entry's hash, forming
    a tamper-evident chain. Any modification to a past entry will
    invalidate all subsequent hashes.
    """

    def __init__(self, log_path: str = None):
        """Initialize audit logger.

        Args:
            log_path: Path to persist audit log as JSONL.
                If None, log is kept in memory only.
        """
        self.log_path = log_path
        self._entries: List[AuditEntry] = []
        self._last_hash = "genesis"

        # Load existing log if present
        if log_path and Path(log_path).exists():
            self._load_log(log_path)

    def log(self, event_type: AuditEventType, actor: str,
            details: Dict[str, Any] = None) -> AuditEntry:
        """Append an event to the audit log.

        Args:
            event_type: Type of event
            actor: Who/what triggered the event (user ID, system component)
            details: Event-specific metadata

        Returns:
            The created AuditEntry
        """
        if details is None:
            details = {}

        entry = AuditEntry(
            event_type=event_type,
            timestamp=datetime.utcnow().isoformat(),
            actor=actor,
            details=details,
            previous_hash=self._last_hash,
        )

        self._entries.append(entry)
        self._last_hash = entry.entry_hash

        # Persist
        if self.log_path:
            self._append_to_file(entry)

        return entry

    def log_training_run(self, actor: str, model_name: str,
                          hyperparams: Dict, data_version: str,
                          metrics: Dict = None,
                          privacy_budget: Dict = None) -> AuditEntry:
        """Log a model training run.

        Args:
            actor: Who initiated training
            model_name: Name/version of model
            hyperparams: Training hyperparameters
            data_version: Hash or version of training data
            metrics: Final evaluation metrics
            privacy_budget: Privacy budget consumed (epsilon, delta)

        Returns:
            AuditEntry for the training event
        """
        details = {
            'model_name': model_name,
            'hyperparams': hyperparams,
            'data_version': data_version,
        }
        if metrics:
            details['metrics'] = metrics
        if privacy_budget:
            details['privacy_budget'] = privacy_budget

        return self.log(AuditEventType.TRAINING_END, actor, details)

    def log_inference(self, actor: str, input_hash: str,
                       prediction: Any, confidence: float,
                       model_version: str) -> AuditEntry:
        """Log an inference request.

        Args:
            actor: System component or API key
            input_hash: SHA-256 hash of input (not the raw data)
            prediction: Model output
            confidence: Prediction confidence
            model_version: Model version used

        Returns:
            AuditEntry
        """
        return self.log(AuditEventType.INFERENCE, actor, {
            'input_hash': input_hash,
            'prediction': prediction,
            'confidence': confidence,
            'model_version': model_version,
        })

    def log_fairness_evaluation(self, actor: str,
                                  evaluation_results: Dict,
                                  passed: bool,
                                  violations: List[str] = None) -> AuditEntry:
        """Log a fairness evaluation.

        Args:
            actor: Who ran the evaluation
            evaluation_results: Full fairness metrics
            passed: Whether fairness criteria were met
            violations: List of specific violations

        Returns:
            AuditEntry
        """
        return self.log(AuditEventType.FAIRNESS_EVAL, actor, {
            'passed': passed,
            'violations': violations or [],
            'summary': {
                k: v for k, v in evaluation_results.items()
                if k in ('demographic_parity', 'equalized_odds',
                         'calibration_gap')
            },
        })

    def log_data_access(self, actor: str, dataset: str,
                         operation: str, n_records: int,
                         purpose: str) -> AuditEntry:
        """Log a data access event.

        Args:
            actor: Who accessed the data
            dataset: Dataset identifier
            operation: read, write, delete, export
            n_records: Number of records accessed
            purpose: Stated purpose for access

        Returns:
            AuditEntry
        """
        return self.log(AuditEventType.DATA_ACCESS, actor, {
            'dataset': dataset,
            'operation': operation,
            'n_records': n_records,
            'purpose': purpose,
        })

    def log_alert(self, actor: str, severity: str,
                   message: str, context: Dict = None) -> AuditEntry:
        """Log an alert (e.g., fairness violation, privacy breach).

        Args:
            actor: Component raising the alert
            severity: critical, high, medium, low
            message: Alert description
            context: Additional context

        Returns:
            AuditEntry
        """
        return self.log(AuditEventType.ALERT, actor, {
            'severity': severity,
            'message': message,
            'context': context or {},
        })

    def verify_integrity(self) -> Dict:
        """Verify hash chain integrity of the audit log.

        Returns:
            Dict with is_valid, n_entries, and any broken_at index
        """
        if not self._entries:
            return {'is_valid': True, 'n_entries': 0}

        expected_prev = "genesis"

        for i, entry in enumerate(self._entries):
            if entry.previous_hash != expected_prev:
                return {
                    'is_valid': False,
                    'n_entries': len(self._entries),
                    'broken_at': i,
                    'expected_previous_hash': expected_prev,
                    'actual_previous_hash': entry.previous_hash,
                }

            # Recompute hash to detect tampering
            recomputed = entry._compute_hash()
            if recomputed != entry.entry_hash:
                return {
                    'is_valid': False,
                    'n_entries': len(self._entries),
                    'tampered_at': i,
                    'expected_hash': recomputed,
                    'actual_hash': entry.entry_hash,
                }

            expected_prev = entry.entry_hash

        return {
            'is_valid': True,
            'n_entries': len(self._entries),
            'last_hash': self._last_hash,
        }

    def get_entries(self, event_type: AuditEventType = None,
                     actor: str = None,
                     since: str = None,
                     limit: int = None) -> List[Dict]:
        """Query audit log entries.

        Args:
            event_type: Filter by event type
            actor: Filter by actor
            since: ISO timestamp, return entries after this time
            limit: Maximum entries to return

        Returns:
            List of matching entries as dicts
        """
        results = self._entries

        if event_type is not None:
            results = [e for e in results if e.event_type == event_type]

        if actor is not None:
            results = [e for e in results if e.actor == actor]

        if since is not None:
            results = [e for e in results if e.timestamp >= since]

        if limit is not None:
            results = results[-limit:]

        return [e.to_dict() for e in results]

    def get_summary(self) -> Dict:
        """Get summary statistics of the audit log."""
        type_counts = {}
        for entry in self._entries:
            key = entry.event_type.value
            type_counts[key] = type_counts.get(key, 0) + 1

        actors = set(e.actor for e in self._entries)

        integrity = self.verify_integrity()

        return {
            'n_entries': len(self._entries),
            'event_type_counts': type_counts,
            'n_actors': len(actors),
            'integrity_valid': integrity['is_valid'],
            'first_entry': self._entries[0].timestamp if self._entries else None,
            'last_entry': self._entries[-1].timestamp if self._entries else None,
        }

    def export(self, output_path: str):
        """Export full audit log to JSON file.

        Args:
            output_path: Path to write JSON file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        entries = [e.to_dict() for e in self._entries]
        with open(output_path, 'w') as f:
            json.dump(entries, f, indent=2)

    def _append_to_file(self, entry: AuditEntry):
        """Append a single entry to the JSONL log file."""
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry.to_dict()) + '\n')

    def _load_log(self, log_path: str):
        """Load existing JSONL audit log."""
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                entry = AuditEntry(
                    event_type=AuditEventType(data['event_type']),
                    timestamp=data['timestamp'],
                    actor=data['actor'],
                    details=data['details'],
                    previous_hash=data['previous_hash'],
                    entry_hash=data['entry_hash'],
                )
                self._entries.append(entry)
                self._last_hash = entry.entry_hash
