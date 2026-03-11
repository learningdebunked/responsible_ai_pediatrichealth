"""Consent management for pediatric health data.

Implements tiered consent management for families participating in
retail-based developmental screening, following COPPA, FERPA, and
HIPAA requirements for pediatric data.

Key design decisions:
- Consent is opt-in, granular, and revocable at any time
- Separate consent required for data collection, model training,
  and result sharing
- Consent records are immutable (append-only audit log)
- Minors' data requires parental/guardian consent
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path


class ConsentScope(Enum):
    """Scopes for which consent can be granted or revoked."""
    DATA_COLLECTION = "data_collection"
    MODEL_TRAINING = "model_training"
    RESULT_SHARING = "result_sharing"
    RESEARCH_USE = "research_use"
    THIRD_PARTY_SHARING = "third_party_sharing"


class ConsentAction(Enum):
    """Actions in the consent audit trail."""
    GRANT = "grant"
    REVOKE = "revoke"
    EXPIRE = "expire"
    RENEW = "renew"


@dataclass
class ConsentRecord:
    """Immutable record of a consent action."""
    family_id: str
    scope: ConsentScope
    action: ConsentAction
    timestamp: str
    guardian_id: str
    child_age_months: int
    expiry_date: Optional[str] = None
    reason: Optional[str] = None
    record_hash: str = ""

    def __post_init__(self):
        if not self.record_hash:
            self.record_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of record contents for tamper detection."""
        content = (
            f"{self.family_id}|{self.scope.value}|{self.action.value}|"
            f"{self.timestamp}|{self.guardian_id}|{self.child_age_months}|"
            f"{self.expiry_date}|{self.reason}"
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class ConsentManager:
    """Manage consent for pediatric data processing.

    Provides methods to:
    - Grant/revoke consent per scope
    - Check consent status before data operations
    - Auto-expire consent after configurable period
    - Export consent audit trail
    - Validate COPPA compliance
    """

    def __init__(self, default_expiry_days: int = 365,
                 min_child_age_months: int = 0,
                 max_child_age_months: int = 216):
        """Initialize consent manager.

        Args:
            default_expiry_days: Days until consent auto-expires
            min_child_age_months: Minimum child age (0 = newborn)
            max_child_age_months: Maximum child age (216 = 18 years)
        """
        self.default_expiry_days = default_expiry_days
        self.min_child_age_months = min_child_age_months
        self.max_child_age_months = max_child_age_months

        # Append-only consent ledger
        self._records: List[ConsentRecord] = []

        # Current consent state (derived from records)
        # {family_id: {scope: bool}}
        self._consent_state: Dict[str, Dict[ConsentScope, bool]] = {}

    def grant_consent(self, family_id: str, scope: ConsentScope,
                      guardian_id: str, child_age_months: int,
                      expiry_days: int = None,
                      reason: str = None) -> ConsentRecord:
        """Grant consent for a specific scope.

        Args:
            family_id: Family identifier
            scope: Consent scope to grant
            guardian_id: ID of consenting guardian
            child_age_months: Current age of child
            expiry_days: Days until consent expires (default: self.default_expiry_days)
            reason: Optional reason for granting

        Returns:
            ConsentRecord documenting the grant

        Raises:
            ValueError: If child age is outside allowed range
        """
        self._validate_child_age(child_age_months)

        if expiry_days is None:
            expiry_days = self.default_expiry_days

        now = datetime.utcnow()
        expiry = now + timedelta(days=expiry_days)

        record = ConsentRecord(
            family_id=family_id,
            scope=scope,
            action=ConsentAction.GRANT,
            timestamp=now.isoformat(),
            guardian_id=guardian_id,
            child_age_months=child_age_months,
            expiry_date=expiry.isoformat(),
            reason=reason,
        )

        self._records.append(record)
        self._consent_state.setdefault(family_id, {})[scope] = True

        return record

    def revoke_consent(self, family_id: str, scope: ConsentScope,
                       guardian_id: str, child_age_months: int,
                       reason: str = None) -> ConsentRecord:
        """Revoke previously granted consent.

        Args:
            family_id: Family identifier
            scope: Consent scope to revoke
            guardian_id: ID of revoking guardian
            child_age_months: Current age of child
            reason: Optional reason for revocation

        Returns:
            ConsentRecord documenting the revocation
        """
        record = ConsentRecord(
            family_id=family_id,
            scope=scope,
            action=ConsentAction.REVOKE,
            timestamp=datetime.utcnow().isoformat(),
            guardian_id=guardian_id,
            child_age_months=child_age_months,
            reason=reason or "Guardian-initiated revocation",
        )

        self._records.append(record)
        self._consent_state.setdefault(family_id, {})[scope] = False

        return record

    def check_consent(self, family_id: str, scope: ConsentScope) -> bool:
        """Check if consent is currently active for a scope.

        Also checks expiry: if the most recent grant has expired,
        consent is treated as revoked.

        Args:
            family_id: Family identifier
            scope: Consent scope to check

        Returns:
            True if consent is active and not expired
        """
        state = self._consent_state.get(family_id, {})
        if not state.get(scope, False):
            return False

        # Check expiry of most recent grant
        grants = [
            r for r in self._records
            if r.family_id == family_id
            and r.scope == scope
            and r.action == ConsentAction.GRANT
        ]

        if not grants:
            return False

        latest_grant = grants[-1]
        if latest_grant.expiry_date:
            expiry = datetime.fromisoformat(latest_grant.expiry_date)
            if datetime.utcnow() > expiry:
                # Auto-expire
                self._consent_state[family_id][scope] = False
                self._records.append(ConsentRecord(
                    family_id=family_id,
                    scope=scope,
                    action=ConsentAction.EXPIRE,
                    timestamp=datetime.utcnow().isoformat(),
                    guardian_id=latest_grant.guardian_id,
                    child_age_months=latest_grant.child_age_months,
                    reason="Auto-expired",
                ))
                return False

        return True

    def check_all_required_consent(self, family_id: str,
                                    required_scopes: List[ConsentScope] = None
                                    ) -> Tuple[bool, List[ConsentScope]]:
        """Check if all required consent scopes are active.

        Args:
            family_id: Family identifier
            required_scopes: Scopes to check. Defaults to
                [DATA_COLLECTION, MODEL_TRAINING].

        Returns:
            Tuple of (all_granted: bool, missing_scopes: list)
        """
        if required_scopes is None:
            required_scopes = [
                ConsentScope.DATA_COLLECTION,
                ConsentScope.MODEL_TRAINING,
            ]

        missing = [s for s in required_scopes
                    if not self.check_consent(family_id, s)]

        return len(missing) == 0, missing

    def get_consent_history(self, family_id: str) -> List[Dict]:
        """Get full consent history for a family.

        Args:
            family_id: Family identifier

        Returns:
            List of consent records as dicts
        """
        return [
            {
                'scope': r.scope.value,
                'action': r.action.value,
                'timestamp': r.timestamp,
                'guardian_id': r.guardian_id,
                'expiry_date': r.expiry_date,
                'reason': r.reason,
                'record_hash': r.record_hash,
            }
            for r in self._records
            if r.family_id == family_id
        ]

    def validate_coppa_compliance(self, family_id: str,
                                   child_age_months: int) -> Dict:
        """Validate COPPA compliance for a family's data.

        COPPA (Children's Online Privacy Protection Act) requires:
        - Verifiable parental consent for children under 13
        - Clear notice of data collection practices
        - Right to review and delete data

        Args:
            family_id: Family identifier
            child_age_months: Current child age

        Returns:
            Dict with compliance status and any violations
        """
        violations = []

        # Check if child is under 13 (156 months)
        is_under_13 = child_age_months < 156

        if is_under_13:
            # Must have parental consent for data collection
            has_collection = self.check_consent(
                family_id, ConsentScope.DATA_COLLECTION
            )
            if not has_collection:
                violations.append(
                    "COPPA: Missing parental consent for data collection "
                    "for child under 13"
                )

            # Third-party sharing requires separate consent
            has_third_party = self.check_consent(
                family_id, ConsentScope.THIRD_PARTY_SHARING
            )
            # Not a violation if not sharing, just flag it
            third_party_status = "granted" if has_third_party else "not_granted"
        else:
            third_party_status = "not_applicable"

        return {
            'family_id': family_id,
            'child_age_months': child_age_months,
            'is_under_13': is_under_13,
            'coppa_applicable': is_under_13,
            'compliant': len(violations) == 0,
            'violations': violations,
            'third_party_consent': third_party_status,
        }

    def export_audit_trail(self, output_path: str = None) -> List[Dict]:
        """Export full audit trail as JSON.

        Args:
            output_path: Optional file path to write JSON. If None, returns data.

        Returns:
            List of all consent records as dicts
        """
        records = [
            {
                'family_id': r.family_id,
                'scope': r.scope.value,
                'action': r.action.value,
                'timestamp': r.timestamp,
                'guardian_id': r.guardian_id,
                'child_age_months': r.child_age_months,
                'expiry_date': r.expiry_date,
                'reason': r.reason,
                'record_hash': r.record_hash,
            }
            for r in self._records
        ]

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(records, f, indent=2)

        return records

    def get_summary(self) -> Dict:
        """Get summary statistics of consent state."""
        n_families = len(self._consent_state)
        scope_counts = {}

        for family_state in self._consent_state.values():
            for scope, granted in family_state.items():
                key = scope.value
                scope_counts.setdefault(key, {'granted': 0, 'revoked': 0})
                if granted:
                    scope_counts[key]['granted'] += 1
                else:
                    scope_counts[key]['revoked'] += 1

        return {
            'n_families': n_families,
            'n_records': len(self._records),
            'scope_counts': scope_counts,
        }

    def _validate_child_age(self, age_months: int):
        """Validate child age is within allowed range."""
        if age_months < self.min_child_age_months:
            raise ValueError(
                f"Child age {age_months} months is below minimum "
                f"{self.min_child_age_months} months"
            )
        if age_months > self.max_child_age_months:
            raise ValueError(
                f"Child age {age_months} months exceeds maximum "
                f"{self.max_child_age_months} months"
            )
