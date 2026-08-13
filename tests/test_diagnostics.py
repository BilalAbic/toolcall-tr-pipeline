from __future__ import annotations

import pytest
from pydantic import ValidationError

from toolcall_tr.diagnostics import Diagnostic, UnknownDiagnosticCode, diagnostic


def test_known_code_populates_stable_semantics() -> None:
    item = diagnostic("PARSE_INVALID_JSON", "bad JSON")
    assert (item.stage, item.severity, item.retryable) == ("ingest", "error", False)


def test_unknown_code_is_rejected_fail_closed() -> None:
    with pytest.raises(UnknownDiagnosticCode):
        diagnostic("UNKNOWN_NEW_CODE", "must not pass")


def test_envelope_cannot_change_catalog_meaning() -> None:
    with pytest.raises(ValidationError):
        Diagnostic(
            code="PARSE_INVALID_JSON",
            stage="registry",
            severity="warning",
            source_occurrence_id=None,
            episode_id=None,
            json_pointer=None,
            source_line=None,
            message="bad",
            retryable=True,
            evidence_refs=[],
        )
