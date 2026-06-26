"""Outcome grader for portfolio thesis drift monitor runs.

The grader is intentionally deterministic. It does not judge whether an
investment thesis is correct; it verifies that the monitor produced traceable,
well-supported alerts that can be audited before notifying a user.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .monitoring_schema import MonitorRun, PortfolioAlert, ToolEvidence


class GradeStatus(str, Enum):
    """Single rubric finding status."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class GradeFinding(BaseModel):
    """One rubric finding from the monitor grader."""

    status: GradeStatus
    criterion: str
    message: str
    ticker: str | None = None
    alert_index: int | None = None
    evidence_id: str | None = None


class MonitorRunGrade(BaseModel):
    """Aggregate grade for one monitor run."""

    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    summary: str
    findings: list[GradeFinding]


CATEGORY_TOOL_SUPPORT = {
    "valuation": {"get_company_info_basic", "get_company_info"},
    "analyst_target": {"get_company_info_basic", "get_company_info"},
    "price_action": {"get_historical_prices"},
    "news_keyword": {"get_ticker_news", "search_news_by_ticker"},
}


def grade_monitor_run(run: MonitorRun) -> MonitorRunGrade:
    """Grade a monitor run against a compact evidence-quality rubric."""

    findings: list[GradeFinding] = []
    evidence_by_id = _index_evidence(run.evidence, findings)

    if run.positions_checked < 0:
        findings.append(
            GradeFinding(
                status=GradeStatus.FAIL,
                criterion="run_metadata",
                message="positions_checked cannot be negative.",
            )
        )

    if run.alerts and not run.evidence:
        findings.append(
            GradeFinding(
                status=GradeStatus.FAIL,
                criterion="evidence_ledger",
                message="Run produced alerts without an evidence ledger.",
            )
        )

    if not run.alerts:
        findings.append(
            GradeFinding(
                status=GradeStatus.PASS,
                criterion="no_alerts",
                message="No alerts were generated; no alert evidence to verify.",
            )
        )

    for index, alert in enumerate(run.alerts):
        _grade_alert_shape(alert, index, findings)
        _grade_alert_evidence(alert, index, evidence_by_id, findings)

    fail_count = sum(1 for finding in findings if finding.status == GradeStatus.FAIL)
    warn_count = sum(1 for finding in findings if finding.status == GradeStatus.WARN)
    score = max(0.0, 1.0 - fail_count * 0.25 - warn_count * 0.1)
    passed = fail_count == 0

    summary = (
        f"{'PASS' if passed else 'FAIL'}: "
        f"{fail_count} failed checks, {warn_count} warnings, score={score:.2f}"
    )
    return MonitorRunGrade(
        passed=passed,
        score=round(score, 2),
        summary=summary,
        findings=findings,
    )


def _index_evidence(
    evidence: list[ToolEvidence],
    findings: list[GradeFinding],
) -> dict[str, ToolEvidence]:
    evidence_by_id: dict[str, ToolEvidence] = {}
    for item in evidence:
        if item.evidence_id in evidence_by_id:
            findings.append(
                GradeFinding(
                    status=GradeStatus.FAIL,
                    criterion="evidence_id_unique",
                    message="Duplicate evidence_id in run evidence ledger.",
                    ticker=item.ticker,
                    evidence_id=item.evidence_id,
                )
            )
        evidence_by_id[item.evidence_id] = item
    return evidence_by_id


def _grade_alert_shape(
    alert: PortfolioAlert,
    index: int,
    findings: list[GradeFinding],
) -> None:
    required_text = {
        "title": alert.title,
        "rationale": alert.rationale,
        "suggested_action": alert.suggested_action,
    }
    for field_name, value in required_text.items():
        if not value.strip():
            findings.append(
                GradeFinding(
                    status=GradeStatus.FAIL,
                    criterion="alert_required_text",
                    message=f"Alert {field_name} is empty.",
                    ticker=alert.ticker,
                    alert_index=index,
                )
            )

    if not alert.evidence_ids:
        findings.append(
            GradeFinding(
                status=GradeStatus.FAIL,
                criterion="alert_evidence_required",
                message="Alert has no evidence_ids.",
                ticker=alert.ticker,
                alert_index=index,
            )
        )


def _grade_alert_evidence(
    alert: PortfolioAlert,
    index: int,
    evidence_by_id: dict[str, ToolEvidence],
    findings: list[GradeFinding],
) -> None:
    resolved_evidence: list[ToolEvidence] = []
    for evidence_id in alert.evidence_ids:
        item = evidence_by_id.get(evidence_id)
        if item is None:
            findings.append(
                GradeFinding(
                    status=GradeStatus.FAIL,
                    criterion="alert_evidence_resolves",
                    message="Alert references missing evidence_id.",
                    ticker=alert.ticker,
                    alert_index=index,
                    evidence_id=evidence_id,
                )
            )
            continue

        resolved_evidence.append(item)
        if item.ticker.upper() != alert.ticker.upper():
            findings.append(
                GradeFinding(
                    status=GradeStatus.FAIL,
                    criterion="alert_evidence_ticker_match",
                    message="Alert references evidence for a different ticker.",
                    ticker=alert.ticker,
                    alert_index=index,
                    evidence_id=evidence_id,
                )
            )

    if not resolved_evidence:
        return

    supported_tools = CATEGORY_TOOL_SUPPORT.get(alert.category)
    if supported_tools is None:
        findings.append(
            GradeFinding(
                status=GradeStatus.WARN,
                criterion="alert_category_known",
                message=f"Unknown alert category '{alert.category}'.",
                ticker=alert.ticker,
                alert_index=index,
            )
        )
        return

    tool_match = any(item.tool in supported_tools for item in resolved_evidence)
    if not tool_match:
        findings.append(
            GradeFinding(
                status=GradeStatus.FAIL,
                criterion="alert_category_tool_support",
                message=(
                    f"Alert category '{alert.category}' is not supported by "
                    f"the referenced evidence tools."
                ),
                ticker=alert.ticker,
                alert_index=index,
            )
        )
        return

    _grade_category_specific_evidence(alert, index, resolved_evidence, findings)


def _grade_category_specific_evidence(
    alert: PortfolioAlert,
    index: int,
    evidence_items: list[ToolEvidence],
    findings: list[GradeFinding],
) -> None:
    if alert.category == "valuation":
        if not _has_numeric_field(evidence_items, "pe_ratio"):
            findings.append(
                _category_fail(alert, index, "valuation", "Missing numeric pe_ratio.")
            )
    elif alert.category == "price_action":
        if not _has_numeric_field(evidence_items, "distance_from_high_pct"):
            findings.append(
                _category_fail(
                    alert,
                    index,
                    "price_action",
                    "Missing numeric distance_from_high_pct.",
                )
            )
    elif alert.category == "analyst_target":
        if not (
            _has_numeric_field(evidence_items, "current_price")
            and _has_numeric_field(evidence_items, "target_price")
        ):
            findings.append(
                _category_fail(
                    alert,
                    index,
                    "analyst_target",
                    "Missing numeric current_price or target_price.",
                )
            )
    elif alert.category == "news_keyword":
        has_news = any(
            _as_float(item.data.get("news_count")) not in (None, 0)
            or bool(item.data.get("news"))
            for item in evidence_items
        )
        if not has_news:
            findings.append(
                _category_fail(alert, index, "news_keyword", "Missing news evidence.")
            )


def _category_fail(
    alert: PortfolioAlert,
    index: int,
    criterion: str,
    message: str,
) -> GradeFinding:
    return GradeFinding(
        status=GradeStatus.FAIL,
        criterion=criterion,
        message=message,
        ticker=alert.ticker,
        alert_index=index,
    )


def _has_numeric_field(evidence_items: list[ToolEvidence], field_name: str) -> bool:
    return any(_as_float(item.data.get(field_name)) is not None for item in evidence_items)


def _as_float(value: Any) -> float | None:
    if value in (None, "N/A", ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

