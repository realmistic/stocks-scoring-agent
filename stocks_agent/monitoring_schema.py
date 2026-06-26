"""Schemas for portfolio thesis drift monitoring."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class MonitorMode(str, Enum):
    """How aggressively the monitor should use tools."""

    LIGHT = "light"
    HEAVY = "heavy"
    SAMPLE = "sample"


class AlertSeverity(str, Enum):
    """User-facing alert severity."""

    INFO = "info"
    REVIEW = "review"
    MATERIAL_CHANGE = "material_change"
    RISK_ALERT = "risk_alert"


class ToolPolicy(BaseModel):
    """Cost-aware tool usage policy for one monitor run."""

    light_price_period: str = "5d"
    light_price_interval: str = "1d"
    news_limit: int = Field(default=5, ge=0, le=20)
    max_tools_per_ticker: int = Field(default=4, ge=1, le=20)
    max_alerts_per_run: int = Field(default=10, ge=1, le=100)
    heavy_sec_on_trigger: bool = False
    heavy_social_on_trigger: bool = False
    heavy_structured_output_on_trigger: bool = False


class PortfolioPosition(BaseModel):
    """A monitored holding or watchlist item."""

    ticker: str
    thesis: str
    notes: str | None = None
    max_distance_from_high_pct: float | None = Field(
        default=None,
        description="Trigger if price is at least this far below period high.",
    )
    max_pe_ratio: float | None = Field(
        default=None,
        description="Trigger if trailing PE rises above this value.",
    )
    min_target_upside_pct: float | None = Field(
        default=None,
        description="Trigger if analyst target upside drops below this value.",
    )
    watch_keywords: list[str] = Field(
        default_factory=list,
        description="News keywords that should trigger review when present.",
    )


class PortfolioConfig(BaseModel):
    """Portfolio monitor configuration."""

    name: str = "Portfolio thesis drift monitor"
    base_currency: str = "USD"
    positions: list[PortfolioPosition]
    tool_policy: ToolPolicy = Field(default_factory=ToolPolicy)


class ToolEvidence(BaseModel):
    """One compact record of a tool call and the key facts it produced."""

    evidence_id: str
    ticker: str
    tool: str
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)


class PortfolioAlert(BaseModel):
    """A material thesis drift alert."""

    ticker: str
    severity: AlertSeverity
    category: str
    title: str
    rationale: str
    suggested_action: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"


class MonitorRun(BaseModel):
    """Full run output for audit, Telegram digest, or JSONL ledger."""

    run_id: str
    created_at_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    portfolio_name: str
    mode: MonitorMode
    positions_checked: int
    alerts: list[PortfolioAlert]
    evidence: list[ToolEvidence]
    tool_policy: ToolPolicy

