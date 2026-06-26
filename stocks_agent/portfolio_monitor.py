"""Portfolio thesis drift monitoring harness.

This module intentionally avoids forecasting. It turns the existing data tools
into a scheduled monitoring loop that highlights material changes against an
explicit portfolio thesis.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Protocol

from .monitoring_schema import (
    AlertSeverity,
    MonitorMode,
    MonitorRun,
    PortfolioAlert,
    PortfolioConfig,
    PortfolioPosition,
    ToolEvidence,
    ToolPolicy,
)
from .monitor_grader import grade_monitor_run
from .monitor_report import write_monitor_report


class MarketDataProvider(Protocol):
    """Small interface used by the monitor and tests."""

    def get_company_info_basic(self, ticker: str) -> dict[str, Any]:
        ...

    def get_historical_prices(
        self, ticker: str, period: str = "5d", interval: str = "1d"
    ) -> dict[str, Any]:
        ...

    def get_ticker_news(self, ticker: str, limit: int = 5) -> dict[str, Any]:
        ...


class DefaultMarketDataProvider:
    """Adapter over existing stock tools."""

    def get_company_info_basic(self, ticker: str) -> dict[str, Any]:
        from .tools import get_company_info_basic

        return get_company_info_basic(ticker)

    def get_historical_prices(
        self, ticker: str, period: str = "5d", interval: str = "1d"
    ) -> dict[str, Any]:
        from .tools import get_historical_prices

        return get_historical_prices(ticker, period=period, interval=interval)

    def get_ticker_news(self, ticker: str, limit: int = 5) -> dict[str, Any]:
        from .tools import get_ticker_news

        return get_ticker_news(ticker, limit=limit)


class SampleMarketDataProvider:
    """Offline sample provider for demos and smoke tests."""

    def get_company_info_basic(self, ticker: str) -> dict[str, Any]:
        data = {
            "NVDA": {
                "ticker": "NVDA",
                "name": "NVIDIA Corporation",
                "pe_ratio": 86.2,
                "current_price": 126.0,
                "target_price": 142.0,
                "recommendation": "buy",
            },
            "GOOG": {
                "ticker": "GOOG",
                "name": "Alphabet Inc.",
                "pe_ratio": 21.4,
                "current_price": 184.0,
                "target_price": 210.0,
                "recommendation": "buy",
            },
        }
        return data.get(ticker.upper(), {"ticker": ticker, "pe_ratio": "N/A"})

    def get_historical_prices(
        self, ticker: str, period: str = "5d", interval: str = "1d"
    ) -> dict[str, Any]:
        data = {
            "NVDA": {
                "ticker": "NVDA",
                "period": period,
                "current_price": 126.0,
                "period_high": 151.0,
                "distance_from_high_pct": -16.56,
                "momentum": "negative",
            },
            "GOOG": {
                "ticker": "GOOG",
                "period": period,
                "current_price": 184.0,
                "period_high": 190.0,
                "distance_from_high_pct": -3.16,
                "momentum": "mixed",
            },
        }
        return data.get(
            ticker.upper(),
            {
                "ticker": ticker,
                "period": period,
                "distance_from_high_pct": 0.0,
                "momentum": "unknown",
            },
        )

    def get_ticker_news(self, ticker: str, limit: int = 5) -> dict[str, Any]:
        data = {
            "NVDA": [
                {
                    "title": "Nvidia faces export restriction concern",
                    "publisher": "Sample News",
                    "summary": "Export restrictions may affect data-center demand.",
                }
            ],
            "GOOG": [
                {
                    "title": "Alphabet cloud demand remains stable",
                    "publisher": "Sample News",
                    "summary": "Cloud and advertising demand remain steady.",
                }
            ],
        }
        news = data.get(ticker.upper(), [])[:limit]
        return {"ticker": ticker, "news_count": len(news), "news": news}


def load_portfolio_config(path: str | Path) -> PortfolioConfig:
    """Load portfolio monitor configuration from JSON."""

    raw = Path(path).read_text()
    return PortfolioConfig.model_validate_json(raw)


def run_monitor(
    config: PortfolioConfig,
    provider: MarketDataProvider | None = None,
    mode: MonitorMode = MonitorMode.LIGHT,
) -> MonitorRun:
    """Run thesis drift monitoring for every position in the portfolio."""

    provider = provider or DefaultMarketDataProvider()
    evidence: list[ToolEvidence] = []
    alerts: list[PortfolioAlert] = []

    for position in config.positions:
        position_evidence, position_alerts = analyze_position(
            position=position,
            provider=provider,
            policy=config.tool_policy,
        )
        evidence.extend(position_evidence)
        alerts.extend(position_alerts)

    alerts = alerts[: config.tool_policy.max_alerts_per_run]

    return MonitorRun(
        run_id=str(uuid.uuid4()),
        portfolio_name=config.name,
        mode=mode,
        positions_checked=len(config.positions),
        alerts=alerts,
        evidence=evidence,
        tool_policy=config.tool_policy,
    )


def analyze_position(
    position: PortfolioPosition,
    provider: MarketDataProvider,
    policy: ToolPolicy,
) -> tuple[list[ToolEvidence], list[PortfolioAlert]]:
    """Analyze a single position for material thesis drift."""

    ticker = position.ticker.upper()
    evidence: list[ToolEvidence] = []
    alerts: list[PortfolioAlert] = []

    info = provider.get_company_info_basic(ticker)
    info_evidence_id = f"{ticker}-company-info"
    evidence.append(
        ToolEvidence(
            evidence_id=info_evidence_id,
            ticker=ticker,
            tool="get_company_info_basic",
            summary=_summarize_company_info(info),
            data=_compact_dict(info),
        )
    )

    prices = provider.get_historical_prices(
        ticker,
        period=policy.light_price_period,
        interval=policy.light_price_interval,
    )
    price_evidence_id = f"{ticker}-price"
    evidence.append(
        ToolEvidence(
            evidence_id=price_evidence_id,
            ticker=ticker,
            tool="get_historical_prices",
            summary=_summarize_price_action(prices),
            data=_compact_dict(prices),
        )
    )

    news: dict[str, Any] = {"ticker": ticker, "news_count": 0, "news": []}
    tools_used = len(evidence)
    if policy.news_limit > 0 and tools_used < policy.max_tools_per_ticker:
        news = provider.get_ticker_news(ticker, limit=policy.news_limit)
        news_evidence_id = f"{ticker}-news"
        evidence.append(
            ToolEvidence(
                evidence_id=news_evidence_id,
                ticker=ticker,
                tool="get_ticker_news",
                summary=_summarize_news(news),
                data=_compact_dict(news),
            )
        )
    else:
        news_evidence_id = ""

    pe_ratio = _as_float(info.get("pe_ratio"))
    if position.max_pe_ratio is not None and pe_ratio is not None:
        if pe_ratio > position.max_pe_ratio:
            alerts.append(
                PortfolioAlert(
                    ticker=ticker,
                    severity=AlertSeverity.REVIEW,
                    category="valuation",
                    title=f"{ticker} valuation above thesis threshold",
                    rationale=(
                        f"Trailing PE is {pe_ratio:.1f}, above configured "
                        f"threshold {position.max_pe_ratio:.1f}."
                    ),
                    suggested_action=(
                        "Review whether the original valuation thesis still holds."
                    ),
                    evidence_ids=[info_evidence_id],
                    confidence="medium",
                )
            )

    distance_from_high = _as_float(prices.get("distance_from_high_pct"))
    if (
        position.max_distance_from_high_pct is not None
        and distance_from_high is not None
    ):
        drawdown = abs(distance_from_high)
        if distance_from_high < 0 and drawdown >= position.max_distance_from_high_pct:
            alerts.append(
                PortfolioAlert(
                    ticker=ticker,
                    severity=AlertSeverity.MATERIAL_CHANGE,
                    category="price_action",
                    title=f"{ticker} moved far enough to revisit thesis",
                    rationale=(
                        f"Price is {drawdown:.1f}% below the monitored-period "
                        f"high, crossing threshold "
                        f"{position.max_distance_from_high_pct:.1f}%."
                    ),
                    suggested_action=(
                        "Check if the move is noise, valuation reset, or thesis drift."
                    ),
                    evidence_ids=[price_evidence_id],
                    confidence="medium",
                )
            )

    current_price = _as_float(info.get("current_price"))
    target_price = _as_float(info.get("target_price"))
    if (
        position.min_target_upside_pct is not None
        and current_price
        and target_price
    ):
        upside = (target_price - current_price) / current_price * 100
        if upside < position.min_target_upside_pct:
            alerts.append(
                PortfolioAlert(
                    ticker=ticker,
                    severity=AlertSeverity.REVIEW,
                    category="analyst_target",
                    title=f"{ticker} target upside below threshold",
                    rationale=(
                        f"Implied target upside is {upside:.1f}%, below "
                        f"configured threshold {position.min_target_upside_pct:.1f}%."
                    ),
                    suggested_action="Review target-price support and current upside.",
                    evidence_ids=[info_evidence_id],
                    confidence="low",
                )
            )

    keyword_hits = _find_keyword_hits(news, position.watch_keywords)
    if keyword_hits:
        alerts.append(
            PortfolioAlert(
                ticker=ticker,
                severity=AlertSeverity.REVIEW,
                category="news_keyword",
                title=f"{ticker} news matched monitored thesis keywords",
                rationale=f"Matched keywords: {', '.join(sorted(keyword_hits))}.",
                suggested_action=(
                    "Review the latest news before triggering heavy analysis."
                ),
                evidence_ids=[news_evidence_id] if news_evidence_id else [],
                confidence="medium",
            )
        )

    return evidence, alerts


def write_ledger(run: MonitorRun, ledger_dir: str | Path = "runs") -> Path:
    """Append a monitor run to a date-partitioned JSONL ledger."""

    ledger_path = Path(ledger_dir)
    ledger_path.mkdir(parents=True, exist_ok=True)
    day = run.created_at_utc.strftime("%Y-%m-%d")
    output_path = ledger_path / f"{day}.jsonl"
    with output_path.open("a") as handle:
        handle.write(run.model_dump_json() + "\n")
    return output_path


def _as_float(value: Any) -> float | None:
    if value in (None, "N/A", ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compact_dict(data: dict[str, Any], max_string: int = 500) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str) and len(value) > max_string:
            compact[key] = value[:max_string] + "..."
        elif isinstance(value, list):
            compact[key] = value[:5]
        else:
            compact[key] = value
    return compact


def _summarize_company_info(info: dict[str, Any]) -> str:
    parts = [
        f"name={info.get('name', 'N/A')}",
        f"price={info.get('current_price', 'N/A')}",
        f"pe={info.get('pe_ratio', 'N/A')}",
        f"target={info.get('target_price', 'N/A')}",
        f"recommendation={info.get('recommendation', 'N/A')}",
    ]
    return "; ".join(parts)


def _summarize_price_action(prices: dict[str, Any]) -> str:
    return (
        f"period={prices.get('period', 'N/A')}; "
        f"current={prices.get('current_price', 'N/A')}; "
        f"distance_from_high_pct={prices.get('distance_from_high_pct', 'N/A')}; "
        f"momentum={prices.get('momentum', 'N/A')}"
    )


def _summarize_news(news: dict[str, Any]) -> str:
    count = news.get("news_count", 0)
    titles = [
        item.get("title", "")
        for item in news.get("news", [])
        if isinstance(item, dict)
    ][:3]
    return f"news_count={count}; titles={titles}"


def _find_keyword_hits(news: dict[str, Any], keywords: list[str]) -> set[str]:
    if not keywords:
        return set()

    haystack_parts: list[str] = []
    for item in news.get("news", []):
        if not isinstance(item, dict):
            continue
        haystack_parts.extend(
            [
                str(item.get("title", "")),
                str(item.get("summary", "")),
                str(item.get("description", "")),
            ]
        )
    haystack = " ".join(haystack_parts).lower()

    return {keyword for keyword in keywords if keyword.lower() in haystack}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run portfolio thesis drift monitor.")
    parser.add_argument(
        "--portfolio",
        default="examples/portfolio.json",
        help="Path to portfolio JSON config.",
    )
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Use offline sample market data instead of live tools.",
    )
    parser.add_argument(
        "--write-ledger",
        action="store_true",
        help="Append run output to runs/YYYY-MM-DD.jsonl.",
    )
    parser.add_argument(
        "--ledger-dir",
        default="runs",
        help="Directory for JSONL run ledger.",
    )
    parser.add_argument(
        "--report-html",
        help="Optional path for a self-contained HTML report artifact.",
    )
    parser.add_argument(
        "--grade",
        action="store_true",
        help="Run an outcome grader and include the grade in JSON output.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = load_portfolio_config(args.portfolio)
    provider: MarketDataProvider | None = (
        SampleMarketDataProvider() if args.sample_data else None
    )
    mode = MonitorMode.SAMPLE if args.sample_data else MonitorMode.LIGHT
    run = run_monitor(config=config, provider=provider, mode=mode)

    if args.write_ledger:
        output_path = write_ledger(run, args.ledger_dir)
        print(f"Wrote ledger record to {output_path}")

    if args.report_html:
        report_path = write_monitor_report(run, args.report_html)
        print(f"Wrote HTML report to {report_path}")

    output: dict[str, Any] = {"run": run.model_dump(mode="json")}
    if args.grade:
        grade = grade_monitor_run(run)
        output["grade"] = grade.model_dump(mode="json")

    print(json.dumps(output if args.grade else output["run"], indent=2))


if __name__ == "__main__":
    main()
