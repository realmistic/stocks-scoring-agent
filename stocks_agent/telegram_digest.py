"""Telegram digest helpers for portfolio thesis drift alerts."""

from __future__ import annotations

import os
from typing import Any

import requests

from .monitoring_schema import MonitorRun


def format_telegram_digest(run: MonitorRun, max_alerts: int = 10) -> str:
    """Format monitor output as a compact Telegram message."""

    header = (
        f"Portfolio thesis drift monitor\n"
        f"{run.portfolio_name} | {run.created_at_utc:%Y-%m-%d %H:%M UTC}\n"
        f"Checked: {run.positions_checked} | Alerts: {len(run.alerts)}"
    )

    if not run.alerts:
        return (
            f"{header}\n\n"
            "No material thesis drift detected by the configured light monitor."
        )

    lines = [header, ""]
    for index, alert in enumerate(run.alerts[:max_alerts], 1):
        lines.extend(
            [
                f"{index}. {alert.ticker} | {alert.severity.value} | {alert.category}",
                alert.title,
                alert.rationale,
                f"Action: {alert.suggested_action}",
                "",
            ]
        )

    if len(run.alerts) > max_alerts:
        lines.append(f"...and {len(run.alerts) - max_alerts} more alerts.")

    lines.append("Not financial advice. Review evidence before acting.")
    return "\n".join(lines).strip()


def send_telegram_digest(
    text: str,
    bot_token: str | None = None,
    chat_id: str | None = None,
    timeout: int = 10,
) -> dict[str, Any]:
    """Send a digest through Telegram using env vars or explicit credentials."""

    token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    target_chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not target_chat_id:
        raise ValueError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")

    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": target_chat_id, "text": text},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()

