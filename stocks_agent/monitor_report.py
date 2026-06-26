"""HTML report artifact for portfolio thesis drift monitor runs."""

from __future__ import annotations

from html import escape
from pathlib import Path

from .monitoring_schema import MonitorRun, PortfolioAlert, ToolEvidence


def render_monitor_report(run: MonitorRun) -> str:
    """Render a self-contained HTML report for one monitor run."""

    alert_rows = "\n".join(_render_alert_row(alert) for alert in run.alerts)
    evidence_rows = "\n".join(_render_evidence_row(item) for item in run.evidence)
    severity_counts = _severity_counts(run)

    if not alert_rows:
        alert_rows = (
            "<tr><td colspan=\"6\">No thesis drift alerts were generated.</td></tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(run.portfolio_name)} - thesis drift report</title>
  <style>
    :root {{
      --bg: #f7f7f4;
      --text: #202124;
      --muted: #5f6368;
      --line: #d9d9d4;
      --panel: #ffffff;
      --accent: #174ea6;
      --risk: #b3261e;
      --review: #b06000;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 40px 24px 56px;
    }}
    h1, h2 {{
      margin: 0;
      letter-spacing: 0;
    }}
    h1 {{
      font-size: 30px;
      line-height: 1.2;
    }}
    h2 {{
      font-size: 18px;
      margin-top: 34px;
      margin-bottom: 12px;
    }}
    .meta {{
      color: var(--muted);
      margin-top: 8px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 24px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .metric strong {{
      display: block;
      font-size: 24px;
      margin-bottom: 2px;
    }}
    .metric span {{
      color: var(--muted);
      font-size: 13px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
      background: #fafaf8;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    .severity-risk_alert,
    .severity-material_change {{
      color: var(--risk);
      font-weight: 650;
    }}
    .severity-review {{
      color: var(--review);
      font-weight: 650;
    }}
    .pill {{
      display: inline-block;
      padding: 2px 7px;
      border-radius: 999px;
      background: #eef3fe;
      color: var(--accent);
      font-size: 12px;
      margin-right: 4px;
      margin-bottom: 4px;
    }}
    .disclaimer {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 28px;
    }}
    @media (max-width: 760px) {{
      .summary {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      main {{
        padding: 28px 14px 40px;
      }}
    }}
  </style>
</head>
<body>
<main>
  <h1>{escape(run.portfolio_name)}</h1>
  <div class="meta">
    Run {escape(run.run_id)} | {run.created_at_utc:%Y-%m-%d %H:%M UTC} |
    mode={escape(run.mode.value)}
  </div>

  <section class="summary">
    <div class="metric"><strong>{run.positions_checked}</strong><span>positions checked</span></div>
    <div class="metric"><strong>{len(run.alerts)}</strong><span>alerts generated</span></div>
    <div class="metric"><strong>{len(run.evidence)}</strong><span>evidence records</span></div>
    <div class="metric"><strong>{escape(severity_counts)}</strong><span>severity mix</span></div>
  </section>

  <h2>Alerts</h2>
  <table>
    <thead>
      <tr>
        <th>Ticker</th>
        <th>Severity</th>
        <th>Category</th>
        <th>Rationale</th>
        <th>Suggested action</th>
        <th>Evidence</th>
      </tr>
    </thead>
    <tbody>
      {alert_rows}
    </tbody>
  </table>

  <h2>Evidence Ledger</h2>
  <table>
    <thead>
      <tr>
        <th>ID</th>
        <th>Ticker</th>
        <th>Tool</th>
        <th>Summary</th>
      </tr>
    </thead>
    <tbody>
      {evidence_rows}
    </tbody>
  </table>

  <p class="disclaimer">
    This report is a monitoring artifact, not financial advice. Review source
    evidence and portfolio context before making any investment decision.
  </p>
</main>
</body>
</html>
"""


def write_monitor_report(run: MonitorRun, output_path: str | Path) -> Path:
    """Write a self-contained HTML report and return its path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_monitor_report(run))
    return path


def _render_alert_row(alert: PortfolioAlert) -> str:
    evidence = "".join(
        f"<span class=\"pill\">{escape(evidence_id)}</span>"
        for evidence_id in alert.evidence_ids
    )
    return f"""<tr>
  <td>{escape(alert.ticker)}</td>
  <td class="severity-{escape(alert.severity.value)}">{escape(alert.severity.value)}</td>
  <td>{escape(alert.category)}</td>
  <td><strong>{escape(alert.title)}</strong><br>{escape(alert.rationale)}</td>
  <td>{escape(alert.suggested_action)}</td>
  <td>{evidence}</td>
</tr>"""


def _render_evidence_row(item: ToolEvidence) -> str:
    return f"""<tr>
  <td>{escape(item.evidence_id)}</td>
  <td>{escape(item.ticker)}</td>
  <td>{escape(item.tool)}</td>
  <td>{escape(item.summary)}</td>
</tr>"""


def _severity_counts(run: MonitorRun) -> str:
    counts: dict[str, int] = {}
    for alert in run.alerts:
        counts[alert.severity.value] = counts.get(alert.severity.value, 0) + 1
    if not counts:
        return "none"
    return ", ".join(f"{key}: {value}" for key, value in sorted(counts.items()))

