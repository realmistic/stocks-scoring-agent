# Portfolio Thesis Drift Monitor

This harness turns the existing stock-analysis tools into a scheduled monitoring
workflow. It deliberately avoids forecasting and trading advice. The goal is to
answer one narrower question:

> Has the reason for holding or watching this stock materially changed?

## Why This Is A Harness, Not Another Agent

The current agents are useful for interactive analysis. A monitor needs a
repeatable runtime around those tools:

- explicit portfolio state;
- cost-aware tool policy;
- typed alert schema;
- run ledger;
- Telegram digest;
- self-contained HTML report artifact;
- deterministic outcome grader for alert/evidence quality.

This makes the system easier to test, debug, and productize than a generic
chatbot that re-reads everything on every prompt.

## Daily Light Run

The default run should be cheap:

- basic company info;
- recent price/momentum summary;
- latest news headlines;
- configured thesis triggers.

Heavy tools such as SEC filing analysis, paid social search, or full structured
LLM output should run only when the light run crosses a trigger.

## Alert Policy

Alerts are not buy/sell recommendations. They are review prompts:

- valuation crossed a configured threshold;
- price moved far enough to revisit thesis;
- target upside dropped below threshold;
- monitored news keywords appeared.

Each alert includes evidence IDs so the user can trace why it fired.

## Portfolio Config

The monitor reads a JSON portfolio config. The sample lives at
`examples/portfolio.json`.

```json
{
  "name": "Sample thesis drift portfolio",
  "tool_policy": {
    "light_price_period": "5d",
    "news_limit": 5,
    "max_tools_per_ticker": 4
  },
  "positions": [
    {
      "ticker": "NVDA",
      "thesis": "AI data-center demand supports premium valuation...",
      "max_distance_from_high_pct": 15,
      "max_pe_ratio": 80,
      "min_target_upside_pct": 10,
      "watch_keywords": ["export restriction", "data-center demand"]
    }
  ]
}
```

Key fields:

| Field | Purpose |
|---|---|
| `positions[].ticker` | Stock ticker to monitor. |
| `positions[].thesis` | Human-readable reason for holding or watching the stock. |
| `max_distance_from_high_pct` | Trigger a review if price is this far below the monitored-period high. |
| `max_pe_ratio` | Trigger a valuation review if trailing PE exceeds this level. |
| `min_target_upside_pct` | Trigger a review if target-price upside drops below this level. |
| `watch_keywords` | Trigger a news review when latest headlines/summaries match these phrases. |
| `tool_policy.news_limit` | Maximum news items to fetch during the light run. |
| `tool_policy.max_tools_per_ticker` | Hard budget for light-run tools per ticker. |
| `heavy_*_on_trigger` | Reserved switches for later heavy analysis gates. They are documented now but not executed yet. |

The current config format is JSON to avoid adding a YAML dependency.

## Example

```bash
python -m stocks_agent.portfolio_monitor \
  --portfolio examples/portfolio.json \
  --sample-data
```

To append a JSONL run ledger:

```bash
python -m stocks_agent.portfolio_monitor \
  --portfolio examples/portfolio.json \
  --sample-data \
  --write-ledger
```

To write a self-contained HTML report:

```bash
python -m stocks_agent.portfolio_monitor \
  --portfolio examples/portfolio.json \
  --sample-data \
  --report-html outputs/report.html
```

To include the local outcome grader:

```bash
python -m stocks_agent.portfolio_monitor \
  --portfolio examples/portfolio.json \
  --sample-data \
  --grade
```

To format a Telegram digest:

```python
from stocks_agent.portfolio_monitor import load_portfolio_config, run_monitor
from stocks_agent.portfolio_monitor import SampleMarketDataProvider
from stocks_agent.telegram_digest import format_telegram_digest

config = load_portfolio_config("examples/portfolio.json")
run = run_monitor(config, provider=SampleMarketDataProvider())
print(format_telegram_digest(run))
```

Sending requires `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.

## Outputs

| Output | How to produce it | Use |
|---|---|---|
| Structured JSON | default CLI output | Machine-readable monitor result. |
| JSONL ledger | `--write-ledger` | Append-only audit trail under `runs/YYYY-MM-DD.jsonl`. |
| HTML report | `--report-html outputs/report.html` | Human-readable run artifact with alerts and evidence. |
| Telegram digest text | `format_telegram_digest(run)` | Compact alert message for a bot or manual send. |
| Outcome grade | `--grade` or `grade_monitor_run(run)` | Regression gate for alert/evidence quality. |

## Relation To Data Analyst Agent Patterns

The Anthropic `managed_agents/data_analyst_agent.ipynb` cookbook uses a
different stack: Claude Managed Agents, hosted environments, mounted resources,
streamed events, and Files API outputs. This repo uses local Python modules and
the existing OpenAI Agents SDK tools, so the runtime should not be copied
directly.

The useful transferable pattern is the output contract: every run should produce
inspectable artifacts, not only a chat response. This harness therefore writes
structured JSON, optional JSONL ledger records, Telegram digests, and an optional
self-contained HTML report.

## Outcome Grader

The Anthropic `CMA_verify_with_outcome_grader.ipynb` cookbook uses a second
agent to verify an artifact against a rubric. This repo does not use that hosted
runtime, so the first grader is deterministic and local.

The grader checks production properties:

- every alert has title, rationale, suggested action, and evidence IDs;
- every evidence ID resolves to the run evidence ledger;
- alert evidence belongs to the same ticker;
- alert category is supported by the expected tool family;
- category-specific data is present, such as `pe_ratio` for valuation alerts or
  `distance_from_high_pct` for price-action alerts.

This does not prove the investment thesis is right. It prevents a weaker failure
mode: notifying a user with an untraceable or unsupported alert.

The grade output contains:

- `passed`: true when there are no failed rubric checks;
- `score`: 0.0-1.0 score penalized for failed checks and warnings;
- `summary`: one-line result;
- `findings`: per-criterion rubric findings.

## Verification

Run the test suite:

```bash
uv run python -m unittest discover -s tests
```

Run a full offline smoke test with sample data, HTML report, and grade:

```bash
uv run python -m stocks_agent.portfolio_monitor \
  --portfolio examples/portfolio.json \
  --sample-data \
  --report-html outputs/report.html \
  --grade
```

Compile-check the modules:

```bash
uv run python -m compileall stocks_agent tests
```

## Current Limits

This is a first harness slice, not a complete Telegram SaaS product:

- no scheduler is included yet;
- no Telegram polling/webhook bot loop is included yet;
- heavy tools are represented in policy but not automatically executed yet;
- alerts are threshold/rule based, not predictive;
- the grader checks evidence quality, not investment correctness.
