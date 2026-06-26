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
- eval/grader hooks in a later iteration.

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
