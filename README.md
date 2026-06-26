# 📈 Stock Analysis Agents

AI-powered institutional-grade stock analysis with SEC filings, social sentiment, and comprehensive market data. Built with OpenAI Agents SDK (gpt-5.4-mini).

## 🎯 Overview

Four intelligent agent types for different analysis needs:

- **SimpleAgent** - Stateless Q&A for quick lookups (OpenAI API)
- **ConversationAgent** - Memory-enabled multi-turn conversations with auto-ticker tracking (OpenAI API)
- **StructuredAgent** - Returns narrative + structured JSON (30 fields) using Pydantic Structured Outputs (OpenAI API)
- **FreeAgent** - Same functionality as SimpleAgent but uses free local Ollama models (Qwen, Llama, etc.)

**17 analysis tools** covering fundamentals, SEC filings, social sentiment, earnings, news, and screening.

Also includes a **Portfolio Thesis Drift Monitor** harness for scheduled,
cost-aware monitoring of a portfolio without making buy/sell predictions.

## ✨ Key Features

✅ **SEC Filing Analysis** - 10-K/10-Q with period comparisons  
✅ **Social Sentiment** - High-engagement Twitter/X and Reddit analysis  
✅ **Real-time Market Data** - Yahoo Finance integration  
✅ **EPS Trend Analysis** - Historical tracking with analyst revisions  
✅ **Comprehensive Earnings** - Estimates, revisions, growth projections  
✅ **Advanced Screening** - Value/growth company filters  
✅ **Structured Outputs** - 30-field Pydantic models for programmatic use  
✅ **Conversational Memory** - Context-aware follow-up questions  

## 🚀 Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Set up environment variables
cp .envrc.example .envrc
# Edit .envrc with your API keys:
#   - OPENAI_API_KEY (required)
#   - SEC_IDENTITY_EMAIL (required for SEC filings)
#   - POLYGON_API_KEY (optional)
direnv allow .

# 3. Activate and test
source .venv/bin/activate
python -c "from stocks_agent import SimpleAgent; print('✅ Ready!')"
```

## 🤖 Agents Usage

### SimpleAgent
Stateless - each query is independent.

```python
from stocks_agent import SimpleAgent

agent = SimpleAgent(model="gpt-5.4-mini")
response = await agent.ask("What's AAPL's PE ratio and recent SEC filings?")
```

### ConversationAgent
Maintains context across questions.

```python
from stocks_agent import ConversationAgent

agent = ConversationAgent(track_tickers=True)
await agent.ask("Analyze TSLA's latest 10-Q filing")
await agent.ask("What about social sentiment?")  # Auto-knows TSLA
await agent.ask("Compare to competitors")        # Still TSLA

agent.switch_to("AAPL")  # Switch ticker
agent.reset()            # Clear history
```

### StructuredAgent
Returns both text analysis AND structured data (30 fields).

```python
from stocks_agent import StructuredAgent

agent = StructuredAgent()
text, data = await agent.analyze('NVDA')

# Access structured fields
print(data['pe_ratio'])                  # 45.57
print(data['eps_trend_direction'])       # "improving"
print(data['valuation_level'])           # "expensive"
print(data['analyst_sentiment'])         # "improving"
print(data['social_sentiment'])          # "Bullish discussions..."
print(data['recommendation'])            # "buy"
print(data['confidence_score'])          # 8/10
```

### FreeAgent
Uses free local Ollama models instead of OpenAI API. No API costs!

```python
from stocks_agent import FreeAgent

# Setup (one-time)
# 1. Install Ollama: https://ollama.com/download
# 2. Pull model: `ollama pull qwen3:32b`  
# 3. Start Ollama: `ollama serve`

agent = FreeAgent(model='qwen3:32b')
response = await agent.ask("Analyze AAPL's valuation")

# Check status
print(agent.get_status())  # ✅ Connected to Ollama
print(agent.list_models()) # See recommended models
```

**Recommended Models:**
- `qwen3:32b` (best) - Excellent tool calling, requires 32GB+ RAM
- `qwen3:14b` (good) - Solid performance, requires 8GB+ RAM  
- `llama3.1:8b` (fast) - Lightweight, requires 4GB+ RAM

### Portfolio Thesis Drift Monitor
Runs a light daily portfolio check and emits review alerts when a configured
holding crosses valuation, price-action, target-upside, or news-keyword
thresholds.

```bash
# Offline smoke test with sample market data
python -m stocks_agent.portfolio_monitor \
  --portfolio examples/portfolio.json \
  --sample-data

# Append an auditable JSONL run ledger
python -m stocks_agent.portfolio_monitor \
  --portfolio examples/portfolio.json \
  --sample-data \
  --write-ledger

# Produce a self-contained HTML report artifact
python -m stocks_agent.portfolio_monitor \
  --portfolio examples/portfolio.json \
  --sample-data \
  --report-html outputs/report.html

# Include the deterministic outcome grader
python -m stocks_agent.portfolio_monitor \
  --portfolio examples/portfolio.json \
  --sample-data \
  --grade
```

Format alerts for Telegram:

```python
from stocks_agent.portfolio_monitor import SampleMarketDataProvider
from stocks_agent.portfolio_monitor import load_portfolio_config, run_monitor
from stocks_agent.telegram_digest import format_telegram_digest

config = load_portfolio_config("examples/portfolio.json")
run = run_monitor(config, provider=SampleMarketDataProvider())
print(format_telegram_digest(run))
```

Sending the digest requires `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
See `docs/portfolio-thesis-drift-monitor.md` for the harness design.

Check Telegram delivery:

```bash
export TELEGRAM_BOT_TOKEN='your-telegram-bot-token-here'
export TELEGRAM_CHAT_ID='your-telegram-chat-id-here'

uv run python - <<'PY'
from stocks_agent.portfolio_monitor import (
    SampleMarketDataProvider,
    load_portfolio_config,
    run_monitor,
)
from stocks_agent.telegram_digest import format_telegram_digest, send_telegram_digest

config = load_portfolio_config("examples/portfolio.json")
run = run_monitor(config, provider=SampleMarketDataProvider())
text = format_telegram_digest(run)
print(send_telegram_digest(text))
PY
```

Run monitor tests:

```bash
uv run python -m unittest discover -s tests
```

## 🛠️ Available Tools (17)

### 📊 Core Fundamentals (7 tools)
| Tool | Description |
|------|-------------|
| `get_company_info_basic` | Essential metrics (15 fields) |
| `get_company_info` | Comprehensive company data |
| `get_eps_trend` | EPS estimates across time periods |
| `get_earnings_dates` | Earnings calendar with surprises |
| `get_earnings_analysis` | Analyst estimates, revisions, growth projections |
| `get_historical_prices` | OHLCV data with momentum indicators |
| `get_ticker_news` | Latest news articles |

### 📋 SEC Filings (1 tool)
| Tool | Description |
|------|-------------|
| `get_sec_filing` | 10-K/10-Q filing text with period comparisons |

### 💬 Social Sentiment (3 tools)
| Tool | Description |
|------|-------------|
| `get_twitter_posts_by_engagement` | Viral Twitter/X posts sorted by engagement |
| `get_reddit_discussions_by_impact` | Reddit posts sorted by impact score |
| `get_social_sentiment` | Combined Twitter + Reddit analysis |

### 🔍 Search & Screening (6 tools)
| Tool | Description |
|------|-------------|
| `search_news_by_ticker` | Keyword-filtered news for ticker |
| `search_news_by_query` | General news search |
| `search_companies` | Advanced company filtering |
| `get_top_value_companies` | Value stock screener |
| `get_top_growth_companies` | Growth stock screener |
| `WebSearchTool` | General web search for context |

## 📊 Structured Output Models

### Production: StockAnalysisOutput (30 fields)
Used by `StructuredAgent` - comprehensive institutional analysis.

**Field Categories:**
- **Basic Info** (4): ticker, company_name, sector, industry
- **EPS & Earnings** (3): estimates, trend, surprise %
- **Valuation** (8): PE, forward PE, PEG, P/B, price, target, distance from highs/lows
- **Analyst Data** (4): count, revisions, sentiment, targets
- **Market Activity** (3): news count, sentiment score, social sentiment
- **Technical** (2): momentum, volatility
- **Investment Summary** (5): thesis, catalysts, risks, recommendation, confidence
- **Analysis** (1): comprehensive narrative

### Tutorial: SimpleStockAnalysis (15 fields)
Simpler version demonstrated in `notebooks/1_tools_and_sample_agents.ipynb`.

**Use StockAnalysisOutput for production, SimpleStockAnalysis for learning.**

## 📁 Project Structure

```
stocks-scoring-agent/
├── stocks_agent/
│   ├── tools.py              # 17 analysis tools
│   ├── simple_agent.py       # Stateless agent (OpenAI)
│   ├── conversation_agent.py # Memory-enabled agent (OpenAI)
│   ├── structured_agent.py   # Structured output agent (OpenAI, 30 fields)
│   ├── free_agent.py         # Local Ollama agent (Qwen, Llama, etc.)
│   ├── monitoring_schema.py  # Portfolio monitor schemas
│   ├── monitor_grader.py     # Deterministic outcome grader
│   ├── portfolio_monitor.py  # Thesis drift monitoring harness
│   ├── monitor_report.py     # Self-contained HTML report artifact
│   └── telegram_digest.py    # Telegram digest helpers
├── docs/
│   └── portfolio-thesis-drift-monitor.md
├── examples/
│   └── portfolio.json        # Sample monitor config
├── tests/
│   └── test_portfolio_monitor.py
├── notebooks/
│   ├── 0_api_endpoints_test_data.ipynb    # API testing
│   ├── 1_tools_and_sample_agents.ipynb    # Tool demos & tutorials
│   └── 2_testing_py_code.ipynb            # Agent testing
├── .envrc.example            # Environment template
├── pyproject.toml            # Dependencies
└── README.md
```

## 🔧 Setup Details

### Dependencies
Managed with `uv`. Key libraries:
- `openai-agents` - AI agent framework
- `yfinance` - Market data
- `edgar` - SEC filing access
- `pydantic` - Structured outputs
- `jupyter` - Notebook support

### Environment Variables
Required:
```bash
export OPENAI_API_KEY='your-openai-key'
export SEC_IDENTITY_EMAIL='your-email@example.com'  # For SEC API
```

Optional:
```bash
export XAI_API_KEY='your-xai-key'
export POLYGON_API_KEY='your-polygon-key'
```

### Jupyter Setup
```bash
source .venv/bin/activate
python -m ipykernel install --user --name=stocks-scoring-agent
# Then select "stocks-scoring-agent" kernel in VS Code
```

## 📚 Learning Path

1. **Start here:** `notebooks/1_tools_and_sample_agents.ipynb` - Learn tools & basic agents
2. **Test agents:** `notebooks/2_testing_py_code.ipynb` - Test production code
3. **Explore API:** `notebooks/0_api_endpoints_test_data.ipynb` - Raw API testing

## ⚠️ Disclaimer

**For research and educational purposes only.** Not financial advice. The author is not responsible for financial losses. Always conduct your own research and consult financial advisors before making investment decisions.

## 📄 License

See LICENSE file.
