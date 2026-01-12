# 📈 Stock Analysis Agents

AI-powered stock analysis system with multiple agent types for different use cases. Built with OpenAI Agents SDK and yfinance data.

## Overview

This project provides three types of intelligent agents that analyze stocks using real-time market data:

- **SimpleAgent** - Stateless Q&A for quick stock lookups
- **ConversationAgent** - Memory-enabled multi-turn conversations with automatic ticker tracking
- **StructuredAgent** - Returns both narrative analysis + structured JSON data (uses OpenAI Structured Outputs)

Each agent has access to 11 stock analysis tools covering company info, earnings, prices, news, and market screening.

## Features

✅ Real-time stock data from Yahoo Finance
✅ EPS trend analysis with historical tracking
✅ Analyst sentiment and price targets
✅ News aggregation and filtering
✅ Company screening (growth/value)
✅ Structured outputs for programmatic use
✅ Conversational memory for follow-up questions

## Project Structure

```
stocks_agent/
├── tools.py              # 11 stock analysis tools (get_company_info, get_eps_trend, etc.)
├── simple_agent.py       # Stateless agent for one-off queries
├── conversation_agent.py # Agent with memory and ticker tracking
└── structured_agent.py   # Agent returning structured JSON + text

notebooks/
├── 1_testing_tools.ipynb      # Test individual tools
└── 2_testing_py_code.ipynb    # Test agents
```

## Disclaimer

This project is for research and educational purposes only. It does not constitute financial advice. The author is not responsible for financial losses.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Activate environment
source .venv/bin/activate

# 3. Set up environment variables
cp .envrc.example .envrc
# Edit .envrc with your OPENAI_API_KEY
direnv allow .

# 4. Test in Python or Jupyter
python -c "from stocks_agent import SimpleAgent; print('Ready!')"
```

## 🤖 Agents

### SimpleAgent
Stateless agent for one-off queries. No conversation memory.

```python
from stocks_agent import SimpleAgent

agent = SimpleAgent()
response = await agent.ask("What's AAPL's PE ratio?")
print(response)
```

### ConversationAgent
Maintains conversation context and automatically tracks tickers mentioned.

```python
from stocks_agent import ConversationAgent

agent = ConversationAgent(track_tickers=True)
await agent.ask("What's TSLA's valuation?")
await agent.ask("What about earnings?")  # Auto-knows TSLA
await agent.ask("And the news?")        # Still TSLA

agent.switch_to("AAPL")  # Switch context
agent.reset()            # Clear history
```

### StructuredAgent
Returns tuple of (text_analysis, structured_data) using OpenAI Structured Outputs.

```python
from stocks_agent import StructuredAgent

agent = StructuredAgent()
text, data = await agent.analyze('NVDA')

print(text)  # Narrative analysis
print(data['pe_ratio'])              # 45.57
print(data['eps_trend_direction'])   # "improving"
print(data['valuation_summary'])     # "expensive"
```

## 🛠️ Available Tools

All agents have access to these 11 tools:

| Tool | Description |
|------|-------------|
| `get_company_info_basic` | Basic company info (15 key fields) |
| `get_company_info` | Comprehensive company data |
| `get_eps_trend` | EPS trends over time with historical tracking |
| `get_earnings_dates` | Earnings dates and estimates |
| `get_earnings_analysis` | Analyst earnings estimates and revisions |
| `get_historical_prices` | Price history with momentum indicators |
| `get_ticker_news` | Latest news for a ticker |
| `search_news_by_ticker` | Search news by keyword for a ticker |
| `search_news_by_query` | General news search |
| `search_companies` | Search/filter companies |
| `get_top_value_companies` | Screen for value stocks |
| `get_top_growth_companies` | Screen for growth stocks |

---

## 📦 Setup Details

### Dependencies

This project uses `uv` for dependency management. Key dependencies:
- `openai-agents` - AI agent framework
- `yfinance` - Yahoo Finance data
- `jupyter` - Notebook support
- `pydantic` - Data validation

```bash
uv sync                    # Install all dependencies
source .venv/bin/activate  # Activate virtual environment
```

### Jupyter Notebooks

Register the Jupyter kernel for VS Code:

```bash
source .venv/bin/activate
python -m ipykernel install --user --name=stocks-scoring-agent
```

In VS Code:
1. Open a notebook
2. Select kernel → "Jupyter Kernel" → "stocks-scoring-agent"
3. Launch VS Code from terminal (`code .`) to load environment variables automatically

### Environment Variables

This project uses `direnv` for secure environment variable management.

Create `.envrc` in project root:
```bash
export OPENAI_API_KEY='your-openai-api-key-here'
export POLYGON_API_KEY='your-polygon-api-key-here'
```

Setup direnv (first time only):
```bash
brew install direnv
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
direnv allow .
```

When you `cd` into the project, variables auto-load. When you leave, they auto-unload.
