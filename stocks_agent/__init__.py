"""Stock Analysis Agents."""

from .simple_agent import SimpleAgent
from .conversation_agent import ConversationAgent
from .structured_agent import StructuredAgent

# Also expose for direct use
from .tools import (
    get_company_info_basic,
    get_company_info,
    get_eps_trend,
    get_earnings_dates,
    get_earnings_analysis,
    get_historical_prices,
    get_ticker_news,
    search_news_by_ticker,
    search_news_by_query,
    search_companies,
    get_top_value_companies,
    get_top_growth_companies
)

__all__ = [
    'SimpleAgent',
    'ConversationAgent',
    'StructuredAgent',
    # Tools
    'get_company_info_basic',
    'get_company_info',
    'get_eps_trend',
    'get_earnings_dates',
    'get_earnings_analysis',
    'get_historical_prices',
    'get_ticker_news',
    'search_news_by_ticker',
    'search_news_by_query',
    'search_companies',
    'get_top_value_companies',
    'get_top_growth_companies',
]
