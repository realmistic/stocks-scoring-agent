"""Stock Analysis Agents."""

from .simple_agent import SimpleAgent
from .conversation_agent import ConversationAgent
from .structured_agent import StructuredAgent
from .free_agent import FreeAgent
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
    get_top_growth_companies,
    # NEW: Enhanced social sentiment tools
    get_twitter_posts_by_engagement,
    get_reddit_discussions_by_impact,
    get_social_sentiment,
    # NEW: SEC filing analysis
    get_sec_filing
)

__all__ = [
    'SimpleAgent',
    'ConversationAgent',
    'StructuredAgent',
    'FreeAgent',
    'AlertSeverity',
    'MonitorMode',
    'MonitorRun',
    'PortfolioAlert',
    'PortfolioConfig',
    'PortfolioPosition',
    'ToolEvidence',
    'ToolPolicy',
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
    'get_twitter_posts_by_engagement',
    'get_reddit_discussions_by_impact',
    'get_social_sentiment',
    'get_sec_filing'
]
