"""Stock analysis tools - callable directly or via agents."""

import yfinance as yf
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import re
from io import StringIO
import os
import requests
from datetime import timezone
from minsearch import Index
from tqdm import tqdm
from pprint import pprint

# Ticker deprecation mapping
TICKER_MAPPING = {
    'FB': 'META',
    'GOOGL': 'GOOG',  # Sometimes consolidated
}


def normalize_ticker(ticker: str) -> str:
    """Normalize ticker symbol, handling deprecations."""
    ticker = ticker.upper().strip()
    return TICKER_MAPPING.get(ticker, ticker)

def get_company_info_basic(ticker: str) -> Dict[str, Any]:
    """
    Get basic company information including name, sector, industry, and key metrics.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')

    Returns:
        Dictionary with company info including PE ratio, market cap, etc.

    Data transformation:
        RAW: yfinance stock.info dictionary (100+ fields)
        TRANSFORMED: Extracts 15 key fields relevant for analysis

    Example:
        >>> info = get_company_info('AAPL')
        >>> print(f"PE Ratio: {info['pe_ratio']}")
    """
    ticker = normalize_ticker(ticker)
    stock = yf.Ticker(ticker)
    info = stock.info

    return {
        'ticker': ticker,
        'name': info.get('longName', 'N/A'),
        'sector': info.get('sector', 'N/A'),
        'industry': info.get('industry', 'N/A'),
        'market_cap': info.get('marketCap', 'N/A'),
        'pe_ratio': info.get('trailingPE', 'N/A'),
        'forward_pe': info.get('forwardPE', 'N/A'),
        'peg_ratio': info.get('pegRatio', 'N/A'),
        'price_to_book': info.get('priceToBook', 'N/A'),
        'dividend_yield': info.get('dividendYield', 'N/A'),
        'beta': info.get('beta', 'N/A'),
        'current_price': info.get('currentPrice', 'N/A'),
        'target_price': info.get('targetMeanPrice', 'N/A'),
        'recommendation': info.get('recommendationKey', 'N/A'),
        'website': info.get('website', 'N/A'),
    }


def get_company_info(ticker: str) -> dict[str, Any]:
    """
    Get comprehensive company information and fundamental data for a stock ticker.
    
    Returns key metrics organized by category:
    - Company basics: website, industry, sector, employees, officers
    - Price data: current, previous close, day range, 52-week range
    - Market metrics: market cap, volume, beta, PE ratios
    - Valuation: margins, book value, price ratios
    - Ownership: insider/institutional holdings, short interest
    - Analyst data: EPS estimates, targets, recommendations
    - Financial health: cash, returns, growth rates
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')
    
    Returns:
        Dictionary with ticker and organized company information
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.get_info()

        if not info:
            return {"ticker": ticker, "error": "No company info available"}

        # Define the key fields to extract (organized by category)
        key_fields = {
            # Company basics
            "company": ["website", "industry", "sector", "longBusinessSummary",
                        "fullTimeEmployees", "companyOfficers", "region", "fullExchangeName"],

            # Price data
            "price": ["currentPrice", "previousClose", "open", "dayLow", "dayHigh",
                    "regularMarketDayRange", "fiftyTwoWeekLow", "fiftyTwoWeekHigh",
                    "fiftyTwoWeekRange", "allTimeHigh", "allTimeLow"],

            # Market metrics
            "market": ["marketCap", "volume", "averageVolume", "averageVolume10days",
                    "beta", "trailingPE", "forwardPE", "trailingPegRatio"],

            # Moving averages
            "averages": ["fiftyDayAverage", "twoHundredDayAverage",
                        "fiftyDayAverageChange", "twoHundredDayAverageChange"],

            # Valuation ratios
            "valuation": ["priceToSalesTrailing12Months", "priceToBook", "bookValue",
                        "profitMargins", "grossMargins", "ebitdaMargins", "operatingMargins"],

            # Ownership & short interest
            "ownership": ["sharesOutstanding", "floatShares", "sharesPercentSharesOut",
                        "heldPercentInsiders", "heldPercentInstitutions",
                        "sharesShort", "shortRatio", "shortPercentOfFloat"],

            # EPS & earnings
            "earnings": ["trailingEps", "forwardEps", "earningsQuarterlyGrowth",
                        "earningsGrowth", "revenueGrowth", "epsTrailingTwelveMonths",
                        "epsForward", "epsCurrentYear"],

            # Analyst targets & recommendations
            "analyst": ["targetHighPrice", "targetLowPrice", "targetMeanPrice",
                        "targetMedianPrice", "recommendationMean", "recommendationKey",
                        "numberOfAnalystOpinions", "averageAnalystRating"],

            # Financial health
            "financial": ["totalCash", "totalCashPerShare", "totalDebt", "totalRevenue",
                        "freeCashflow", "operatingCashflow", "returnOnAssets",
                        "returnOnEquity", "debtToEquity", "currentRatio", "quickRatio"]
        }

        # Extract data by category
        result = {"ticker": ticker}

        for category, fields in key_fields.items():
            category_data = {}
            for field in fields:
                if field in info:
                    category_data[field] = info[field]
            if category_data:
                result[category] = category_data # type: ignore

        return result

    except Exception as e:
        return {"ticker": ticker, "error": f"Failed to get company info: {str(e)}"}



def get_eps_trend(ticker: str) -> dict[str, Any]:
    """
    Get the EPS (Earnings Per Share) trend for a given stock ticker - showing how analyst consensus has changed over time for different periods (quarterly, yearly)
    and diffent points in the past (current, 7daysAgo, 30daysAgo, etc.).
    Index: 0q (This Quarter),  +1q (Next Quarter),  0y (This Year),  +1y (Next Year) 
    and columns showing estimates from different points in the past (current, 7daysAgo, 30daysAgo, etc.). 

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')
    
    Returns:
        Dictionary with ticker and EPS trend data
    """

    try:
        ticker_obj = yf.Ticker(ticker)
        result = ticker_obj.get_eps_trend()

        if isinstance(result, pd.DataFrame):
            if result.empty:
                return {"ticker": ticker, "error": "No EPS trend data available"}
            result['period']=result.index # create a new column 'period' from the index
            return {"ticker": ticker, "data": result.to_dict(orient='records')}

        # Fallback: wrap unexpected types
        if isinstance(result, dict):
            return {"ticker": ticker, "data": [result]}

        raise TypeError(f"Unexpected return type from get_eps_trend: {type(result)}")

    except Exception as e:
        return {"ticker": ticker, "error": f"Failed to get EPS trend: {str(e)}"}



def get_earnings_dates(ticker: str) -> dict[str, Any]:
    """
    Get earnings call dates for a stock ticker.
    
    Returns historical earnings data including:
    - Expected EPS
    - Actual EPS  
    - Surprise percentage
    - Earnings dates from multiple quarters and years
    - Next earnings call date
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')

    Returns:
        Dictionary with ticker and earnings dates data, surprise (%) - how reported earnings compared to expectations
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        result = ticker_obj.get_earnings_dates()

        if isinstance(result, pd.DataFrame):
            if result.empty:
                return {"ticker": ticker, "error": "No earnings data available"}

            # Normalize common datetime-like columns to yyyy-mm-dd strings
            result.index = pd.to_datetime(result.index, errors="coerce").strftime("%Y-%m-%d")
          
            # Include the index (dates) as a column before converting to dict
            result = result.reset_index().rename(columns={"index": "date"})
            return {"ticker": ticker, "data": result.to_dict(orient='records')}

        # Unexpected type fallback
        raise TypeError(f"Unexpected return type from get_earnings_dates: {type(result)}")

    except Exception as e:
        return {"ticker": ticker, "error": f"Failed to get earnings dates: {str(e)}"}


def get_earnings_analysis(ticker: str) -> Dict[str, Any]:
    """
    Get analyst earnings and EPS analysts estimates and revisions.

    Combines multiple analyst data sources:
    0. Basic Analyst Info - Count, recommendation, target prices
    1. Earnings Estimates - Consensus EPS estimates (avg, low, high, year-ago, analyst count)
    2. EPS Revisions - How analysts have revised estimates (up/down last 7/30 days)
    3. Growth Estimates - Expected earnings growth vs index benchmark
    4. Earnings History - Historical actual vs estimated EPS with surprise %
    
    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with analyst estimates and sentiment

    Data transformation:
        RAW: yfinance stock.info dictionary
        TRANSFORMED: Extracts analyst-related fields (recommendations, targets)
    """

    ticker = normalize_ticker(ticker)
    ticker_obj = yf.Ticker(ticker)
    
    result = {
            "ticker": ticker,
            "basic_analyst_info": None,
            "earnings_estimates": None,
            "eps_revisions": None,
            "growth_estimates": None,
            "earnings_history": None
    }

    # 0. Basic Analyst Info
    # Extract key analyst fields from stock.info

    try:
        info = ticker_obj.info    
        result["basic_analyst_info"] = {
            "analyst_count": info.get('numberOfAnalystOpinions', 'N/A'),
            "recommendation": info.get('recommendationKey', 'N/A'),
            "recommendation_mean": info.get('recommendationMean', 'N/A'),
            "target_high": info.get('targetHighPrice', 'N/A'),
            "target_low": info.get('targetLowPrice', 'N/A'),
            "target_mean": info.get('targetMeanPrice', 'N/A'),
            "target_median": info.get('targetMedianPrice', 'N/A'),
            "current_price": info.get('currentPrice', 'N/A'),
        }
    except Exception as e:
            result["basic_analyst_info"] = {"error": str(e)}

    # 1. Get earnings estimates
    try:
        earnings_est = ticker_obj.get_earnings_estimate()
        if isinstance(earnings_est, pd.DataFrame) and not earnings_est.empty:
            earnings_est = earnings_est.reset_index().rename(columns={"index": "period"})
            result["earnings_estimates"] = earnings_est.to_dict(orient='records')
    except Exception as e:
        result["earnings_estimates"] = {"error": str(e)}

    # 2. Get EPS revisions
    try:
        eps_rev = ticker_obj.get_eps_revisions()
        if isinstance(eps_rev, pd.DataFrame) and not eps_rev.empty:
            eps_rev = eps_rev.reset_index().rename(columns={"index": "period"})
            result["eps_revisions"] = eps_rev.to_dict(orient='records')
    except Exception as e:
        result["eps_revisions"] = {"error": str(e)}

    # 3. Get growth estimates
    try:
        growth_est = ticker_obj.get_growth_estimates()
        if isinstance(growth_est, pd.DataFrame) and not growth_est.empty:
            growth_est = growth_est.reset_index().rename(columns={"index": "period"})
            result["growth_estimates"] = growth_est.to_dict(orient='records')
    except Exception as e:
        result["growth_estimates"] = {"error": str(e)}

    # 4. Get earnings history
    try:
        earnings_hist = ticker_obj.get_earnings_history()
        if isinstance(earnings_hist, pd.DataFrame) and not earnings_hist.empty:
            earnings_hist = earnings_hist.reset_index().rename(columns={"index": "quarter"})
            result["earnings_history"] = earnings_hist.to_dict(orient='records')
    except Exception as e:
        result["earnings_history"] = {"error": str(e)}

    # Check if we got any data at all
    has_data = any(
        result[key] is not None and not isinstance(result[key], dict) or (isinstance(result[key], dict) and "error" not in result[key])
        for key in ["earnings_estimates", "eps_revisions", "growth_estimates", "earnings_history"]
    )

    if not has_data:
        return {"ticker": ticker, "error": "No earnings analysis data available"}

    return result


def get_historical_prices(ticker: str, period: str = '1y', interval: str = '1d') -> Dict[str, Any]:
    """
    Get historical price data - return key statistics and trends.
    This is a simplified version focusing on key metrics. 
    NO FULL TIME SERIES RETURNED.

    Args:
        ticker: Stock ticker symbol
        period: Time period ('1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
        interval: Data interval ('1d', '1wk', '1mo')

    Returns:
        Dictionary with price history and key statistics

    Data transformation:
        RAW: yfinance history DataFrame (OHLCV data for each period)
        CALCULATED:
            - distance_from_high_pct = (current - 52w_high) / 52w_high * 100
            - distance_from_low_pct = (current - 52w_low) / 52w_low * 100
            - avg_volume = mean of all volume data
        DERIVED:
            - momentum = 'positive' if price > MA20 > MA50
                       = 'negative' if price < MA20 < MA50
                       = 'mixed' otherwise
    """
    ticker = normalize_ticker(ticker)
    stock = yf.Ticker(ticker)

    hist = stock.history(period=period, interval=interval)

    if hist.empty:
        return {
            'ticker': ticker,
            'error': 'No price data available'
        }

    # Calculate key metrics
    current_price = hist['Close'].iloc[-1]
    period_high = hist['High'].max()
    period_low = hist['Low'].min()
    avg_volume = hist['Volume'].mean()

    # Price momentum
    if len(hist) >= 20:
        ma_20 = hist['Close'].tail(20).mean()
        ma_50 = hist['Close'].tail(50).mean() if len(hist) >= 50 else ma_20
        momentum = 'positive' if current_price > ma_20 > ma_50 else 'negative' if current_price < ma_20 < ma_50 else 'mixed'
    else:
        momentum = 'insufficient_data'

    return {
        'ticker': ticker,
        'period': period,
        'interval': interval,
        'current_price': float(current_price),
        'period_high': float(period_high),
        'period_low': float(period_low),
        'distance_from_high_pct': float((current_price - period_high) / period_high * 100),
        'distance_from_low_pct': float((current_price - period_low) / period_low * 100),
        'avg_volume': float(avg_volume),
        'momentum': momentum,
        'data_points': len(hist),
    }


def get_ticker_news(ticker: str, limit: int = 50) -> Dict[str, Any]:
    """
    Get recent news for a specific ticker (sourced from yfinance).

    Args:
        ticker: Stock ticker symbol
        limit: Maximum number of news items to return

    Returns:
        Dictionary with news articles

    Data transformation:
        RAW: yfinance stock.news list (nested structure with 'content' wrapper)
        TRANSFORMED: Flattens nested structure, extracts key fields:
                     - title, publisher, link, published date, description, summary
                     - Handles both old and new yfinance API formats
    """
    ticker = normalize_ticker(ticker)
    stock = yf.Ticker(ticker)

    news = stock.news if hasattr(stock, 'news') else []

    news_items = []
    for item in news[:limit]:
        # Handle new yfinance API format (content nested)
        if 'content' in item:
            content = item['content']
            news_items.append({
                'title': content.get('title', 'N/A'),
                'publisher': content.get('provider', {}).get('displayName', 'N/A'),
                'link': content.get('canonicalUrl', {}).get('url', 'N/A'),
                'published': content.get('pubDate', 'N/A'),
                'description': content.get('description', ''),
                'summary': content.get('summary', '')
            })
        else:
            # Handle old format (fallback)
            news_items.append({
                'title': item.get('title', 'N/A'),
                'publisher': item.get('publisher', 'N/A'),
                'link': item.get('link', 'N/A'),
                'published': datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M') if item.get('providerPublishTime') else 'N/A',
                'summary': ''
            })

    return {
        'ticker': ticker,
        'news_count': len(news_items),
        'news': news_items
    }

# ================ Additional Search Tools ===================

# Global variable to store the news index (built once, reused)
_news_index = None
_news_documents = None

def build_polygon_news_index(api_calls: int = 5, news_per_call: int = 1000) -> dict[str, Any]:
    """
    Fetch news from Polygon.io (Massive.com) and build a searchable index.
    
    This should be called once to fetch and index news. The index is stored
    globally and reused by search functions.
    
    Args:
        api_calls: Number of API calls to make (default: 5, fetches ~5000 articles)
        news_per_call: Number of news articles per API call (max: 1000)
    
    Returns:
        Dictionary with status and article count
    """
    global _news_index, _news_documents

    try:
        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key:
            return {"error": "POLYGON_API_KEY not found in environment"}

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        all_news = None
        max_date = now

        print(f"Fetching {api_calls * news_per_call} news articles...")

        for i in tqdm(range(api_calls), desc="API calls"):
            url = f"https://api.massive.com/v2/reference/news?order=desc&limit={news_per_call}&sort=published_utc&published_utc.lt={max_date}&apiKey={api_key}"

            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                data = r.json()

                if 'results' not in data:
                    print(f"No 'results' in response. Keys: {data.keys()}")
                    continue

                cur = pd.json_normalize(data['results'])

                if all_news is None:
                    all_news = cur
                else:
                    all_news = pd.concat([all_news, cur], ignore_index=True)

                max_date = cur.published_utc.min()

            except requests.exceptions.RequestException as e:
                print(f"API call {i+1} failed: {e}")
                continue

        if all_news is None or all_news.empty:
            return {"error": "Failed to fetch news articles"}

        # Convert to documents
        _news_documents = all_news.to_dict(orient='records')

        # Preprocess documents
        print("Preprocessing documents...")
        for doc in tqdm(_news_documents, desc="Converting fields"):
            if isinstance(doc.get('tickers'), list):
                doc['tickers'] = ', '.join(doc['tickers'])
            if isinstance(doc.get('keywords'), list):
                doc['keywords'] = ', '.join(doc['keywords'])

            for field in ['title', 'description', 'author']:
                if doc.get(field) is None:
                    doc[field] = ''
                elif not isinstance(doc.get(field), str):
                    doc[field] = str(doc[field])

        # Build index
        print("Building search index...")
        _news_index = Index(
            text_fields=["title", "description", "keywords", "author", "tickers"],
            keyword_fields=["published_utc", "publisher.name"]
        )
        _news_index.fit(_news_documents)

        return {
            "status": "success",
            "articles_indexed": len(_news_documents),
            "message": f"Index built with {len(_news_documents)} articles"
        }

    except Exception as e:
        return {"error": f"Failed to build news index: {str(e)}"}

def search_news_by_ticker(ticker: str, query: str = "", num_results: int = 30) -> dict[str, Any]:
    """
    Search indexed news articles for a specific stock ticker.
    
    Searches across title, description, keywords, and tickers with boosting
    that prioritizes ticker matches.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'TSLA', 'AAPL', 'GOOGL')
        num_results: Maximum number of results to return (default: 30)
    
    Returns:
        Dictionary with ticker and matching news articles
    """
    global _news_index

    if _news_index is None:
        # build the index automatically if not present
        result = build_polygon_news_index(api_calls=5)
        pprint(result)
        # return {
        #     "ticker": ticker,
        #     "error": "News index not built. Call build_polygon_news_index() first."
        # }

    try:
        results = _news_index.search( # type: ignore
            query=ticker+(" " + query if query else ""),
            num_results=num_results,
            boost_dict={
                'tickers': 5.0,      # Highest boost for ticker field
                'title': 3.0,        # High boost for title
                'description': 1.5,  # Medium boost for description
                'keywords': 1.0      # Standard boost for keywords
            }
        )

        return {
            "ticker": ticker,
            "count": len(results),
            "data": results
        }

    except Exception as e:
        return {"ticker": ticker, "error": f"Search failed: {str(e)}"}


def search_news_by_query(query: str, num_results: int = 30) -> dict[str, Any]:
    """
    Search indexed news articles by free-text query.
    
    Searches across title, description, keywords, and tickers with boosting
    that prioritizes description and keyword matches.
    
    Args:
        query: Search query (e.g., 'Tesla competitors EV market', 'AI robotics')
        num_results: Maximum number of results to return (default: 30)
    
    Returns:
        Dictionary with query and matching news articles
    """
    global _news_index

    if _news_index is None:
        # build the index automatically if not present
        result = build_polygon_news_index(api_calls=5)
        pprint(result)
        # return {
        #     "query": query,
        #     "error": "News index not built. Call build_polygon_news_index() first."
        # }

    try:
        results = _news_index.search( # type: ignore
            query=query,
            num_results=num_results,
            boost_dict={
                'tickers': 1.0,       # Standard boost for ticker field
                'title': 3.0,         # High boost for title
                'description': 5.0,   # Highest boost for description
                'keywords': 5.0       # Highest boost for keywords
            }
        )

        return {
            "query": query,
            "count": len(results),
            "data": results
        }

    except Exception as e:
        return {"query": query, "error": f"Search failed: {str(e)}"}



# Global variables to cache the databases
_companies_marketcap_db = None
_companies_pe_db = None
_companies_dividend_db = None
_companies_margin_db = None
_unified_db = None

def load_all_companies_databases(force_refresh: bool = False) -> dict[str, Any]:
    """
    Download and load all company databases from companiesmarketcap.com.
    
    Loads 4 databases:
    1. Market Cap - Top companies by market capitalization
    2. P/E Ratio - Top companies by price-to-earnings ratio
    3. Dividend Yield - Top companies by dividend yield percentage
    4. Operating Margin - Top companies by operating margin percentage
    
    The databases are cached globally and merged by ticker symbol for unified searching.
    Use force_refresh=True to re-download.
    
    Args:
        force_refresh: If True, re-download all databases even if cached
    
    Returns:
        Dictionary with status and database info
    """
    global _companies_marketcap_db, _companies_pe_db, _companies_dividend_db
    global _companies_margin_db, _unified_db

    # Return cached data if available
    if _unified_db is not None and not force_refresh:
        df = pd.DataFrame(_unified_db)
        return {
            "status": "loaded_from_cache",
            "total_companies": len(_unified_db),
            "available_columns": list(df.columns),
            "message": f"All databases loaded from cache"
        }

    try:
        databases = {
            "marketcap": "https://companiesmarketcap.com/usd/?download=csv",
            "pe_ratio": "https://companiesmarketcap.com/top-companies-by-pe-ratio/?download=csv",
            "dividend": "https://companiesmarketcap.com/top-companies-by-dividend-yield/?download=csv",
            "margin": "https://companiesmarketcap.com/top-companies-by-operating-margin/?download=csv"
        }

        loaded = {}

        for name, url in databases.items():
            print(f"Downloading {name} database...")
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                csv_data = StringIO(response.text)
                df = pd.read_csv(csv_data)
                loaded[name] = df
                print(f"✓ Loaded {len(df)} companies from {name}")
            except Exception as e:
                print(f"✗ Failed to load {name}: {e}")
                loaded[name] = None

        # Store individual databases
        _companies_marketcap_db = loaded["marketcap"]
        _companies_pe_db = loaded["pe_ratio"]
        _companies_dividend_db = loaded["dividend"]
        _companies_margin_db = loaded["margin"]

        # Merge all databases by Symbol for unified view
        print("\nMerging databases...")

        # Start with market cap as base
        unified = loaded["marketcap"].copy()

        # Merge P/E ratio - only keep pe_ratio_ttm column
        if loaded["pe_ratio"] is not None and 'pe_ratio_ttm' in loaded["pe_ratio"].columns:
            pe_df = loaded["pe_ratio"][['Symbol', 'pe_ratio_ttm']]
            unified = unified.merge(pe_df, on='Symbol', how='left')
            print(f"✓ Added P/E ratio column")

        # Merge Dividend yield - only keep dividend_yield_ttm column
        if loaded["dividend"] is not None and 'dividend_yield_ttm' in loaded["dividend"].columns:
            div_df = loaded["dividend"][['Symbol', 'dividend_yield_ttm']]
            # Convert percentage to decimal
            div_df['dividend_yield_ttm'] = div_df['dividend_yield_ttm'] / 100.0 
            unified = unified.merge(div_df, on='Symbol', how='left')
            print(f"✓ Added Dividend yield column")

        # Merge Operating margin - only keep operating_margin_ttm column
        if loaded["margin"] is not None and 'operating_margin_ttm' in loaded["margin"].columns:
            margin_df = loaded["margin"][['Symbol', 'operating_margin_ttm']]
            # Convert percentage to decimal
            margin_df['operating_margin_ttm'] = margin_df['operating_margin_ttm']/100.0
            unified = unified.merge(margin_df, on='Symbol', how='left')
            print(f"✓ Added Operating margin column")

        print(f"\nFinal columns: {list(unified.columns)}")

        _unified_db = unified.to_dict(orient='records')

        return {
            "status": "success",
            "databases_loaded": {
                "marketcap": len(loaded["marketcap"]) if loaded["marketcap"] is not None else 0,
                "pe_ratio": len(loaded["pe_ratio"]) if loaded["pe_ratio"] is not None else 0,
                "dividend": len(loaded["dividend"]) if loaded["dividend"] is not None else 0,
                "margin": len(loaded["margin"]) if loaded["margin"] is not None else 0
            },
            "total_companies": len(_unified_db),
            "available_columns": list(unified.columns),
            "message": f"All databases merged with {len(_unified_db)} unique companies"
        }

    except Exception as e:
        return {"error": f"Failed to load databases: {str(e)}"}


def get_available_columns() -> dict[str, Any]:
    """
    Get list of available columns in the unified database.
    
    Returns:
        Dictionary with available column names
    """
    global _unified_db

    if _unified_db is None:
        return {"error": "Database not loaded. Call load_all_companies_databases() first."}

    df = pd.DataFrame(_unified_db)
    return {
        "columns": list(df.columns),
        "total_columns": len(df.columns)
    }


def search_companies(
    query: Optional[str] = None,
    ticker: Optional[str] = None,
    min_market_cap: Optional[float] = None,
    max_market_cap: Optional[float] = None,
    min_pe: Optional[float] = None,
    max_pe: Optional[float] = None,
    min_dividend: Optional[float] = None,
    max_dividend: Optional[float] = None,
    min_margin: Optional[float] = None,
    max_margin: Optional[float] = None,
    country: Optional[str] = None,
    limit: int = 50
) -> dict[str, Any]:
    """
    Search companies across all databases with comprehensive filtering.
    
    Args:
        query: Search by company name (case-insensitive partial match)
        ticker: Search by exact ticker symbol
        min_market_cap: Minimum market cap in USD
        max_market_cap: Maximum market cap in USD
        min_pe: Minimum P/E ratio
        max_pe: Maximum P/E ratio
        min_dividend: Minimum dividend yield (%)
        max_dividend: Maximum dividend yield (%)
        min_margin: Minimum operating margin (%)
        max_margin: Maximum operating margin (%)
        country: Filter by country (e.g., 'USA', 'China')
        limit: Maximum number of results (default: 50)
    
    Returns:
        Dictionary with matching companies and all available metrics
    """
    global _unified_db

    if _unified_db is None:
        # Load databases
        result = load_all_companies_databases(force_refresh=True)
        pprint(result)
        # return {"error": "Database not loaded. Call load_all_companies_databases() first."}

    try:
        df = pd.DataFrame(_unified_db)

        # Apply filters
        if ticker:
            df = df[df['Symbol'].str.upper() == ticker.upper()]

        if query:
            df = df[df['Name'].str.contains(query, case=False, na=False)]

        if min_market_cap is not None:
            df = df[df['marketcap'] >= min_market_cap]

        if max_market_cap is not None:
            df = df[df['marketcap'] <= max_market_cap]

        if 'pe_ratio_ttm' in df.columns:
            if min_pe is not None:
                df = df[pd.to_numeric(df['pe_ratio_ttm'], errors='coerce') >= min_pe]
            if max_pe is not None:
                df = df[pd.to_numeric(df['pe_ratio_ttm'], errors='coerce') <= max_pe]

        if 'dividend_yield_ttm' in df.columns:
            if min_dividend is not None:
                df = df[pd.to_numeric(df['dividend_yield_ttm'], errors='coerce') >= min_dividend]
            if max_dividend is not None:
                df = df[pd.to_numeric(df['dividend_yield_ttm'], errors='coerce') <= max_dividend]

        if 'operating_margin_ttm' in df.columns:
            if min_margin is not None:
                df = df[pd.to_numeric(df['operating_margin_ttm'], errors='coerce') >= min_margin]
            if max_margin is not None:
                df = df[pd.to_numeric(df['operating_margin_ttm'], errors='coerce') <= max_margin]

        if country:
            df = df[df['country'].str.upper() == country.upper()]

        # Limit results
        df = df.head(limit)

        if df.empty:
            return {
                "count": 0,
                "message": "No companies found matching criteria"
            }

        return {
            "count": len(df),
            "data": df.to_dict(orient='records')
        }

    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


def get_top_value_companies(
    min_dividend: float = 2.0/100.0, # 2%
    max_pe: float = 25,
    min_margin: float = 10/100.0, # 10%
    min_market_cap: float = 1_000_000_000,
    limit: int = 50
) -> dict[str, Any]:
    """
    Find potential value companies based on fundamental criteria.
    
    Default criteria:
    - Dividend yield >= 2%
    - P/E ratio <= 25
    - Operating margin >= 10%
    - Market cap >= $1B
    
    Args:
        min_dividend: Minimum dividend yield %
        max_pe: Maximum P/E ratio
        min_margin: Minimum operating margin %
        min_market_cap: Minimum market cap
        limit: Maximum results
    
    Returns:
        Dictionary with companies meeting value criteria
    """
    return search_companies(
        min_dividend=min_dividend,
        max_pe=max_pe,
        min_margin=min_margin,
        min_market_cap=min_market_cap,
        limit=limit
    )


def get_top_growth_companies(
    min_margin: float = 20/100.0, # 20%
    max_pe: Optional[float] = None,
    min_market_cap: float = 1_000_000_000,
    limit: int = 50
) -> dict[str, Any]:
    """
    Find potential growth companies based on fundamental criteria.
    
    Default criteria:
    - Operating margin >= 20% (high profitability)
    - Market cap >= $1B
    
    Args:
        min_margin: Minimum operating margin %
        max_pe: Maximum P/E ratio (optional)
        min_market_cap: Minimum market cap
        limit: Maximum results
    
    Returns:
        Dictionary with companies meeting growth criteria
    """
    return search_companies(
        min_margin=min_margin,
        max_pe=max_pe,
        min_market_cap=min_market_cap,
        limit=limit
    )



# ================= Agent Tool Wrappers ===================
# Import function_tool decorator for agent use
try:
    from agents import function_tool

    # Create decorated versions for agents
    get_company_info_tool = function_tool(get_company_info)
    get_company_info_basic_tool = function_tool(get_company_info_basic)
    get_eps_trend_tool = function_tool(get_eps_trend)
    get_earnings_dates_tool = function_tool(get_earnings_dates)
    get_earnings_analysis_tool = function_tool(get_earnings_analysis)
    get_historical_prices_tool = function_tool(get_historical_prices)
    get_ticker_news_tool = function_tool(get_ticker_news)
    search_news_by_ticker_tool = function_tool(search_news_by_ticker)
    search_news_by_query_tool = function_tool(search_news_by_query)
    search_companies_tool = function_tool(search_companies)
    get_top_value_companies_tool = function_tool(get_top_value_companies)
    get_top_growth_companies_tool = function_tool(get_top_growth_companies)

    # List of all tools for agent use
    AGENT_TOOLS = [
        get_company_info_tool,
        get_eps_trend_tool,
        get_earnings_dates_tool,
        get_earnings_analysis_tool,
        get_historical_prices_tool,
        get_ticker_news_tool,
        search_news_by_ticker_tool,
        search_news_by_query_tool,
        search_companies_tool,
        get_top_value_companies_tool,
        get_top_growth_companies_tool,
    ]

except ImportError:
    # If openai_agents not available, tools can still be used directly
    AGENT_TOOLS = []
