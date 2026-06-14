"""Structured stock analysis agent returning Pydantic models with real structured output.

ENHANCED VERSION: This implementation extends the SimpleStockAnalysis model from 
notebook 1_tools_and_sample_agents.ipynb with additional fields for comprehensive
institutional-grade analysis:

- 30 fields vs 15 fields (notebook version)
- Additional metrics: sector, industry, forward_pe, peg_ratio, price_to_book
- Enhanced analysis: target_price, social_sentiment, momentum_direction, volatility
- Investment details: key_catalysts, risk_factors, analysis_summary
- Integrates SEC filings, social sentiment, and comprehensive analyst data

For simpler analysis matching the notebook, use the 15-field SimpleStockAnalysis 
pattern shown in notebook Cell 73-74.
"""

from typing import Optional, Dict, Any, Tuple, Literal
from enum import Enum
from pydantic import BaseModel, Field
from agents import Agent, Runner, ModelSettings
from agents import WebSearchTool
from .tools import AGENT_TOOLS, normalize_ticker


class TrendDirection(str, Enum):
    """Trend direction enumeration."""
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    UNKNOWN = "unknown"


class ValuationLevel(str, Enum):
    """Valuation level enumeration."""
    CHEAP = "cheap"
    FAIR = "fair"
    EXPENSIVE = "expensive"
    UNKNOWN = "unknown"


class Recommendation(str, Enum):
    """Investment recommendation enumeration."""
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"


class StockAnalysisOutput(BaseModel):
    """Structured output schema for comprehensive stock analysis."""
    # Basic Info
    ticker: str = Field(description="Stock ticker symbol")
    company_name: str = Field(description="Company name")
    sector: Optional[str] = Field(description="Business sector")
    industry: Optional[str] = Field(description="Industry classification")
    
    # 1. EPS & Earnings Analysis
    eps_current_estimate: Optional[float] = Field(description="Current quarter EPS estimate")
    eps_trend_direction: TrendDirection = Field(description="EPS trend direction over time")
    earnings_surprise_pct: Optional[float] = Field(description="Latest earnings surprise %")
    
    # 2. Price & Valuation Metrics
    current_price: Optional[float] = Field(description="Current stock price")
    pe_ratio: Optional[float] = Field(description="Trailing P/E ratio")
    forward_pe: Optional[float] = Field(description="Forward P/E ratio")
    peg_ratio: Optional[float] = Field(description="PEG ratio")
    price_to_book: Optional[float] = Field(description="Price-to-book ratio")
    distance_from_52w_high_pct: Optional[float] = Field(description="Distance from 52-week high %")
    distance_from_52w_low_pct: Optional[float] = Field(description="Distance from 52-week low %")
    valuation_level: ValuationLevel = Field(description="Overall valuation assessment")
    
    # 3. Analyst Coverage & Sentiment
    analyst_count: Optional[int] = Field(description="Number of covering analysts")
    analyst_revisions_net_7d: Optional[int] = Field(description="Net analyst revisions last 7 days")
    analyst_sentiment: TrendDirection = Field(description="Overall analyst sentiment trend")
    target_price: Optional[float] = Field(description="Mean analyst target price")
    
    # 4. News & Market Activity
    recent_news_count: Optional[int] = Field(description="Number of recent news articles")
    news_sentiment_score: Optional[float] = Field(ge=-1, le=1, description="News sentiment score (-1 to 1)")
    social_sentiment: Optional[str] = Field(description="Social media sentiment summary")
    
    # 5. Technical & Momentum
    momentum_direction: TrendDirection = Field(description="Price momentum direction")
    volatility_level: Optional[str] = Field(description="Volatility assessment")
    
    # 6. Investment Summary
    investment_thesis: str = Field(description="One-sentence investment thesis")
    key_catalysts: Optional[str] = Field(description="Key upcoming catalysts")
    risk_factors: Optional[str] = Field(description="Primary risk factors")
    recommendation: Recommendation = Field(description="Buy/Hold/Sell recommendation")
    confidence_score: int = Field(ge=1, le=10, description="Confidence level 1-10")
    
    # 7. Comprehensive Analysis
    analysis_summary: str = Field(description="Detailed multi-paragraph analysis")


DEFAULT_INSTRUCTIONS = """You are a comprehensive stock analysis expert providing structured analysis using real-time data.

Your role is to analyze stocks thoroughly using all available tools and provide structured Pydantic output with precise field mapping.

ANALYSIS WORKFLOW:
1. **Company Fundamentals**: Use get_company_info for comprehensive metrics
2. **Earnings Analysis**: Use get_earnings_analysis and get_eps_trend for detailed EPS data
3. **SEC Filing Insights**: Use get_sec_filing for latest developments vs prior periods  
4. **Social Sentiment**: Use get_social_sentiment for Twitter/Reddit analysis
5. **Historical Performance**: Use get_historical_prices for momentum and technical analysis
6. **News Coverage**: Use get_ticker_news for recent developments
7. **Competitive Context**: Use search_companies when relevant

STRUCTURED OUTPUT REQUIREMENTS:
- Map ALL available data from tools to the appropriate Pydantic fields
- For eps_trend_direction: Compare current estimates vs 30/60/90 days ago
- For valuation_level: Consider PE ratio, distance from highs, and peer comparisons
- For analyst_sentiment: Analyze revision trends and recommendation changes
- For momentum_direction: Use price action and technical indicators
- For recommendation: Synthesize all factors into clear buy/hold/sell

FIELD MAPPING GUIDELINES:
- ticker: Extract from function calls
- company_name, sector, industry: From get_company_info
- eps_current_estimate: Current quarter estimate from get_eps_trend
- eps_trend_direction: Compare current vs historical estimates
- current_price, pe_ratio, forward_pe: From get_company_info
- distance_from_52w_high_pct: Calculate from historical data
- analyst_count, target_price: From analyst data tools
- recent_news_count: Count from get_ticker_news
- investment_thesis: One clear sentence summarizing the opportunity
- analysis_summary: Comprehensive 3-4 paragraph analysis integrating all data sources

Be thorough, objective, and data-driven. Fill every relevant field with precise information from the tools."""


class StructuredAgent:
    """
    Stock analysis agent that returns both narrative analysis and structured data.

    Perfect for comprehensive analysis, building dashboards, or feeding data
    to other systems.

    Returns tuple of (text_analysis, structured_data_dict).

    Attributes:
        model: LLM model to use
        agent: The underlying Agent instance
        runner: Runner instance for executing agent

    Example:
        >>> agent = StructuredAgent()
        >>> text, data = await agent.analyze('TSLA')
        >>> print(text)  # Full narrative analysis
        >>> print(f"PE: {data['pe_ratio']}")
        >>> print(f"Trend: {data['eps_trend_direction']}")
        >>> print(f"Valuation: {data['valuation_summary']}")
        >>>
        >>> # With competitors
        >>> text, data = await agent.analyze('AAPL', include_competitors=True)
        >>> print(data['competitor_tickers'])
        >>>
        >>> # Custom model
        >>> agent = StructuredAgent(model='gpt-4o')
    """

    def __init__(
        self,
        model: str = "gpt-5.4-mini",
        temperature: float = 0.3,
        instructions: Optional[str] = None
    ):
        """
        Initialize StructuredAgent.

        Args:
            model: LLM model name (default: gpt-5.4-mini)
            temperature: Model temperature (default: 0.3)
            instructions: Custom instructions (default: DEFAULT_INSTRUCTIONS)
        """
        self.model = model
        self.temperature = temperature

        self.agent = Agent(
            name="structured_stock_agent",
            tools=AGENT_TOOLS + [WebSearchTool()],
            model=model,
            instructions=instructions or DEFAULT_INSTRUCTIONS,
            model_settings=ModelSettings(temperature=temperature),
            output_type=StockAnalysisOutput
        )

        self.runner = Runner()

    async def analyze(
        self,
        ticker: str,
        include_competitors: bool = False,
        show_tools: bool = True,
        show_model: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Perform comprehensive analysis returning text and structured data.

        Args:
            ticker: Stock ticker symbol to analyze
            include_competitors: Whether to include competitor analysis (default: False)
            show_tools: Whether to print tools called (default: True)
            show_model: Whether to print model used (default: True)

        Returns:
            Tuple of (text_analysis, data_dict) where data_dict contains:
                - ticker: str
                - company_name: str
                - sector: str
                - industry: str
                - pe_ratio: float or None
                - forward_pe: float or None
                - current_price: float or None
                - target_price: float or None
                - eps_trend_direction: 'improving' | 'declining' | 'stable' | 'unknown'
                - analyst_sentiment: str (recommendation key)
                - analyst_count: int or None
                - distance_from_high_pct: float (negative value)
                - distance_from_low_pct: float (positive value)
                - valuation_summary: 'cheap' | 'fair' | 'expensive' | 'unknown'
                - momentum: 'positive' | 'negative' | 'mixed'
                - news_count: int
                - competitor_tickers: List[str] (if include_competitors=True)

        Example:
            >>> text, data = await agent.analyze('NVDA')
            >>> print(f"\\n{text}\\n")
            >>> print(f"PE Ratio: {data['pe_ratio']}")
            >>> print(f"EPS Trend: {data['eps_trend_direction']}")
            >>> print(f"Valuation: {data['valuation_summary']}")
            >>> print(f"Distance from high: {data['distance_from_high_pct']:.1f}%")
        """
        ticker = normalize_ticker(ticker)

        # Build comprehensive analysis prompt
        prompt = self._build_analysis_prompt(ticker, include_competitors)

        # Run the agent
        results = await self.runner.run(
            self.agent,
            input=prompt
        )

        # Extract structured output (Pydantic model)
        structured_data = results.final_output

        # Build text response from the analysis_summary
        response = structured_data.analysis_summary

        # Append tools called
        if show_tools:
            tools_called = self._get_tools_called(results)
            if tools_called:
                response += f"\n\n🔧 Tools called: {len(tools_called)}\n"
                for i, tool in enumerate(tools_called, 1):
                    response += f"   {i}. {tool}\n"

        # Prepend model info
        if show_model:
            response = f"🤖 Model: {self.model}\n\n{response}"

        # Convert Pydantic model to dict
        data_dict = structured_data.model_dump()

        return response, data_dict

    async def compare(
        self,
        tickers: list[str],
        show_tools: bool = True,
        show_model: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Compare multiple tickers side-by-side.

        Args:
            tickers: List of ticker symbols to compare
            show_tools: Whether to print tools called (default: True)
            show_model: Whether to print model used (default: True)

        Returns:
            Tuple of (text_comparison, data_dict) with comparative analysis

        Example:
            >>> text, data = await agent.compare(['TSLA', 'NIO', 'RIVN'])
            >>> print(text)
            >>> for ticker, metrics in data['companies'].items():
            ...     print(f"{ticker}: PE={metrics['pe_ratio']}")
        """
        tickers = [normalize_ticker(t) for t in tickers]

        prompt = f"""Perform a comprehensive comparative analysis of these companies:
{', '.join(tickers)}

For each company, analyze:
1. Valuation metrics (PE, PB, etc.)
2. EPS trends
3. Analyst sentiment
4. Price momentum
5. Recent news and catalysts

Then provide a side-by-side comparison highlighting:
- Which is most/least expensive on valuation
- Which has strongest/weakest earnings trends
- Which has most positive/negative analyst sentiment
- Key differentiators and competitive positions

Be objective and data-driven."""

        # Run the agent
        results = await self.runner.run(
            self.agent,
            input=prompt
        )

        # Extract structured output
        structured_data = results.final_output

        # Build response text from analysis_summary
        response = structured_data.analysis_summary

        # Append tools called
        if show_tools:
            tools_called = self._get_tools_called(results)
            if tools_called:
                response += f"\n\n🔧 Tools called: {len(tools_called)}\n"
                for i, tool in enumerate(tools_called, 1):
                    response += f"   {i}. {tool}\n"

        # Prepend model info
        if show_model:
            response = f"🤖 Model: {self.model}\n\n{response}"

        # Return simple comparison data (structured output is for single ticker)
        comparison_data = {
            'tickers': tickers,
            'primary_ticker_data': structured_data.model_dump()
        }

        return response, comparison_data

    def _build_analysis_prompt(self, ticker: str, include_competitors: bool) -> str:
        """Build comprehensive analysis prompt."""
        prompt = f"""Analyze {ticker} and populate ALL structured output fields.

CRITICAL: You MUST fill in every field in the structured output with data from the tools!

Step 1: Call these tools and extract data:
- get_company_info({ticker}) → Extract: company_name, sector, industry, pe_ratio, forward_pe, peg_ratio, price_to_book, current_price, target_price, analyst_sentiment, analyst_count, recommendation_mean
- get_eps_trend({ticker}) → Compare current vs 90 days ago to set eps_trend_direction (improving/declining/stable)
- get_historical_prices({ticker}, period="3mo") → Extract: distance_from_high_pct, distance_from_low_pct, momentum
- get_ticker_news({ticker}, limit=5) → Count articles for news_count

Step 2: Populate structured fields:
- ticker: "{ticker}"
- company_name: From company info
- sector: From company info
- industry: From company info
- pe_ratio: Trailing PE as float
- forward_pe: Forward PE as float
- peg_ratio: PEG ratio as float (or null if N/A)
- price_to_book: P/B ratio as float
- current_price: Current price as float
- target_price: Mean target price as float
- eps_trend_direction: "improving" if current > 90d ago, "declining" if worse, "stable" if similar
- analyst_sentiment: Recommendation key (buy/hold/sell)
- analyst_count: Number of analysts
- recommendation_mean: Mean recommendation number
- distance_from_high_pct: Percentage from 52w high (negative number)
- distance_from_low_pct: Percentage from 52w low (positive number)
- valuation_summary: "expensive" if distance_from_high_pct > -10%, "cheap" if < -30%, else "fair"
- momentum: From historical prices ("positive"/"negative"/"mixed")
- news_count: Number of news articles

Step 3: Write analysis_summary:
Write a comprehensive multi-paragraph analysis covering:
- Company overview and valuation metrics
- EPS trend and whether improving/declining
- Analyst sentiment and price targets
- Price momentum and distance from highs/lows
- Recent news themes
- Overall outlook

Be data-driven and objective."""

        if include_competitors:
            prompt += """

Step 4: Competitive Analysis (in analysis_summary):
- Identify 2-3 main competitors
- Compare valuation and growth
- Note competitive advantages
"""

        return prompt

    def _get_tools_called(self, results):
        """Extract list of tools that were called during execution."""
        from agents.items import ToolCallItem

        tools_called = []

        # Extract tool calls from new_items
        for item in results.new_items:
            # Check for ToolCallItem (new agents library format)
            if isinstance(item, ToolCallItem) and hasattr(item, 'raw_item'):
                func_name = item.raw_item.name
                args = item.raw_item.arguments
                tools_called.append(f"{func_name}({args})")

        return tools_called
