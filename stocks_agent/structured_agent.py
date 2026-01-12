"""Structured stock analysis agent returning both text and data."""

from typing import Optional, Dict, Any, Tuple, Literal
import json
import re
from pydantic import BaseModel
from agents import Agent, Runner, ModelSettings
from agents import WebSearchTool
from .tools import AGENT_TOOLS, normalize_ticker


class StockAnalysisOutput(BaseModel):
    """Structured output schema for stock analysis."""
    ticker: str
    company_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    current_price: Optional[float] = None
    target_price: Optional[float] = None
    eps_trend_direction: Literal['improving', 'declining', 'stable', 'unknown'] = 'unknown'
    analyst_sentiment: Optional[str] = None
    analyst_count: Optional[int] = None
    recommendation_mean: Optional[float] = None
    distance_from_high_pct: Optional[float] = None
    distance_from_low_pct: Optional[float] = None
    valuation_summary: Literal['cheap', 'fair', 'expensive', 'unknown'] = 'unknown'
    momentum: Literal['positive', 'negative', 'mixed', 'unknown'] = 'unknown'
    news_count: int = 0
    analysis_summary: str


DEFAULT_INSTRUCTIONS = """You are a stock analysis expert providing comprehensive structured analysis.

Your role is to provide deep, data-driven analysis with structured data outputs.

Key guidelines:
- Always use the provided tools to fetch real-time data
- Fill in ALL structured output fields with data from the tools
- Be objective and data-driven
- If a ticker is deprecated (e.g., FB -> META), use web search to find the correct ticker
- When you encounter a 404 error, search for the correct ticker

Analysis framework:
1. **Company Overview**: Basic info, sector, industry
2. **Valuation Analysis**: PE ratio, P/B, PEG, Forward PE
3. **EPS Trend**: Determine if improving/declining/stable based on quarterly data
4. **Analyst Sentiment**: Recommendations, price targets, analyst count
5. **Price Momentum**: Distance from highs/lows, momentum (positive/negative/mixed)
6. **Recent News**: Count articles, identify themes
7. **Valuation Summary**: Classify as cheap/fair/expensive based on:
   - Distance from 52w high: < -30% = cheap, > -10% = expensive, else fair
8. **Analysis Summary**: Write a comprehensive narrative analysis

IMPORTANT:
- Extract numeric data from tool outputs to populate structured fields
- For eps_trend_direction: Compare recent vs older quarters to determine improving/declining/stable
- For valuation_summary: Use distance_from_high_pct to classify
- For momentum: Analyze price trends from historical data
- For analysis_summary: Write a detailed multi-paragraph analysis

Never provide financial advice - focus on data and analysis."""


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
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        instructions: Optional[str] = None
    ):
        """
        Initialize StructuredAgent.

        Args:
            model: LLM model name (default: gpt-4o-mini)
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
