"""Conversation stock analysis agent with memory and ticker tracking."""

from typing import Optional, Set, List
import re
from agents import Agent, Runner, ModelSettings
from agents import WebSearchTool
from .tools import AGENT_TOOLS, normalize_ticker


DEFAULT_INSTRUCTIONS = """You are a conversational stock analysis expert assistant with advanced memory and contextual understanding.

Your role is to help users analyze stocks through natural, multi-turn conversations, building on previous context and maintaining investment focus across discussions.

CORE CAPABILITIES:
- SEC filing analysis (10-K, 10-Q reports) with period comparisons
- Social media sentiment analysis (high-engagement Twitter/X, Reddit discussions)
- Real-time financial metrics and comprehensive analyst data
- Earnings estimates, revisions, and historical performance tracking
- Historical price analysis and momentum indicators
- Company peer comparisons and screening tools

CONVERSATION MEMORY FEATURES:
- Remember tickers discussed in previous messages
- Understand follow-up questions without re-specifying tickers
- Build on previous analysis with new data points
- Track user's investment interests and focus areas

KEY GUIDELINES:
- Always use the provided tools to fetch real-time data before responding
- Remember context from previous questions in the conversation
- When the user asks follow-up questions, understand which ticker they're referring to
- Be objective and data-driven in your analysis with specific evidence
- If a ticker is deprecated (e.g., FB -> META), use web search to find the correct ticker
- When you encounter a 404 error for a ticker, search for the company name to find the current ticker
- Provide comprehensive answers that build on previous conversation context
- Include relevant metrics like PE ratio, EPS trends, analyst sentiment with supporting data
- Highlight both opportunities and risks with specific evidence

ENHANCED ANALYSIS WORKFLOW:
1. **Company Fundamentals**: Use get_company_info for comprehensive metrics
2. **Earnings Deep Dive**: Use get_earnings_analysis for analyst estimates and revisions
3. **Social Sentiment**: Use get_social_sentiment for Twitter/Reddit buzz analysis
4. **SEC Filing Analysis**: Use get_sec_filing for latest developments vs. prior periods
5. **Price & Technical**: Use get_historical_prices for momentum and technical analysis
6. **News & Catalysts**: Use get_ticker_news and search functions for recent developments
7. **Peer Context**: Use search_companies for competitive landscape
8. **Web Search**: Use for breaking news and additional context

CONTEXTUAL FOLLOW-UP EXAMPLES:
- If the user asks "What about earnings?" after discussing TSLA → analyze TSLA earnings data
- If the user asks "And the social sentiment?" → get social sentiment for the current ticker
- If the user asks "Compare to competitors" → find and analyze peer companies
- If the user asks "Any SEC filing updates?" → get latest SEC filings for current ticker

ENHANCED FEATURES:
- Analyze SEC filings for new vs. prior period developments
- Track social media sentiment shifts and viral discussions
- Identify value/growth opportunities using advanced screeners
- Compare companies across multiple fundamental metrics
- Provide engagement-sorted social discussions for market sentiment

Always provide evidence-based analysis with specific data points and maintain conversation continuity. Never provide financial advice - focus on objective data and comprehensive analysis."""


class ConversationAgent:
    """
    Stock analysis agent with conversation memory and ticker tracking.

    Remembers conversation context and can track which tickers are being discussed.
    Perfect for multi-turn conversations and deep-dive analysis.

    Attributes:
        model: LLM model to use
        track_tickers: Whether to track and extract ticker symbols
        history: List of previous RunResult objects
        tickers: Set of tickers mentioned in conversation
        current_ticker: The primary ticker being discussed

    Example:
        >>> agent = ConversationAgent(track_tickers=True)
        >>> await agent.ask("What's TSLA's valuation?")
        >>> await agent.ask("What about earnings?")  # Auto understands TSLA
        >>> await agent.ask("And the news?")  # Still TSLA
        >>>
        >>> # Switch to different ticker
        >>> agent.switch_to("AAPL")
        >>> await agent.ask("What about earnings?")  # Now AAPL
        >>>
        >>> # Start fresh conversation
        >>> agent.reset()
    """

    def __init__(
        self,
        track_tickers: bool = True,
        model: str = "gpt-5.4-mini",
        temperature: float = 0.3,
        instructions: Optional[str] = None
    ):
        """
        Initialize ConversationAgent.

        Args:
            track_tickers: Enable automatic ticker tracking (default: True)
            model: LLM model name (default: gpt-5.4-mini)
            temperature: Model temperature (default: 0.3)
            instructions: Custom instructions (default: DEFAULT_INSTRUCTIONS)
        """
        self.model = model
        self.temperature = temperature
        self.track_tickers = track_tickers

        # Conversation state
        self.history: List = []
        self.tickers: Set[str] = set()
        self.current_ticker: Optional[str] = None

        self.agent = Agent(
            name="conversation_stock_agent",
            tools=AGENT_TOOLS + [WebSearchTool()],
            model=model,
            instructions=instructions or DEFAULT_INSTRUCTIONS,
            model_settings=ModelSettings(temperature=temperature)
        )

        self.runner = Runner()

    async def ask(
        self,
        question: str,
        auto_context: bool = True,
        show_tools: bool = True,
        show_model: bool = False
    ) -> str:
        """
        Ask a stock analysis question with conversation context.

        Args:
            question: The question to ask
            auto_context: Automatically inject ticker context for follow-ups (default: True)
            show_tools: Whether to include tools called in response (default: True)
            show_model: Whether to include model name in response (default: False)

        Returns:
            The agent's response text with tools called appended

        Example:
            >>> response = await agent.ask("What's TSLA's PE ratio?")
            >>> # Next question automatically knows we're talking about TSLA
            >>> response = await agent.ask("What about the earnings trend?")
        """
        # Extract tickers from question
        if self.track_tickers:
            self._extract_tickers(question)

        # Build context from history
        context = self.history[-1].new_items if self.history else None

        # Auto-inject ticker context for follow-up questions
        auto_context_msg = None
        if auto_context and self.current_ticker and not self._has_ticker(question):
            # Check if this looks like a follow-up question
            follow_up_indicators = [
                'what about', 'and the', 'also', 'how about',
                'show me', 'tell me', 'what are', 'what is'
            ]
            if any(indicator in question.lower() for indicator in follow_up_indicators):
                question = f"For {self.current_ticker}: {question}"
                auto_context_msg = f"💡 Auto-context: Analyzing {self.current_ticker}"

        # Run the agent with context
        results = await self.runner.run(
            self.agent,
            input=question,
            context=context
        )

        # Save to history
        self.history.append(results)

        # Build response
        response = results.final_output

        # Prepend auto-context message
        if auto_context_msg:
            response = f"{auto_context_msg}\n\n{response}"

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

        return response

    def switch_to(self, ticker: str):
        """
        Switch the primary ticker for follow-up questions.

        Args:
            ticker: Ticker symbol to switch to

        Example:
            >>> agent.switch_to("AAPL")
            >>> await agent.ask("What about earnings?")  # Now asks about AAPL
        """
        ticker = normalize_ticker(ticker)
        self.current_ticker = ticker
        self.tickers.add(ticker)
        print(f"🎯 Switched to {ticker}")

    def reset(self):
        """
        Clear conversation history and tracked tickers.

        Example:
            >>> agent.reset()
            >>> # Start fresh conversation
        """
        self.history = []
        self.tickers = set()
        self.current_ticker = None
        print("🔄 Conversation reset")

    def get_tickers(self) -> Set[str]:
        """
        Get all tickers mentioned in conversation.

        Returns:
            Set of ticker symbols

        Example:
            >>> tickers = agent.get_tickers()
            >>> print(f"Discussed: {', '.join(tickers)}")
        """
        return self.tickers.copy()

    def _extract_tickers(self, text: str):
        """Extract ticker symbols from text and update state."""
        # Pattern: 1-5 uppercase letters, often standalone or after $
        pattern = r'\b(?:\$)?([A-Z]{1,5})\b'
        matches = re.findall(pattern, text)

        # Common words to exclude
        exclude = {
            'I', 'A', 'IS', 'THE', 'FOR', 'AND', 'OR', 'BUT',
            'IN', 'ON', 'AT', 'TO', 'OF', 'PE', 'EPS', 'PE', 'VS'
        }

        for match in matches:
            if match not in exclude:
                ticker = normalize_ticker(match)
                self.tickers.add(ticker)
                self.current_ticker = ticker

    def _has_ticker(self, text: str) -> bool:
        """Check if text contains a ticker symbol."""
        pattern = r'\b(?:\$)?([A-Z]{1,5})\b'
        matches = re.findall(pattern, text)
        exclude = {
            'I', 'A', 'IS', 'THE', 'FOR', 'AND', 'OR', 'BUT',
            'IN', 'ON', 'AT', 'TO', 'OF', 'PE', 'EPS', 'VS'
        }
        return any(m not in exclude for m in matches)

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
