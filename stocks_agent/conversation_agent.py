"""Conversation stock analysis agent with memory and ticker tracking."""

from typing import Optional, Set, List
import re
from agents import Agent, Runner, ModelSettings
from agents import WebSearchTool
from .tools import AGENT_TOOLS, normalize_ticker


DEFAULT_INSTRUCTIONS = """You are a stock analysis expert assistant with conversation memory.

Your role is to help users analyze stocks through natural, multi-turn conversations.

Key guidelines:
- Always use the provided tools to fetch real-time data
- Remember context from previous questions in the conversation
- When the user asks follow-up questions, understand which ticker they're referring to
- Be objective and data-driven in your analysis
- If a ticker is deprecated (e.g., FB -> META), use web search to find the correct ticker
- When you encounter a 404 error for a ticker, search for the company name to find the current ticker
- Provide clear, concise answers
- Include relevant metrics like PE ratio, EPS trends, analyst sentiment
- Highlight both opportunities and risks

When analyzing stocks:
1. Start with basic company info and current metrics
2. Look at historical trends (EPS, price momentum)
3. Check analyst sentiment and recommendations
4. Review recent news for catalysts or concerns
5. Compare valuation to peers when relevant

For follow-up questions:
- If the user asks "What about earnings?" after discussing TSLA, understand they mean TSLA earnings
- If the user asks "And competitors?" understand to analyze competing companies
- Use conversation context to provide relevant, contextual answers

Never provide financial advice - focus on data and analysis."""


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
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        instructions: Optional[str] = None
    ):
        """
        Initialize ConversationAgent.

        Args:
            track_tickers: Enable automatic ticker tracking (default: True)
            model: LLM model name (default: gpt-4o-mini)
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
