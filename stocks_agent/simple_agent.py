"""Simple stock analysis agent without memory."""

from typing import Optional
from agents import Agent, Runner, ModelSettings
from agents import WebSearchTool
from .tools import AGENT_TOOLS


DEFAULT_INSTRUCTIONS = """You are a stock analysis expert assistant.

Your role is to help users analyze stocks, understand market trends, and make informed decisions.
TRY TO CALL ALL TOOLS before web searching to get the most complete and extended answer. Call web search to check for the latest information -
include the top stats/trends on the stock what you find.

Key guidelines:
- Always use the provided tools to fetch real-time data
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

Never provide financial advice - focus on data and analysis."""


class SimpleAgent:
    """
    Simple stock analysis agent with no conversation memory.

    Each question is handled independently. Good for one-off queries
    and quick lookups.

    Attributes:
        model: LLM model to use (e.g., 'gpt-4o-mini', 'gpt-4o')
        agent: The underlying Agent instance
        runner: Runner instance for executing agent

    Example:
        >>> agent = SimpleAgent()
        >>> response = await agent.ask("What's AAPL's PE ratio?")
        >>> print(response)

        >>> # Use different model
        >>> agent = SimpleAgent(model="gpt-4o")
        >>> response = await agent.ask("Compare TSLA and NIO")
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        instructions: Optional[str] = None
    ):
        """
        Initialize SimpleAgent.

        Args:
            model: LLM model name (default: gpt-4o-mini)
            temperature: Model temperature for response creativity (default: 0.3)
            instructions: Custom instructions (default: DEFAULT_INSTRUCTIONS)
        """
        self.model = model
        self.temperature = temperature

        self.agent = Agent(
            name="simple_stock_agent",
            tools=AGENT_TOOLS + [WebSearchTool()], # type: ignore
            model=model,
            instructions=instructions or DEFAULT_INSTRUCTIONS,
            model_settings=ModelSettings(temperature=temperature)
        )

        self.runner = Runner()

    async def ask(
        self,
        question: str,
        show_tools: bool = True,
        show_model: bool = True
    ) -> str:
        """
        Ask a stock analysis question.

        Args:
            question: The question to ask
            show_tools: Whether to include tools called in response (default: True)
            show_model: Whether to include model name in response (default: True)

        Returns:
            The agent's response text with tools called appended

        Example:
            >>> response = await agent.ask("What's TSLA's current valuation?")
            >>> response = await agent.ask(
            ...     "Is NVDA expensive?",
            ...     show_tools=False
            ... )
        """
        # Run the agent
        results = await self.runner.run(
            self.agent,
            input=question
        )

        # Build response
        response = results.final_output

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
