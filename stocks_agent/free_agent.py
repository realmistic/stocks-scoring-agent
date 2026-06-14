"""Free stock analysis agent using local Ollama models (Qwen, Llama, etc.)

This agent provides the same functionality as SimpleAgent but uses free local models
via Ollama instead of paid OpenAI API. Perfect for cost-conscious analysis or 
when API limits are a concern.

Supported models:
- qwen3:32b (recommended for tool calling)
- llama3.1:70b 
- codellama:34b
- mistral:7b
- Any Ollama model with tool calling support
"""

from typing import Optional, List, Dict, Any
import json
from ollama import chat, ChatResponse
from .tools import AGENT_TOOLS


DEFAULT_INSTRUCTIONS = """You are an expert stock analyst with access to real-time financial data tools.

Your role is to provide comprehensive, accurate, and actionable stock analysis using all available tools.

ANALYSIS APPROACH:
1. **Always use tools** - Don't rely on training data for current market information
2. **Be thorough** - Use multiple tools to build complete picture (fundamentals + news + sentiment + earnings)
3. **Be specific** - Provide concrete numbers, dates, and data points
4. **Context matters** - Consider market conditions, sector trends, and company-specific factors
5. **Actionable insights** - End with clear investment thesis and risk assessment

TOOL USAGE GUIDELINES:
- For company basics: Use get_company_info or get_company_info_basic
- For earnings analysis: Use get_earnings_analysis and get_eps_trend together
- For market sentiment: Combine get_ticker_news with get_social_sentiment
- For SEC insights: Use get_sec_filing for latest developments
- For technical analysis: Use get_historical_prices for momentum and trends
- For competitive context: Use search_companies when relevant

RESPONSE STYLE:
- Start with executive summary (2-3 sentences)
- Provide detailed analysis with supporting data
- Include both bullish and bearish perspectives
- End with clear recommendation and confidence level
- Always cite specific tool data and numbers

Remember: You have access to real-time data through tools - use them extensively!"""


class FreeAgent:
    """
    Stock analysis agent using free local Ollama models.
    
    Provides same functionality as SimpleAgent but uses local models instead of OpenAI API.
    Perfect for cost-conscious analysis, offline usage, or when API limits are a concern.
    
    Attributes:
        model: Ollama model name (e.g., 'qwen3:32b', 'llama3.1:70b')
        instructions: System instructions for the agent
        temperature: Model temperature for response randomness
        
    Example:
        >>> agent = FreeAgent(model='qwen3:32b')
        >>> response = await agent.ask("Analyze AAPL's valuation and recent earnings")
        >>> print(response)
        >>>
        >>> # With custom model
        >>> agent = FreeAgent(model='llama3.1:70b', temperature=0.1)
    """
    
    def __init__(
        self,
        model: str = "qwen3:32b",
        temperature: float = 0.3,
        instructions: Optional[str] = None
    ):
        """
        Initialize FreeAgent with local Ollama model.
        
        Args:
            model: Ollama model name (default: qwen3:32b)
            temperature: Response randomness 0.0-1.0 (default: 0.3)
            instructions: Custom system instructions (default: DEFAULT_INSTRUCTIONS)
            
        Note:
            Requires Ollama to be installed and running locally.
            Install: https://ollama.com/download
            Pull model: `ollama pull qwen3:32b`
        """
        self.model = model
        self.temperature = temperature
        self.instructions = instructions or DEFAULT_INSTRUCTIONS
        self.tools = self._prepare_tools()
        
    def _prepare_tools(self) -> List[Dict[str, Any]]:
        """Convert agent tools to Ollama tool format."""
        ollama_tools = []
        
        for tool in AGENT_TOOLS:
            # Convert from openai-agents format to Ollama format
            ollama_tool = {
                'type': 'function',
                'function': {
                    'name': tool.name,
                    'description': tool.description,
                    'parameters': {
                        'type': 'object',
                        'properties': {},
                        'required': []
                    }
                }
            }
            
            # Add parameters if available
            if hasattr(tool, 'args_schema') and tool.args_schema:
                schema = tool.args_schema.model_json_schema()
                if 'properties' in schema:
                    ollama_tool['function']['parameters']['properties'] = schema['properties']
                if 'required' in schema:
                    ollama_tool['function']['parameters']['required'] = schema['required']
            
            ollama_tools.append(ollama_tool)
            
        return ollama_tools
        
    async def ask(self, question: str, show_tools: bool = True, show_model: bool = True) -> str:
        """
        Ask the agent a stock analysis question.
        
        Args:
            question: Your stock analysis question
            show_tools: Whether to show tools called (default: True)
            show_model: Whether to show model used (default: True)
            
        Returns:
            Analysis response string
            
        Example:
            >>> response = await agent.ask("What's TSLA's PE ratio and earnings trend?")
            >>> print(response)
        """
        messages = [
            {
                'role': 'system',
                'content': self.instructions
            },
            {
                'role': 'user', 
                'content': question
            }
        ]
        
        tools_called = []
        
        try:
            # Call Ollama with tools
            response: ChatResponse = chat(
                model=self.model,
                messages=messages,
                tools=self.tools,
                options={
                    'temperature': self.temperature
                }
            )
            
            # Handle tool calls
            while response.message.tool_calls:
                tools_called.extend([tc.function.name for tc in response.message.tool_calls])
                
                # Add assistant message with tool calls
                messages.append({
                    'role': 'assistant',
                    'content': response.message.content or '',
                    'tool_calls': [
                        {
                            'id': tc.function.name,
                            'type': 'function',
                            'function': {
                                'name': tc.function.name,
                                'arguments': tc.function.arguments  # Keep as dict, don't JSON stringify
                            }
                        }
                        for tc in response.message.tool_calls
                    ]
                })
                
                # Execute tools and add results
                for tool_call in response.message.tool_calls:
                    # Parse arguments if they're a string
                    arguments = tool_call.function.arguments
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    
                    result = await self._execute_tool(
                        tool_call.function.name,
                        arguments
                    )
                    
                    messages.append({
                        'role': 'tool',
                        'content': json.dumps(result),
                        'tool_call_id': tool_call.function.name
                    })
                
                # Get next response
                response = chat(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    options={
                        'temperature': self.temperature
                    }
                )
            
            # Build final response
            answer = response.message.content or "No response generated."
            
            # Add model info
            if show_model:
                answer = f"🤖 Model: {self.model} (local)\n\n{answer}"
            
            # Add tools called
            if show_tools and tools_called:
                answer += f"\n\n🔧 Tools called: {len(tools_called)}\n"
                for i, tool in enumerate(tools_called, 1):
                    answer += f"   {i}. {tool}\n"
            
            return answer
            
        except Exception as e:
            error_msg = f"Error with Ollama model '{self.model}': {str(e)}"
            if "connection" in str(e).lower():
                error_msg += "\n\n💡 Solutions:\n"
                error_msg += "   1. Install Ollama: https://ollama.com/download\n"
                error_msg += "   2. Start Ollama: `ollama serve`\n"
                error_msg += f"   3. Pull model: `ollama pull {self.model}`"
            return error_msg
    
    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool function by name with arguments."""
        # Find the tool
        target_tool = None
        for tool in AGENT_TOOLS:
            if tool.name == tool_name:
                target_tool = tool
                break
        
        if target_tool is None:
            return {"error": f"Tool '{tool_name}' not found"}
        
        try:
            # FunctionTool objects have a callable property or can be called directly
            if hasattr(target_tool, 'function') and callable(target_tool.function):
                # For openai-agents FunctionTool objects
                result = target_tool.function(**arguments)
            elif callable(target_tool):
                # If the tool itself is callable
                result = target_tool(**arguments)
            else:
                return {"error": f"Tool '{tool_name}' is not callable"}
            
            # Handle async results
            if hasattr(result, '__await__'):
                result = await result
            
            return result
            
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}
    
    def list_models(self) -> str:
        """
        List recommended Ollama models for stock analysis.
        
        Returns:
            String with model recommendations
        """
        return """
🤖 Recommended Ollama Models for Stock Analysis:

**Best Performance (Requires 32GB+ RAM):**
- qwen3:32b (recommended) - Excellent tool calling, financial reasoning
- llama3.1:70b - Strong analytical capabilities
- codellama:34b - Good with financial calculations

**Good Performance (Requires 8GB+ RAM):**
- qwen3:14b - Solid tool calling, faster than 32b
- llama3.1:8b - Fast, decent quality
- mistral:7b - Lightweight, good reasoning

**Installation:**
1. Install Ollama: https://ollama.com/download  
2. Pull model: `ollama pull qwen3:32b`
3. Start Ollama: `ollama serve`

**Usage:**
```python
agent = FreeAgent(model='qwen3:32b')
response = await agent.ask("Analyze AAPL")
```
        """
    
    def get_status(self) -> str:
        """
        Check Ollama connection and model status.
        
        Returns:
            Status information string
        """
        try:
            # Test connection with a simple call
            response = chat(model=self.model, messages=[{'role': 'user', 'content': 'test'}])
            return f"✅ Connected to Ollama\n✅ Model '{self.model}' available\n✅ Ready for analysis"
        except Exception as e:
            error = str(e).lower()
            if "connection" in error:
                return f"❌ Ollama not running\n💡 Start with: `ollama serve`"
            elif "not found" in error:
                return f"❌ Model '{self.model}' not found\n💡 Install with: `ollama pull {self.model}`"
            else:
                return f"❌ Error: {str(e)}"