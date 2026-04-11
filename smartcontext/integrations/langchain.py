"""LangChain integration — use SmartContext as a step in LangChain chains."""

from typing import Any, Optional


class LangChainMiddleware:
    """
    Wraps SmartContext as a LangChain-compatible runnable.

    Usage:
        from smartcontext import SmartContext
        from smartcontext.integrations.langchain import LangChainMiddleware

        ctx = SmartContext.from_yaml("bot.yaml")
        middleware = LangChainMiddleware(ctx)
        result = await middleware.ainvoke("Where is my order?")
    """

    def __init__(self, ctx: Any, session_id: str = "langchain_default"):
        try:
            from langchain_core.runnables import RunnableSerializable  # noqa: F401
        except ImportError:
            raise ImportError(
                "LangChain integration requires langchain-core. "
                "Install with: pip install langchain-core"
            )
        self.ctx = ctx
        self.session_id = session_id

    async def ainvoke(self, input: Any, config: Optional[dict] = None) -> str:
        message = self._extract_message(input)
        result = await self.ctx.chat(message, session_id=self.session_id)
        return result.response

    def invoke(self, input: Any, config: Optional[dict] = None) -> str:
        message = self._extract_message(input)
        result = self.ctx.chat_sync(message, session_id=self.session_id)
        return result.response

    def _extract_message(self, input: Any) -> str:
        if isinstance(input, str):
            return input
        if isinstance(input, dict):
            return input.get("input", input.get("question", input.get("message", str(input))))
        if isinstance(input, list):
            # List of messages — take the last user message
            for msg in reversed(input):
                if hasattr(msg, "content") and hasattr(msg, "type"):
                    if msg.type == "human":
                        return msg.content
                elif isinstance(msg, dict) and msg.get("role") == "user":
                    return msg.get("content", "")
            return str(input[-1]) if input else ""
        return str(input)
