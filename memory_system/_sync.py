import asyncio


def run_sync(coro):
    """Run an async coroutine synchronously. For use in non-async contexts (Flask, Django, scripts)."""
    try:
        asyncio.get_running_loop()
        raise RuntimeError(
            "Cannot use chat_sync() from within an async context. "
            "Use 'await ctx.chat()' instead."
        )
    except RuntimeError as e:
        if "no current event loop" in str(e) or "no running event loop" in str(e):
            return asyncio.run(coro)
        raise
