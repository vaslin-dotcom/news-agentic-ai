"""
mcp_tools.py
------------
MCP session management. Keeps the session alive for the full agent run.
Use run_with_*_tools() instead of get_*_tools() to avoid ClosedResourceError.
"""

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools


async def run_with_tools(script_path: str, coro_fn):
    """
    Spins up an MCP server, loads tools, and calls coro_fn(tools)
    — all within the same session context so the session stays alive.

    coro_fn: async callable that accepts a tools list and returns a result.
    """
    server_params = StdioServerParameters(command="python", args=[script_path])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            return await coro_fn(tools)


async def run_with_github_tools(coro_fn):
    return await run_with_tools("../../mcp/github.py", coro_fn)


async def run_with_sqlite_tools(coro_fn):
    return await run_with_tools("../../mcp/sqlite.py", coro_fn)


async def run_with_chroma_tools(coro_fn):
    return await run_with_tools("../../mcp/chroma.py", coro_fn)