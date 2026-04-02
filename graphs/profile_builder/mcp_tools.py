"""
mcp_tools.py
------------
Loads all MCP servers once at import time.
Each get_*_tools() function returns the relevant tool subset
for that node to use.
"""

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

MCP_SERVERS = {
    "github": {
        "command": "python",
        "args": ["../../mcp/github.py"],
        "transport": "stdio"
    },
    "sqlite": {
        "command": "python",
        "args": ["../../mcp/sqlite.py"],
        "transport": "stdio"
    },
    "chroma": {
        "command": "python",
        "args": ["../../mcp/chroma.py"],
        "transport": "stdio"
    }
}

async def _load_all_tools():
    client = MultiServerMCPClient(MCP_SERVERS)
    return await client.get_tools()

# Load once at module import — stays alive for entire run
ALL_TOOLS = asyncio.run(_load_all_tools())

# ── Tool name → tool object lookup for direct .invoke() calls ──────────────
_TOOL_MAP = {t.name: t for t in ALL_TOOLS}

def get_tool(name: str):
    """Get a single tool by name for direct .invoke() calls."""
    if name not in _TOOL_MAP:
        raise ValueError(f"Tool '{name}' not found. Available: {list(_TOOL_MAP.keys())}")
    return _TOOL_MAP[name]

def get_github_tools():
    return [t for t in ALL_TOOLS if t.name in (
        "get_user_profile", "get_repositories", "get_languages", "get_pinned_topics"
    )]

def get_sqlite_tools():
    return [t for t in ALL_TOOLS if t.name in (
        "upsert_profile", "get_profile",
        "save_article", "save_alert", "save_feedback",
        "get_articles", "url_exists", "execute_query",
    )]

def get_sqlite_read_tools():
    """Read-only SQLite tools — for nodes that only query, never write."""
    return [t for t in ALL_TOOLS if t.name in (
        "get_profile", "get_articles", "url_exists", "execute_query",
    )]

def get_sqlite_write_tools():
    """Write-only SQLite tools — for save nodes."""
    return [t for t in ALL_TOOLS if t.name in (
        "upsert_profile", "save_article", "save_alert", "save_feedback",
    )]

def get_chroma_tools():
    return [t for t in ALL_TOOLS if t.name in (
        "upsert_embedding", "query_similar", "delete_embedding", "list_collections",
    )]