# mcp_tools.py
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

def get_github_tools():
    return [t for t in ALL_TOOLS if t.name in (
        "get_user_profile", "get_repositories", "get_languages", "get_pinned_topics"
    )]

def get_sqlite_tools():
    return [t for t in ALL_TOOLS if t.name in ("upsert_profile",)]

def get_chroma_tools():
    return [t for t in ALL_TOOLS if t.name in ("upsert_embedding",)]