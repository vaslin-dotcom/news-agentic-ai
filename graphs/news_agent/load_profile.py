"""
load_profile_node.py
--------------------
Loads full profile from SQLite and all profile chunks from Chroma.
No agent needed — calls tools directly.
"""

import json
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from state import NewsState
import re

MCP_SERVERS = {
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


def _parse_chroma_text(text: str) -> list[dict]:
    """Parse Chroma query_similar text output into list of dicts."""
    chunks = []
    entries = re.split(r"\n\d+\.\s+ID:", text)
    for entry in entries:
        entry = entry.strip()
        if not entry or entry.startswith("Top "):
            continue
        chunk = {}
        # ID is the first line (we split on "ID:")
        chunk["id"] = entry.splitlines()[0].strip()
        sim = re.search(r"Similarity:\s*([-\d.]+)", entry)
        if sim:
            chunk["similarity"] = float(sim.group(1))
        doc = re.search(r"Document:\s*(.+?)(?=\n\s+Metadata:)", entry, re.DOTALL)
        if doc:
            chunk["document"] = doc.group(1).strip().rstrip(".")
        meta = re.search(r"Metadata:\s*(\{.+?\})", entry, re.DOTALL)
        if meta:
            try:
                chunk["metadata"] = json.loads(meta.group(1))
            except json.JSONDecodeError:
                chunk["metadata"] = {}
        if chunk.get("id"):
            chunks.append(chunk)
    return chunks

async def _load(state: NewsState) -> dict:
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = {t.name: t for t in await client.get_tools()}

    # ── SQLite: get full profile ──────────────────────────────
    raw = await tools["get_profile"].ainvoke({})
    # unwrap: raw is a list containing {'type':'text', 'text': {...}, 'id':...}
    item = raw[0] if isinstance(raw, list) else raw
    text = item["text"] if isinstance(item, dict) else item.text
    profile = json.loads(text) if isinstance(text, str) else text
    # parse JSON-encoded list values
    for key, val in profile.items():
        try:
            profile[key] = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            pass

    # ── Chroma: get all profile vector chunks ─────────────────
    raw_chunks = await tools["query_similar"].ainvoke({
        "collection" : "profile_vectors",
        "query_text" : "professional background skills interests lifestyle personal",
        "n_results"  : 10,
    })
    item = raw_chunks[0] if isinstance(raw_chunks, list) else raw_chunks
    chunks_text = item["text"] if isinstance(item, dict) else item.text
    profile_chunks = _parse_chroma_text(chunks_text)

    return {
        "profile"       : profile,
        "profile_chunks": profile_chunks,
    }

def load_profile_node(state: NewsState) -> dict:
    return asyncio.run(_load(state))


# ── Test block ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint

    result = load_profile_node({
        "profile"          : {},
        "profile_chunks"   : [],
        "search_queries"   : [],
        "raw_articles"     : [],
        "filtered_articles": [],
        "scored_articles"  : [],
        "alert_articles"   : [],
        "errors"           : [],
    })

    print("\n" + "═" * 60)
    print("  PROFILE (SQLite)")
    print("═" * 60)
    pprint.pprint(result["profile"])

    print("\n" + "═" * 60)
    print("  PROFILE CHUNKS (Chroma)")
    print("═" * 60)
    for chunk in result["profile_chunks"]:
        pprint.pprint(chunk)
        print()

    print(f"\nTotal chunks: {len(result['profile_chunks'])}")