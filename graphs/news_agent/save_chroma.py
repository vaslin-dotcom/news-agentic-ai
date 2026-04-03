"""
embed_node.py
-------------
Embeds all scored articles into Chroma article_vectors collection.
Runs in parallel with save_node.
"""

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from state import NewsState

MCP_SERVERS = {
    "chroma": {
        "command": "python",
        "args": ["../../mcp/chroma.py"],
        "transport": "stdio"
    }
}


def _unwrap(raw) -> str:
    item = raw[0] if isinstance(raw, list) else raw
    return item["text"] if isinstance(item, dict) else item.text


async def _embed(state: NewsState) -> dict:
    articles = state["scored_articles"]

    client = MultiServerMCPClient(MCP_SERVERS)
    tools  = {t.name: t for t in await client.get_tools()}

    print(f"\n  Embedding {len(articles)} articles into Chroma...")

    for article in articles:
        try:
            # document = title + summary for semantic search
            document = f"{article.get('title', '')} {article.get('summary', '')}"
            url      = article.get("url", "")

            raw = await tools["upsert_embedding"].ainvoke({
                "collection": "article_vectors",
                "id"        : url,
                "document"  : document,
                "metadata"  : {
                    "source"      : article.get("source", ""),
                    "published_at": article.get("published_at", ""),
                    "urgency"     : int(article.get("urgency", 1)),
                    "url"         : url,
                },
            })
            print(f"  ✓ {article.get('title', '')[:60]}")

        except Exception as e:
            print(f"  ⚠ Embed failed for '{article.get('title', '')[:40]}': {e}")

    return {"errors": []}


def embed_node(state: NewsState) -> dict:
    return asyncio.run(_embed(state))


# ── Test block ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = [
        {
            "title"   : "Tech Layoffs Surge While AI Jobs Soar",
            "url"     : "https://www.techtimes.com/test-layoffs",
            "source"  : "Tech Times",
            "summary" : "Tech layoffs exceed 45,000 globally in early 2026.",
            "published_at": "2026-03-21T11:23:00+00:00",
            "urgency" : 4,
        },
        {
            "title"   : "IPL 2026 schedule this week",
            "url"     : "https://sports.yahoo.com/test-ipl",
            "source"  : "Yahoo Sports",
            "summary" : "IPL 2026 kicked off with RCB winning.",
            "published_at": "2026-03-31T08:12:46+00:00",
            "urgency" : 1,
        },
    ]

    embed_node({
        "profile"          : {},
        "profile_chunks"   : [],
        "search_queries"   : [],
        "raw_articles"     : [],
        "filtered_articles": [],
        "scored_articles"  : sample,
        "alert_articles"   : [],
        "errors"           : [],
    })

    print("\n  Done. Check Chroma article_vectors collection.")