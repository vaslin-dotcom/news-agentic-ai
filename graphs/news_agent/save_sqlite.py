"""
save_node.py
------------
Saves all scored articles to SQLite.
Saves alerts for urgency >= 3.
Stores article_id back onto each article dict for alert_node.
"""

import json
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from state import NewsState

MCP_SERVERS = {
    "sqlite": {
        "command": "python",
        "args": ["../../mcp/sqlite.py"],
        "transport": "stdio"
    }
}


def _unwrap(raw) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        item = raw[0]
        if isinstance(item, dict):
            return item.get("text") or item.get("content") or str(item)
        return str(item)
    if hasattr(raw, "text"):
        return raw.text
    return str(raw)


async def _save(state: NewsState) -> dict:
    articles = state["scored_articles"]

    client = MultiServerMCPClient(MCP_SERVERS)
    tools  = {t.name: t for t in await client.get_tools()}

    print(f"\n  Saving {len(articles)} articles to SQLite...")

    for article in articles:
        try:
            raw    = await tools["save_article"].ainvoke({
                "url"            : article.get("url", ""),
                "title"          : article.get("title", ""),
                "source"         : article.get("source", ""),
                "summary"        : article.get("summary", ""),
                "published_at"   : article.get("published_at", ""),
                "relevance_score": float(article.get("relevance_score", 0.0)),
                "urgency"        : int(article.get("urgency", 1)),
                "reasoning"      : article.get("reasoning", ""),
                "embedding_id"   : article.get("url", ""),
            })

            result = json.loads(_unwrap(raw))
            article["article_id"] = result.get("article_id")
            status = result.get("status", "?")
            print(f"  [{status}] id={article['article_id']}  {article.get('title', '')[:55]}")

            if article.get("urgency", 1) >= 3 and status == "saved":
                await tools["save_alert"].ainvoke({
                    "article_id": article["article_id"],
                    "urgency"   : int(article.get("urgency", 1)),
                    "reasoning" : article.get("reasoning", ""),
                })
                print(f"    → Alert saved (urgency {article.get('urgency')})")

        except Exception as e:
            print(f"  ⚠ Save failed for '{article.get('title', '')[:40]}': {e}")
            article["article_id"] = None

    return {"scored_articles": articles}


def save_node(state: NewsState) -> dict:
    return asyncio.run(_save(state))


if __name__ == "__main__":
    sample = [
        {
            "title"          : "Tech Layoffs Surge While AI Jobs Soar",
            "url"            : "https://www.techtimes.com/test-layoffs",
            "source"         : "Tech Times",
            "summary"        : "Tech layoffs exceed 45,000 globally in early 2026.",
            "published_at"   : "2026-03-21T11:23:00+00:00",
            "relevance_score": 0.85,
            "urgency"        : 4,
            "reasoning"      : "Directly affects career planning.",
            "connected_to"   : [],
        },
        {
            "title"          : "IPL 2026 schedule this week",
            "url"            : "https://sports.yahoo.com/test-ipl",
            "source"         : "Yahoo Sports",
            "summary"        : "IPL 2026 kicked off with RCB winning.",
            "published_at"   : "2026-03-31T08:12:46+00:00",
            "relevance_score": 0.0,
            "urgency"        : 1,
            "reasoning"      : "Routine IPL schedule.",
            "connected_to"   : [],
        },
    ]

    result = save_node({
        "profile": {}, "profile_chunks": [], "search_queries": [],
        "raw_articles": [], "filtered_articles": [],
        "scored_articles": sample, "alert_articles": [], "errors": [],
    })

    print("\n  article_ids assigned:")
    for art in result["scored_articles"]:
        print(f"  {art.get('title', '')[:50]} → id={art.get('article_id')}")