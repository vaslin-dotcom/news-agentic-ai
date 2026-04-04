"""
alert_node.py
-------------
Sends Telegram alerts for all articles with urgency >= 3.
Runs AFTER save_node — relies on article_id being set by save_node.
"""

import asyncio
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from state import NewsState

MCP_SERVERS = {
    "telegram": {
        "command": "python",
        "args": ["../../mcp/telegram.py"],
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


async def _alert(state: NewsState) -> dict:
    articles = state["scored_articles"]

    alert_articles = [
        a for a in articles
        if a.get("urgency", 1) >= 3 and a.get("article_id") is not None
    ]

    missing_id = [
        a for a in articles
        if a.get("urgency", 1) >= 3 and a.get("article_id") is None
    ]
    if missing_id:
        print(f"\n  ⚠ {len(missing_id)} urgent article(s) missing article_id — skipped:")
        for a in missing_id:
            print(f"    - {a.get('title', '')[:60]}")

    if not alert_articles:
        print("\n  No articles with urgency >= 3 and valid article_id. No alerts sent.")
        return {"alert_articles": []}

    client = MultiServerMCPClient(MCP_SERVERS)
    tools  = {t.name: t for t in await client.get_tools()}

    print(f"\n  Sending {len(alert_articles)} Telegram alerts...")

    sent = []
    for article in alert_articles:
        try:
            # Build the article dict explicitly — never pass the full article dict
            # send_alert expects flat named params, not a nested article object
            raw = await tools["send_alert"].ainvoke({
                "article"     : json.dumps({
                    "title"       : str(article.get("title", "")),
                    "url"         : str(article.get("url", "")),
                    "summary"     : str(article.get("summary", "")),
                    "source"      : str(article.get("source", "")),
                    "published_at": str(article.get("published_at", "")),
                }),
                "urgency"     : int(article.get("urgency", 3)),
                "reasoning"   : str(article.get("reasoning", "")),
                "article_id"  : int(article["article_id"]),
                "connected_to": json.dumps([
                    {
                        "title"       : str(c.get("title", "")),
                        "published_at": str(c.get("published_at", "")),
                    }
                    for c in article.get("connected_to", [])
                    if isinstance(c, dict)
                ]),
            })

            print(f"  ✅ {_unwrap(raw)}")
            sent.append(article)

        except Exception as e:
            print(f"  ⚠ Alert failed for '{article.get('title', '')[:40]}': {e}")
            import traceback
            traceback.print_exc()

    return {"alert_articles": sent}


def alert_node(state: NewsState) -> dict:
    return asyncio.run(_alert(state))


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
            "reasoning"      : "As an early-career AI/ML Engineer at Infosys Mysore, this directly affects career planning.",
            "connected_to"   : [],
            "article_id"     : 1,
        },
    ]

    result = alert_node({
        "profile": {}, "profile_chunks": [], "search_queries": [],
        "raw_articles": [], "filtered_articles": [],
        "scored_articles": sample, "alert_articles": [], "errors": [],
    })

    print(f"\n  Alerts sent: {len(result['alert_articles'])}")