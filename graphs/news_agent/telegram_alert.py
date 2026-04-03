"""
alert_node.py
-------------
Sends Telegram alerts for all articles with urgency >= 3.
Requires article_id to be set by save_node (runs after 6a + 6b).
"""

import asyncio
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
    item = raw[0] if isinstance(raw, list) else raw
    return item["text"] if isinstance(item, dict) else item.text


async def _alert(state: NewsState) -> dict:
    articles = state["scored_articles"]

    # filter only urgency >= 3
    alert_articles = [a for a in articles if a.get("urgency", 1) >= 3]

    if not alert_articles:
        print("\n  No articles with urgency >= 3. No alerts sent.")
        return {"alert_articles": []}

    client = MultiServerMCPClient(MCP_SERVERS)
    tools  = {t.name: t for t in await client.get_tools()}

    print(f"\n  Sending {len(alert_articles)} Telegram alerts...")

    sent = []
    for article in alert_articles:
        article_id = article.get("article_id")
        if not article_id:
            print(f"  ⚠ No article_id for '{article.get('title', '')[:40]}' — skipping")
            continue

        try:
            raw = await tools["send_alert"].ainvoke({
                "article"     : {
                    "title"       : article.get("title", ""),
                    "url"         : article.get("url", ""),
                    "summary"     : article.get("summary", ""),
                    "source"      : article.get("source", ""),
                    "published_at": article.get("published_at", ""),
                },
                "urgency"     : int(article.get("urgency", 3)),
                "reasoning"   : article.get("reasoning", ""),
                "article_id"  : int(article_id),
                "connected_to": article.get("connected_to", []),
            })
            print(f"  ✅ Alert sent: {article.get('title', '')[:60]}")
            print(f"     Urgency: {article.get('urgency')}/5")
            sent.append(article)

        except Exception as e:
            print(f"  ⚠ Alert failed for '{article.get('title', '')[:40]}': {e}")

    return {"alert_articles": sent}


def alert_node(state: NewsState) -> dict:
    return asyncio.run(_alert(state))


# ── Test block ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Note: save_node must run first to assign article_ids
    # For testing, we manually set article_id
    sample = [
        {
            "title"          : "Tech Layoffs Surge While AI Jobs Soar",
            "url"            : "https://www.techtimes.com/test-layoffs",
            "source"         : "Tech Times",
            "summary"        : "Tech layoffs exceed 45,000 globally in early 2026.",
            "published_at"   : "2026-03-21T11:23:00+00:00",
            "relevance_score": 0.85,
            "urgency"        : 4,
            "reasoning"      : "As an early-career AI/ML Engineer at Infosys Mysore, this directly affects Antony's career planning. The preference for experienced candidates is a key signal — he needs to build demonstrable expertise in AI agents and RAG systems urgently.",
            "connected_to"   : [],
            "article_id"     : 1,  # set by save_node in real run
        },
    ]

    result = alert_node({
        "profile"          : {},
        "profile_chunks"   : [],
        "search_queries"   : [],
        "raw_articles"     : [],
        "filtered_articles": [],
        "scored_articles"  : sample,
        "alert_articles"   : [],
        "errors"           : [],
    })

    print(f"\n  Alerts sent: {len(result['alert_articles'])}")