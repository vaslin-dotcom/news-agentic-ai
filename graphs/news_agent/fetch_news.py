"""
fetch_news_node.py
------------------
Fetches news for all generated queries using DDG MCP tool.
Deduplicates by URL and filters already-seen articles via SQLite MCP tool.
All calls go through MCP tools — no direct Python APIs.
"""

import json
import asyncio
import re
from langchain_mcp_adapters.client import MultiServerMCPClient
from state import NewsState

MCP_SERVERS = {
    "ddg": {
        "command": "python",
        "args": ["../../mcp/ddg.py"],
        "transport": "stdio"
    },
    "sqlite": {
        "command": "python",
        "args": ["../../mcp/sqlite.py"],
        "transport": "stdio"
    }
}


def _unwrap(raw) -> str:
    item = raw[0] if isinstance(raw, list) else raw
    return item["text"] if isinstance(item, dict) else item.text


def _parse_ddg_response(text: str) -> list[dict]:
    """
    Parse DDG fetch_news text output into list of article dicts.
    Format per entry:
        1. Title
           URL: ...
           Source: ...
           Published: ...
           Summary: ...
    """
    articles = []
    # split on numbered entries
    entries = re.split(r"\n\d+\.\s+", text)
    for entry in entries:
        entry = entry.strip()
        if not entry or entry.startswith("Found "):
            continue

        article = {}

        # title is the first line
        lines = entry.splitlines()
        if lines:
            article["title"] = lines[0].strip()

        url     = re.search(r"URL:\s*(.+)",       entry)
        source  = re.search(r"Source:\s*(.+)",    entry)
        date    = re.search(r"Published:\s*(.+)", entry)
        summary = re.search(r"Summary:\s*(.+)",   entry, re.DOTALL)

        if url:
            article["url"]          = url.group(1).strip()
        if source:
            article["source"]       = source.group(1).strip()
        if date:
            article["published_at"] = date.group(1).strip()
        if summary:
            article["summary"]      = summary.group(1).strip()

        if article.get("url"):
            articles.append(article)

    return articles


async def _fetch(state: NewsState) -> dict:
    queries    = state["search_queries"]
    exclusions = state["profile"].get("news_exclusions", [])
    if isinstance(exclusions, str):
        exclusions = json.loads(exclusions)
    exclusions = [e.lower() for e in exclusions]

    client = MultiServerMCPClient(MCP_SERVERS)
    tools  = {t.name: t for t in await client.get_tools()}

    # ── Step 1: fetch news for ALL queries via DDG MCP ────────────
    print(f"\n  Fetching news for {len(queries)} queries...")
    seen_urls = {}   # url → article dict

    for i, query in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] {query}")
        try:
            raw      = await tools["fetch_news"].ainvoke({
                "query":       query,
                "max_results": 1,                                                                   #20
            })
            text     = _unwrap(raw)
            articles = _parse_ddg_response(text)
            for art in articles:
                url = art.get("url", "").strip()
                if url and url not in seen_urls:
                    seen_urls[url] = art
        except Exception as e:
            print(f"    ⚠ DDG error for '{query}': {e}")

    print(f"\n  Total unique articles after dedup: {len(seen_urls)}")

    # ── Step 2: hard exclusion filter ─────────────────────────────
    if exclusions:
        before    = len(seen_urls)
        seen_urls = {
            url: art for url, art in seen_urls.items()
            if not any(
                excl in (art.get("title", "") + art.get("summary", "")).lower()
                for excl in exclusions
            )
        }
        print(f"  After exclusion filter : {len(seen_urls)} "
              f"(dropped {before - len(seen_urls)})")

    # ── Step 3: filter already-seen URLs via SQLite MCP ───────────
    new_articles = []
    skipped      = 0

    for url, art in seen_urls.items():
        try:
            raw    = await tools["url_exists"].ainvoke({"url": url})
            result = json.loads(_unwrap(raw))
            if result.get("exists"):
                skipped += 1
            else:
                new_articles.append(art)
        except Exception as e:
            new_articles.append(art)

    print(f"  Already seen (skipped) : {skipped}")
    print(f"  New articles to process: {len(new_articles)}")

    return {"raw_articles": new_articles}


def fetch_news_node(state: NewsState) -> dict:
    return asyncio.run(_fetch(state))


# ── Test block ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_queries =  ['Infosys Mysore campus news 2026', 'Infosys layoffs 2026', 'Infosys AI strategy 2026', 'AI ML Engineer salary India 2026', 'AI ML Engineer demand Bangalore 2026', 'IT services AI market 2026', 'Mysore tech startups 2026', 'LangGraph 0.3 release 2026', 'LangChain new features 2026', 'RAG pipelines latest tools 2026', 'Telegram bots API update 2026', 'OCR Python libraries 2026', 'Raspberry Pi 5 projects 2026', 'Spiking Neural Networks breakthrough 2026', 'multi-agent systems research 2026', 'real-time inference edge AI 2026', 'conversational AI startups India 2026', 'assistive hardware AI 2026', 'neuromorphic computing Intel 2026', 'autonomous agents frameworks 2026', 'IPL 2026 schedule', 'India cricket fixtures 2026', 'Tamil movies 2026 releases', 'Malayalam films 2026', 'Telugu cinema 2026', 'Mysore local events 2026', 'Karnataka state news 2026', 'AI ML jobs Mysore 2026', 'AI certifications India 2026', 'AI conferences Bangalore 2026', 'startup funding AI agents 2026', 'RAG systems production deployment 2026', 'voice AI agents India 2026', 'edge AI Raspberry Pi 2026', 'FastAPI latest update 2026', 'Pandas 3.0 release 2026']

    result = fetch_news_node({
        "profile"          : {"news_exclusions": ["gossips"]},
        "profile_chunks"   : [],
        "search_queries"   : sample_queries,
        "raw_articles"     : [],
        "filtered_articles": [],
        "scored_articles"  : [],
        "alert_articles"   : [],
        "errors"           : [],
    })

    print('='*25,'RESULTS','='*25)
    print(result)
    print('='*50)

    articles = result["raw_articles"]
    print(f"\n{'═' * 60}")
    print(f"  RAW ARTICLES — {len(articles)} new articles")
    print(f"{'═' * 60}\n")
    for i, art in enumerate(articles, 1):
        print(f"  [{i}] {art.get('title', 'No title')}")
        print(f"       Source : {art.get('source', '')}")
        print(f"       Date   : {art.get('published_at', '')}")
        print(f"       URL    : {art.get('url', '')}")
        print(f"       Summary: {art.get('summary', '')[:100]}...")
        print()