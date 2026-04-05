"""
DuckDuckGo News MCP Server
--------------------------
Primary  : DuckDuckGo (ddgs library)
Fallback : Google News RSS (feedparser) — used when DDG returns 0 results
No API key required for either source.
"""
import asyncio
import httpx
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
from ddgs import DDGS
import feedparser
import urllib.parse

mcp = FastMCP("ddg-news-mcp")

# ---------------------------------------------------------------------------
# Full content fetcher
# ---------------------------------------------------------------------------

async def _fetch_full_content(url: str) -> str:
    """Fetch and extract full article text from URL."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        article = soup.find("article")
        if article:
            paragraphs = article.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        full_text = " ".join(
            p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
        )
        return full_text if full_text else "Could not extract content."

    except Exception as e:
        return f"Could not fetch full content: {e}"


# ---------------------------------------------------------------------------
# Primary source — DuckDuckGo
# ---------------------------------------------------------------------------
def _fetch_ddg_news(query: str, max_results: int = 20) -> list[dict]:
    articles = []
    try:
        # Fix: encode query to avoid Windows charmap issues
        safe_query = query.strip()
        with DDGS() as ddgs:
            results = ddgs.news(safe_query, max_results=max_results)
            for r in results:
                url = r.get("url", "")
                if not url:
                    continue
                articles.append({
                    "title":        r.get("title", ""),
                    "url":          url,
                    "summary":      r.get("body", ""),
                    "source":       r.get("source", ""),
                    "published_at": r.get("date", ""),
                    "_source":      "ddg",
                })
    except Exception as e:
        error_msg = str(e).encode('ascii', 'ignore').decode()
        print(f"[DDG ERROR] {error_msg}")

        if "403" in error_msg:
            print("[DDG BLOCKED] switching source")

    # any error → return empty → triggers fallback
    return articles


# ---------------------------------------------------------------------------
# Fallback source — Google News RSS
# ---------------------------------------------------------------------------

def _fetch_google_rss_news(query: str, max_results: int = 20) -> list[dict]:
    """
    Fallback: fetch news from Google News RSS feed.
    Free, no API key, very reliable.
    """
    articles = []
    try:
        encoded_query = urllib.parse.quote(query)
        rss_url = (
            f"https://news.google.com/rss/search"
            f"?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        )
        feed = feedparser.parse(rss_url)

        for entry in feed.entries[:max_results]:
            url = entry.get("link", "")
            if not url:
                continue
            # Google RSS wraps the real URL — extract it
            # format: https://news.google.com/rss/articles/...
            # published format: "Sat, 05 Apr 2026 10:00:00 GMT"
            articles.append({
                "title":        entry.get("title", ""),
                "url":          url,
                "summary":      entry.get("summary", ""),
                "source":       entry.get("source", {}).get("title", "Google News")
                                if isinstance(entry.get("source"), dict)
                                else entry.get("source", "Google News"),
                "published_at": entry.get("published", ""),
                "_source":      "google_rss",
            })
    except Exception as e:
        print(f"    ⚠ Google RSS error: {e}")
    return articles


# ---------------------------------------------------------------------------
# Combined fetch — DDG first, Google RSS fallback
# ---------------------------------------------------------------------------

def _fetch_news_with_fallback(query: str, max_results: int = 20) -> list[dict]:
    # Sanitize query at entry point — catches any unicode issues early
    safe_query = query.encode("utf-8", errors="ignore").decode("utf-8").strip()

    articles = _fetch_ddg_news(safe_query, max_results)

    if not articles:
        print(f"    ↩ DDG returned 0 — trying Google RSS for: '{safe_query}'")
        articles = _fetch_google_rss_news(safe_query, max_results)

        if not articles:
            print(f"    ✗ Both sources returned 0 for: '{safe_query}'")

    return articles

# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def fetch_article_content(url: str) -> str:
    """
    Fetch and extract full text content from a specific article URL.
    Use this when you already have the URL and want the full article text.
    No search performed — fetches directly from the URL.

    :param url: The direct article URL to fetch content from
    """
    if not url.strip():
        raise ValueError("'url' cannot be empty.")

    content = await _fetch_full_content(url)
    return content


@mcp.tool()
async def fetch_news(query: str, max_results: int = 20, full_content: bool = False) -> str:
    """
    Fetch recent news articles. Tries DuckDuckGo first, falls back to
    Google News RSS automatically if DDG returns no results.

    :param query: Search query e.g. 'cricket world cup 2026'
    :param max_results: Number of articles to return (1-50). Default 20.
    :param full_content: If True, fetches full article text from each URL.
    """
    if not query.strip():
        raise ValueError("'query' cannot be empty.")

    max_results = max(1, min(max_results, 50))

    articles = await asyncio.to_thread(
        _fetch_news_with_fallback, query, max_results
    )

    if not articles:
        return f"No articles found for query: '{query}'"

    # Fetch full content if requested
    if full_content:
        tasks      = [_fetch_full_content(art["url"]) for art in articles]
        full_texts = await asyncio.gather(*tasks)
        for art, text in zip(articles, full_texts):
            art["full_content"] = text

    lines = [f"Found {len(articles)} articles for '{query}':\n"]
    for i, art in enumerate(articles, 1):
        lines.append(
            f"{i}. {art['title']}\n"
            f"   URL: {art['url']}\n"
            f"   Source: {art['source']}\n"
            f"   Published: {art['published_at']}\n"
            f"   Summary: {art['summary']}\n"
        )
        if full_content and "full_content" in art:
            lines.append(f"   Full Content: {art['full_content']}\n")

    return "\n".join(lines)


async def fetch_news_structured(query: str, max_results: int = 20) -> list[dict]:
    """Returns raw list of dicts — used by nodes directly, not MCP protocol."""
    return await asyncio.to_thread(_fetch_news_with_fallback, query, max_results)


if __name__ == "__main__":
    mcp.run(transport="stdio")