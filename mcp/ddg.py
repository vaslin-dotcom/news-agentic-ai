
"""
DuckDuckGo News MCP Server
--------------------------
Fetches recent news articles via duckduckgo_search library.
No API key required.
"""
import asyncio
import httpx
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
from ddgs import DDGS

mcp = FastMCP("ddg-news-mcp")

# ---------------------------------------------------------------------------
# Helper
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

        # Remove junk tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Extract main content
        # Try article tag first, fall back to paragraphs
        article = soup.find("article")
        if article:
            paragraphs = article.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        full_text = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        return full_text if full_text else "Could not extract content."

    except Exception as e:
        return f"Could not fetch full content: {e}"

def _fetch_ddg_news(query: str, max_results: int = 20) -> list[dict]:
    """
    Uses duckduckgo_search library to fetch news articles.
    Returns a list of article dicts.
    """
    articles = []
    with DDGS() as ddgs:
        results = ddgs.news(query, max_results=max_results)
        for r in results:
            url=r.get("url",'')
            if not url:
                continue
            articles.append({
                "title":        r.get("title", ""),
                "url":          r.get("url", ""),
                "summary":      r.get("body", ""),
                "source":       r.get("source", ""),
                "published_at": r.get("date", ""),
            })
    return articles


# ---------------------------------------------------------------------------
# MCP Tool
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
    Fetch recent news articles from DuckDuckGo.

    :param query: Search query e.g. 'cricket world cup 2025'
    :param max_results: Number of articles to return (1-50). Default 20.
    :param full_content: If True, fetches full article text from each URL.
    """
    if not query.strip():
        raise ValueError("'query' cannot be empty.")

    max_results = max(1, min(max_results, 50))

    try:
        articles = await asyncio.to_thread(_fetch_ddg_news, query, max_results)
    except Exception as e:
        raise RuntimeError(f"Error fetching news: {e}")

    if not articles:
        return f"No articles found for query: '{query}'"

    # Fetch full content if requested
    if full_content:
        tasks = [_fetch_full_content(art["url"]) for art in articles]
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
    """Returns raw list of dicts — used by nodes, not MCP protocol."""
    return await asyncio.to_thread(_fetch_ddg_news, query, max_results)

if __name__ == "__main__":
    mcp.run(transport="stdio")
    # result=asyncio.run(fetch_news("gt vs pbks",20,True))
    # print(result)