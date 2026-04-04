"""
relevance_filter_node.py
------------------------
Filters raw articles by relevance using Chroma vector similarity.
Parallel processing — all articles scored simultaneously via asyncio.gather.
"""

import json
import asyncio
import re
from langchain_mcp_adapters.client import MultiServerMCPClient
from state import NewsState

MCP_SERVERS = {
    "chroma": {
        "url"      : "http://localhost:8001/mcp/",
        "transport": "streamable_http"  # ← change transport too
    }
}

THRESHOLD = -0.5


def _unwrap(raw) -> str:
    item = raw[0] if isinstance(raw, list) else raw
    return item["text"] if isinstance(item, dict) else item.text


def _parse_chroma_text(text: str) -> list[dict]:
    chunks  = []
    entries = re.split(r"\n\d+\.\s+ID:", text)
    for entry in entries:
        entry = entry.strip()
        if not entry or entry.startswith("Top "):
            continue
        chunk = {}
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


def _check_exclusions(article: dict, exclusions: list[str]) -> bool:
    text = (article.get("title", "") + " " + article.get("summary", "")).lower()
    for excl in exclusions:
        if excl.lower().rstrip("s") in text:
            return True
    return False


async def _score_article(tools: dict, article: dict, exclusions: list[str]) -> dict | None:
    """
    Score a single article. Returns enriched article dict if KEEP, None if DROP.
    This runs concurrently for all articles via asyncio.gather.
    """
    title      = article.get("title", "")
    summary    = article.get("summary", "")
    query_text = f"{title} {summary}"

    # ── Step 1: hard exclusion ────────────────────────────────
    if _check_exclusions(article, exclusions):
        return None

    # ── Step 2: profile similarity ────────────────────────────
    try:
        raw             = await tools["query_similar"].ainvoke({
            "collection": "profile_vectors",
            "query_text": query_text,
            "n_results"  : 3,
        })
        profile_results = _parse_chroma_text(_unwrap(raw))
    except Exception:
        profile_results = []

    if profile_results:
        best_similarity = max(r.get("similarity", -999) for r in profile_results)
        top_chunks      = profile_results
    else:
        best_similarity = -999
        top_chunks      = []

    # ── Step 3: feedback penalty ──────────────────────────────
    penalty          = 0.0
    similar_feedback = []
    try:
        raw  = await tools["query_similar"].ainvoke({
            "collection": "feedback_vectors",
            "query_text": query_text,
            "n_results"  : 3,
        })
        text = _unwrap(raw)
        if "is empty" not in text:
            for fb in _parse_chroma_text(text):
                signal = fb.get("metadata", {}).get("signal", "")
                if signal == "skip":
                    penalty = max(penalty, 0.4)
                elif signal == "irrelevant":
                    penalty = max(penalty, 0.2)
                similar_feedback.append(fb)
    except Exception:
        pass

    # ── Step 4: threshold ─────────────────────────────────────
    final_score = best_similarity - penalty
    if final_score < THRESHOLD:
        return None

    return {
        **article,
        "similarity_score"         : best_similarity,
        "feedback_penalty"         : penalty,
        "final_score"              : final_score,
        "matching_profile_chunks"  : top_chunks,
        "similar_feedback_articles": similar_feedback,
    }


async def _filter(state: NewsState) -> dict:
    articles   = state["raw_articles"]
    profile    = state["profile"]
    exclusions = profile.get("news_exclusions", [])
    if isinstance(exclusions, str):
        exclusions = json.loads(exclusions)
    exclusions = [e.lower() for e in exclusions]

    client = MultiServerMCPClient(MCP_SERVERS)
    tools  = {t.name: t for t in await client.get_tools()}

    print(f"\n  Filtering {len(articles)} articles in parallel...")

    # ── Run all articles concurrently ─────────────────────────
    tasks   = [_score_article(tools, article, exclusions) for article in articles]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # ── Collect results ───────────────────────────────────────
    filtered = []
    dropped  = 0
    for article, result in zip(articles, results):
        if isinstance(result, Exception):
            print(f"  ⚠ Error scoring '{article.get('title','')[:40]}': {result}")
            dropped += 1
        elif result is None:
            dropped += 1
        else:
            filtered.append(result)
            print(f"  KEEP  score={result['final_score']:.3f}  {result['title'][:60]}")

    print(f"\n  Kept    : {len(filtered)}")
    print(f"  Dropped : {dropped}")

    return {"filtered_articles": filtered}


def relevance_filter_node(state: NewsState) -> dict:
    return asyncio.run(_filter(state))


# ── Test block ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_articles = [{'title': 'A promise of CSR funds and Rs 6.3 crore fraud by fake Infosys employees', 'url': 'https://www.msn.com/en-in/money/general/a-promise-of-csr-funds-and-rs-6-3-crore-fraud-by-fake-infosys-employees/ar-AA207SEM?ocid=BingNewsVerp', 'source': 'NDTV on MSN', 'published_at': '2026-02-21T08:04:34+00:00', 'summary': 'It all began with one Gagan N Deep initiating a meeting with the Mysore Mercantile Company Limited (MMCL), claiming to be a ...'}, {'title': 'Tech layoffs surpass 45,000 in early 2026', 'url': 'https://www.networkworld.com/article/4143749/tech-layoffs-surpass-45000-in-early-2026.html', 'source': 'Network World', 'published_at': '2026-03-10T18:30:00+00:00', 'summary': 'Tech companies have announced more than 45,000 layoffs since the start of 2026, as firms across the industry restructure operations and shift resources toward AI and automation investments. A recent ...'}, {'title': '90 days, 1 lakh layoffs: What happens to workers after the pink slip?', 'url': 'https://www.msn.com/en-in/money/economy/90-days-1-lakh-layoffs-what-happens-to-workers-after-the-pink-slip/ar-AA1ZSi55?ocid=BingNewsVerp', 'source': 'India Today on MSN', 'published_at': '2026-04-01T08:04:40+00:00', 'summary': 'Nearly one lakh professionals are facing uncertain job prospects as large-scale layoffs continue through 2026. After ...'}, {'title': 'TCS, Infosys, and the Tech rout: Nifty IT Index down 23% in 2026 – Is the ‘AI Correction’ finally over?', 'url': 'https://www.financialexpress.com/market/tcs-infosys-and-the-tech-rout-nifty-it-index-down-23-in-2026-is-the-ai-correction-finally-over-4174063/', 'source': 'The Financial Express', 'published_at': '2026-03-15T18:30:00+00:00', 'summary': 'Nifty IT Index is down 23% in 2026, but experts see attractive valuations in top tech stocks—review your investment strategy ...'}

    ]

    sample_profile = {
        "company"         : "Infosys",
        "job"             : "AI/ML Engineer",
        "location"        : "Mysore",
        "news_exclusions" : ["gossips"],
        "personal_interests": ["cricket", "movies of other languages with subtitles"],
    }

    result = relevance_filter_node({
        "profile"          : sample_profile,
        "profile_chunks"   : [],
        "search_queries"   : [],
        "raw_articles"     : sample_articles,
        "filtered_articles": [],
        "scored_articles"  : [],
        "alert_articles"   : [],
        "errors"           : [],
    })

    print('='*25,'RESULTS','='*25)
    print(result)
    print('='*50)

    articles = result["filtered_articles"]
    print(f"\n{'═' * 60}")
    print(f"  FILTERED ARTICLES — {len(articles)} passed")
    print(f"{'═' * 60}\n")
    for art in articles:
        print(f"  Title      : {art['title'][:70]}")
        print(f"  Similarity : {art['similarity_score']:.4f}")
        print(f"  Penalty    : {art['feedback_penalty']}")
        print(f"  Final Score: {art['final_score']:.4f}")
        print(f"  Top Chunks : {[c['id'] for c in art['matching_profile_chunks']]}")
        print()