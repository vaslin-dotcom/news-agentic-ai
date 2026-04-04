"""
deep_reasoning_node.py
----------------------
Deep reasoning node using Sequential Thinking MCP.
Parallel processing with semaphore — max 3 articles concurrently.
For each article:
  1. Fetch full article content directly from URL
  2. Run sequential thinking chain to assess personal impact
  3. Output urgency (1-5), relevance_score, reasoning, connected_to
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import asyncio
import re
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from llm import get_llm
from state import NewsState

MCP_SERVERS = {
    "sequential-thinking": {
        "command"  : "npx",
        "args"     : ["-y", "@modelcontextprotocol/server-sequential-thinking"],
        "transport": "stdio",
    },
    "chroma": {
        "url"      : "http://localhost:8001/mcp/",
        "transport": "streamable_http"
    },
    "sqlite": {
        "command"  : "python",
        "args"     : ["../../mcp/sqlite.py"],
        "transport": "stdio"
    },
    "ddg": {
        "command"  : "python",
        "args"     : ["../../mcp/ddg.py"],
        "transport": "stdio"
    }
}

CONCURRENT_LIMIT = 3  # max articles reasoned simultaneously

SYSTEM_PROMPT = """
You are a personal news analyst. Your job is to deeply assess how a news article
affects a SPECIFIC person — not people in general.

You have access to three tools:
- sequentialthinking : use for EVERY reasoning step
- query_similar      : search Chroma for past connected articles
- execute_query      : fetch article details from SQLite

STRICT RULES:
- Use sequentialthinking for EVERY thought — never reason in plain text
- Always refer to the person by name in your reasoning
- Be specific — mention their job title, company, location
- Minimum 6 thoughts, more if needed
- Set nextThoughtNeeded=False only on the final thought

THINKING SEQUENCE:

Thought 1 — Article Facts
  What is this article concretely about?
  Key facts only — no assumptions.

Thought 2 — Profile Mapping
  Which specific parts of this person's profile does this article touch?
  Use ONLY the profile chunks provided — these are the relevant dimensions.
  Job? Company? Location? Tech stack? Personal interests?

Thought 3 — Personal Specificity
  How does this person's SPECIFIC context make this more or less relevant?
  Would this affect them differently than the average person?
  Be concrete — name their role, company, city.

Thought 4 — Past Article Connections
  Use query_similar to search article_vectors for past articles related to this one.
  If results found, use execute_query to fetch title, summary, urgency, reasoning.
  Do past articles combine with this to create a bigger picture?
  Is the combined urgency higher than this article alone?

Thought 5 — Feedback Awareness
  Were any similar articles previously skipped or marked irrelevant?
  Check similar_feedback_articles in the context.
  Does that lower urgency? Or is this article different enough?

Thought 6 — Real-World Impact
  What is the actual real-world impact on THIS person?
  Not hypothetical — concrete and personal.
  Would they need to DO something after reading this?

Thought 7 — Revision Check
  Look back at thoughts 3, 4, 5, 6.
  Did you miss anything? Were any assumptions wrong?
  Revise if needed (set isRevision=True).

Thought 8 — Final Verdict
  Summarize into the required JSON fields.
  Set nextThoughtNeeded=False here.

URGENCY SCALE:
  1 → mildly interesting, no action needed
  2 → worth knowing, informational only
  3 → notable — worth an alert
  4 → directly affects their work or life — alert needed
  5 → act on this now — urgent alert

OUTPUT FORMAT (after all thinking is complete):
Return ONLY this JSON — no markdown, no explanation, no backticks:
{
  "relevance_score": 0.0,
  "urgency": 1,
  "reasoning": "",
  "connected_to": []
}
"""


def _unwrap(raw) -> str:
    item = raw[0] if isinstance(raw, list) else raw
    return item["text"] if isinstance(item, dict) else item.text


def _build_prompt(
    article      : dict,
    profile      : dict,
    full_content : str,
) -> str:
    """
    Build focused user message.
    Uses only matching_profile_chunks (top 3) — not all 7.
    This keeps context tight and relevant per article.
    """
    # only the chunks that actually matched this article in Node 4
    matching_chunks = article.get("matching_profile_chunks", [])
    chunks_text = "\n".join([
        f"  [{c.get('metadata', {}).get('type', '?')}]: {c.get('document', '')}"
        for c in matching_chunks
    ])

    # feedback context
    feedback     = article.get("similar_feedback_articles", [])
    feedback_text = ""
    if feedback:
        feedback_text = "\nPREVIOUSLY SKIPPED SIMILAR ARTICLES:\n" + "\n".join([
            f"  - {f.get('document', '')[:100]} "
            f"(signal: {f.get('metadata', {}).get('signal', '?')})"
            for f in feedback
        ])

    return f"""
PERSON:
  Name     : {profile.get('name', 'Unknown')}
  Job      : {profile.get('job', '')}
  Company  : {profile.get('company', '')}
  Location : {profile.get('location', '')} / {profile.get('locality', '')}
  Industry : {profile.get('industry', '')}
  Goals    : {profile.get('goals', '')}

RELEVANT PROFILE DIMENSIONS (matched by similarity):
{chunks_text}

ARTICLE:
  Title     : {article.get('title', '')}
  Source    : {article.get('source', '')}
  Published : {article.get('published_at', '')}
  Summary   : {article.get('summary', '')}
  Content   : {full_content[:4000] if full_content else 'Not available'}
  URL       : {article.get('url', '')}

PRE-FILTER:
  Similarity : {article.get('similarity_score', 0):.4f}
  Penalty    : {article.get('feedback_penalty', 0)}
  Matched    : {[c.get('id') for c in matching_chunks]}
{feedback_text}

Analyze this article using sequential thinking.
After all thoughts are complete, return ONLY the JSON verdict.
"""


def _extract_verdict(content: str) -> dict:
    """Extract final JSON verdict from agent response."""
    content = re.sub(r"```(?:json)?\s*", "", content)
    content = re.sub(r"```", "", content)

    match = re.search(r"\{[^{}]*relevance_score[^{}]*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {
        "relevance_score": 0.3,
        "urgency"        : 1,
        "reasoning"      : "Could not parse reasoning.",
        "connected_to"   : []
    }

async def _reason_article(
    semaphore  : asyncio.Semaphore,
    agent,
    tools_dict : dict,
    article    : dict,
    profile    : dict,
    index      : int,
    total      : int,
) -> dict:
    """
    Reason about a single article.
    Semaphore limits max concurrent executions to CONCURRENT_LIMIT.
    """
    async with semaphore:
        title = article.get("title", "")[:60]
        print(f"\n  [{index}/{total}] START  {title}")

        # ── Step 1: fetch full content from URL ───────────────
        full_content = ""
        try:
            raw          = await tools_dict["fetch_article_content"].ainvoke({
                "url": article.get("url", "")
            })
            full_content = _unwrap(raw)
            print(f"    [{index}] ✓ Content: {len(full_content)} chars")
        except Exception as e:
            print(f"    [{index}] ⚠ Content fetch failed: {e}")

        # ── Step 2: sequential thinking ───────────────────────
        prompt = _build_prompt(article, profile, full_content)

        try:
            result   = await agent.ainvoke({
                "messages": [HumanMessage(content=prompt)]
            })
            response = result["messages"][-1].content
            verdict  = _extract_verdict(response)
        except Exception as e:
            print(f"    [{index}] ⚠ Reasoning failed: {e}")
            verdict = {
                "relevance_score": 0.3,
                "urgency"        : 1,
                "reasoning"      : f"Reasoning failed: {str(e)[:100]}",
                "connected_to"   : []
            }

        print(f"    [{index}] DONE  urgency={verdict.get('urgency')}/5  "
              f"score={verdict.get('relevance_score')}  "
              f"→ {verdict.get('reasoning', '')[:60]}")

        return {
            **article,
            "full_content"   : full_content[:500],
            "relevance_score": verdict.get("relevance_score", 0.3),
            "urgency"        : verdict.get("urgency", 1),
            "reasoning"      : verdict.get("reasoning", ""),
            "connected_to"   : verdict.get("connected_to", []),
        }


async def _reason(state: NewsState) -> dict:
    articles = state["filtered_articles"]
    profile  = state["profile"]

    client     = MultiServerMCPClient(MCP_SERVERS)
    tools      = await client.get_tools()
    tools_dict = {t.name: t for t in tools}

    smart_llm = get_llm(mode="think")
    agent     = create_react_agent(
        smart_llm.primary.bind_tools(tools), tools, prompt=SYSTEM_PROMPT
    )

    print(f"\n  Deep reasoning on {len(articles)} articles "
          f"(max {CONCURRENT_LIMIT} concurrent)...")

    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)

    # launch all articles concurrently — semaphore caps active ones at 3
    tasks = [
        _reason_article(
            semaphore, agent, tools_dict,
            article, profile, i, len(articles)
        )
        for i, article in enumerate(articles, 1)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # collect results — preserve original article on failure
    scored_articles = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"  ⚠ Article {i+1} failed: {result}")
            scored_articles.append({
                **articles[i],
                "full_content"   : "",
                "relevance_score": 0.3,
                "urgency"        : 1,
                "reasoning"      : "Processing failed.",
                "connected_to"   : [],
            })
        else:
            scored_articles.append(result)

    urgent = sum(1 for a in scored_articles if a.get("urgency", 1) >= 3)
    print(f"\n  Scored   : {len(scored_articles)}")
    print(f"  Urgent   : {urgent} (urgency >= 3)")

    return {"scored_articles": scored_articles}



def deep_reasoning_node(state: NewsState) -> dict:
    return asyncio.run(_reason(state))


# ── Test block ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_filtered = [
        {
            "title"       : "Tech Layoffs Surge While AI Jobs Soar: Key Trends Shaping the 2026 Tech Industry",
            "url"         : "https://www.techtimes.com/articles/315282/20260321",
            "summary"     : "Tech layoffs exceed 45,000 globally in early 2026, driven by AI restructuring. Companies prefer experienced candidates.",
            "source"      : "Tech Times",
            "published_at": "2026-03-21T11:23:00+00:00",
            "similarity_score"        : -0.294,
            "feedback_penalty"        : 0.0,
            "final_score"             : -0.294,
            "matching_profile_chunks" : [
                {
                    "id"      : "vaslin-dotcom:professional",
                    "document": "AI/ML Engineer with expertise in Python LangGraph LangChain",
                    "metadata": {"type": "professional"}
                },
                {
                    "id"      : "vaslin-dotcom:company",
                    "document": "Infosys IT services Early-career AI/ML Engineer at Infosys Mysore",
                    "metadata": {"type": "company"}
                },
            ],
            "similar_feedback_articles": [],
        },
        {
            "title"       : "IPL 2026 schedule this week: Full list of matches, dates, timings",
            "url"         : "https://sports.yahoo.com/articles/ipl-2026-schedule-week",
            "summary"     : "IPL 2026 kicked off with Virat Kohli leading Royal Challengers Bengaluru to a win.",
            "source"      : "Yahoo Sports",
            "published_at": "2026-03-31T08:12:46+00:00",
            "similarity_score"        : -0.475,
            "feedback_penalty"        : 0.0,
            "final_score"             : -0.475,
            "matching_profile_chunks" : [
                {
                    "id"      : "vaslin-dotcom:personal",
                    "document": "cricket, movies of other languages with subtitles",
                    "metadata": {"type": "personal"}
                },
            ],
            "similar_feedback_articles": [],
        },
    ]

    sample_profile = {
        "name"    : "ANTONY VASLIN",
        "job"     : "AI/ML Engineer",
        "company" : "Infosys",
        "location": "Mysore",
        "locality": "infosys campus",
        "industry": "IT services",
        "goals"   : "Building production-grade AI agents and RAG systems.",
    }

    result = deep_reasoning_node({
        "profile"          : sample_profile,
        "profile_chunks"   : [],
        "search_queries"   : [],
        "raw_articles"     : [],
        "filtered_articles": sample_filtered,
        "scored_articles"  : [],
        "alert_articles"   : [],
        "errors"           : [],
    })

    articles = result["scored_articles"]
    print(f"\n{'═' * 60}")
    print(f"  SCORED ARTICLES — {len(articles)} total")
    print(f"{'═' * 60}\n")
    for art in articles:
        print(f"  Title        : {art['title'][:70]}")
        print(f"  Urgency      : {art['urgency']}/5")
        print(f"  Relevance    : {art['relevance_score']}")
        print(f"  Reasoning    : {art['reasoning']}")
        print(f"  Connected to : {art['connected_to']}")
        print()