"""
generate_queries_node.py
------------------------
Pure LLM node — no tools needed.
Reads profile from state and generates as many DDG search queries
as needed to cover ALL dimensions of the person's life.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from datetime import datetime
import json

current_date = datetime.now().strftime("%B %d, %Y")
import re
from llm import get_llm
from state import NewsState


SYSTEM_PROMPT = """
You are a news query generator for a personalised news system.

Given a person's full profile, generate as many specific DuckDuckGo search
queries as needed to collect ALL current news relevant to every dimension
of their life.

Cover ALL of these dimensions — do not skip any:

PROFESSIONAL
- Their company (news, announcements, layoffs, growth, strategy)
- Their job role (trends, demand, salary, future)
- Their industry (market shifts, regulations, disruptions)
- Their location + company (local office news, city tech scene)

TECHNICAL INTERESTS
- Each technology in their tech stack (updates, releases, new tools)
- Each area of technical interest (research, breakthroughs, tutorials)


PERSONAL INTERESTS
- Each personal interest (sports teams, leagues, tournaments, results)
- Entertainment (cinema, music, culture relevant to their languages)
- Local region news (city, state, local events)

CAREER GROWTH
- Job market in their field and location
- Certifications, courses, conferences relevant to their skills
- Startup ecosystem in their domain

RULES:
- Make every query specific — no generic queries like "technology news"
- Use current year in queries where recency matters
- Each query should return FOCUSED results on one topic
- Do NOT limit yourself — generate as many as needed
- Queries should be short (3-6 words) for best DDG results

Respond ONLY with a valid JSON array of strings.
No markdown, no explanation, no backticks.

Example format:
["Infosys layoffs 2026", "LangGraph new features", "IPL 2026 schedule", ...]
"""


def generate_queries_node(state: NewsState) -> dict:
    profile = state["profile"]
    llm     = get_llm(mode="think")

    user_message = f"""
Generate all search queries for this person:

{json.dumps(profile, indent=2, default=str)}

Today's date: {current_date}.
Generate queries covering every dimension of their life.
"""

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ])

    content = response.content.strip()

    # strip markdown code fences if LLM adds them
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "",        content)

    try:
        queries = json.loads(content)
        if not isinstance(queries, list):
            queries = []
    except json.JSONDecodeError:
        # fallback: extract anything that looks like a JSON array
        match = re.search(r"\[.*\]", content, re.DOTALL)
        queries = json.loads(match.group()) if match else []

    return {"search_queries": queries}


# ── Test block ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # reuse the profile from Node 1 output
    print("Generating......")
    sample_profile = {
        "name": "ANTONY VASLIN",
        "job": "AI/ML Engineer",
        "company": "Infosys",
        "industry": "IT services",
        "location": "Mysore",
        "locality": "infosys campus",
        "tech_stack": ["Python", "LangGraph", "LangChain", "RAG", "Telegram API",
                       "OCR", "Raspberry Pi", "Spiking Neural Networks", "FastAPI", "Pandas"],
        "skills": ["Python", "LangGraph", "LangChain", "RAG pipelines", "Telegram bots",
                   "OCR", "Raspberry Pi", "Spiking Neural Networks", "multi-agent systems",
                   "real-time inference"],
        "interests": ["conversational AI", "assistive hardware", "neuromorphic computing",
                      "autonomous agents", "career-tech fusion"],
        "personal_interests": ["cricket", "movies of other languages with subtitles"],
        "languages_spoken": ["tamil", "english", "malayalam", "telugu"],
        "goals": "Building production-grade AI agents and RAG systems that interact naturally across voice, text and hardware.",
        "lifestyle_context": "system engineer trainee, trained in oracle 21c but not interested, building own projects",
        "news_exclusions": ["gossips"],
        "news_reading_time": "morning",
    }

    result = generate_queries_node({
        "profile"          : sample_profile,
        "profile_chunks"   : [],
        "search_queries"   : [],
        "raw_articles"     : [],
        "filtered_articles": [],
        "scored_articles"  : [],
        "alert_articles"   : [],
        "errors"           : [],
    })

    queries = result["search_queries"]
    print(f"\n{'═' * 60}")
    print(f"  GENERATED QUERIES — {len(queries)} total")
    print(f"{'═' * 60}\n")
    for i, q in enumerate(queries, 1):
        print(f"  {i:2}. {q}")