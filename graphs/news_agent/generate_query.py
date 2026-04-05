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

#generate as many specific DuckDuckGo search
SYSTEM_PROMPT =  """
You are a search query generator for a personalised daily news system.

Given a person's profile, generate DuckDuckGo search queries by following
EXACTLY the templates below for EVERY field present in the profile.
Do not skip any field. Do not invent new categories.

TEMPLATES TO FOLLOW:

COMPANY (generate all 4):
- "[company] news {year}"
- "[company] layoffs {year}"
- "[company] AI strategy {year}"
- "[company] stock {year}"

JOB ROLE (generate all 3):
- "[job] demand India {year}"
- "[job] salary India {year}"
- "[job] future {year}"

INDUSTRY (generate 2):
- "[industry] market trends {year}"
- "[industry] AI disruption {year}"

LOCATION (generate 2):
- "[city] tech news {year}"
- "[city] AI jobs {year}"

SKILLS — for EACH skill (generate 2 per skill):
- "[skill] latest update {year}"
- "[skill] tutorial {year}"

TECH STACK — for EACH tech (generate 1 per tech, only if not already covered by skills):
- "[tech] new features {year}"

INTERESTS — for EACH interest (generate 2 per interest):
- "[interest] news {year}"
- "[interest] breakthrough {year}"

PERSONAL INTERESTS — for EACH personal interest apply smart expansion:
  - if cricket:
      "[tournament name] {year} schedule"
      "[tournament name] {year} results"
      "India cricket {year}"
      "cricket latest news {year}"
  - if movies:
      For EACH language in languages_spoken (generate 1 per language):
      "[language] movies {year} releases"
      "[language] OTT releases {year}"
  - for any other interest:
      "[interest] news {year}"
      "[interest] latest {year}"

CAREER GROWTH (generate these always):
- "[job] certifications {year}"
- "[job] conferences India {year}"
- "AI startup funding {year}"
- "AI jobs India {year}"

UNKNOWN FIELDS — for ANY field in the profile not covered by the templates above:
- Read the field name and its value
- Generate 2 relevant news queries based on what that field likely means
- Use the same pattern: "[value] news {year}", "[value] latest {year}"
- If the value is a list, generate 1 query per item
- If the value is a string, generate 2 queries from it
- Use common sense — "hobbies: chess" → "chess tournaments {year}", "chess AI {year}"

RULES:
- Replace {year} with the current year or current date provided according to the need
- Keep queries short — 3 to 6 words
- No duplicate queries
- No generic queries like "technology news" or "latest news"
- Every query must be specific and focused
- Output ONLY a valid JSON array of strings
- No markdown, no explanation, no backticks

Example output format:
["Infosys news 2026", "LangGraph latest update 2026", "IPL 2026 schedule", ...]


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
    sample_profile = {'profile': {'name': 'ANTONY VASLIN', 'location': 'Mysore', 'skills': ['Python', 'LangGraph', 'LangChain', 'RAG pipelines', 'Telegram bots', 'OCR', 'Raspberry Pi', 'Spiking Neural Networks', 'multi-agent systems', 'real-time inference'], 'goals': 'Building production-grade AI agents and RAG systems that interact naturally across voice, text and hardware.', 'company': 'Infosys', 'interests': ['conversational AI', 'assistive hardware', 'neuromorphic computing', 'autonomous agents', 'career-tech fusion'], 'tech_stack': ['Python', 'LangGraph', 'LangChain', 'RAG', 'Telegram API', 'OCR', 'Raspberry Pi', 'Spiking Neural Networks', 'FastAPI', 'Pandas'], 'professional_context': 'Early-career AI/ML Engineer at Infosys in Mysore who has spent the last 18 months rapidly prototyping intelligent systems—ranging from voice-enabled RAG chatbots and multi-agent debaters to Raspberry-Pi smart glasses and neuromorphic drowsiness detection—showcasing a clear focus on conversational AI, retrieval-augmented generation and edge inference.', 'github_username': 'vaslin-dotcom', 'job': 'AI/ML Engineer', 'industry': 'IT services', 'locality': 'infosys campus', 'daily_habits': 'morning tv news while stretching, push-ups, sit-ups; youtube during breakfast, lunch, dinner; 1 hr weight training with resistance band at night', 'news_reading_time': 'morning', 'lifestyle_context': 'system engineer trainee, trained in oracle 21c but not interested, building own projects', 'languages_spoken': ['tamil', 'english', 'malayalam', 'telugu'], 'news_exclusions': ['gossips'], 'personal_interests': ['cricket', 'movies of other languages with subtitles']}, 'profile_chunks': [{'id': 'vaslin-dotcom:interests', 'similarity': -0.271, 'document': 'Interests: conversational AI, assistive hardware, neuromorphic computing, autonomous agents, career-tech fusion', 'metadata': {'updated_at': '2026-04-02T07:59:50.775836', 'username': 'vaslin-dotcom', 'type': 'interests'}}, {'id': 'vaslin-dotcom:lifestyle', 'similarity': -0.4399, 'document': 'infosys campus, morning tv news while stretching, push-ups, sit-ups; youtube during breakfast, lunch, dinner; 1 hr weight training with resistance ban', 'metadata': {'type': 'lifestyle', 'username': 'vaslin-dotcom', 'updated_at': '2026-04-02T08:05:32.476852'}}, {'id': 'vaslin-dotcom:exclusions', 'similarity': -0.6443, 'document': 'Avoid these topics in news: gossips', 'metadata': {'username': 'vaslin-dotcom', 'updated_at': '2026-04-02T08:05:32.476852', 'type': 'exclusions'}}, {'id': 'vaslin-dotcom:professional', 'similarity': -0.6875, 'document': 'AI/ML Engineer with expertise in Python, LangGraph, LangChain, RAG pipelines, Telegram bots, OCR, Raspberry Pi, Spiking Neural Networks, multi-agent s', 'metadata': {'type': 'professional', 'username': 'vaslin-dotcom', 'updated_at': '2026-04-02T07:59:50.775836'}}, {'id': 'vaslin-dotcom:company', 'similarity': -0.7188, 'document': 'Infosys IT services Early-career AI/ML Engineer at Infosys in Mysore who has spent the last 18 months rapidly prototyping intelligent systems—ranging ', 'metadata': {'updated_at': '2026-04-02T07:59:50.775836', 'username': 'vaslin-dotcom', 'type': 'company'}}, {'id': 'vaslin-dotcom:goals', 'similarity': -0.7348, 'document': 'Building production-grade AI agents and RAG systems that interact naturally across voice, text and hardware', 'metadata': {'type': 'goals', 'updated_at': '2026-04-02T07:59:50.775836', 'username': 'vaslin-dotcom'}}, {'id': 'vaslin-dotcom:personal', 'similarity': -1.0194, 'document': 'cricket, movies of other languages with subtitles', 'metadata': {'type': 'personal', 'username': 'vaslin-dotcom', 'updated_at': '2026-04-02T08:05:32.476852'}}]}

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

    print('='*25,'RESULTS','='*25)
    print(result)
    print('='*50)

    queries = result["search_queries"]
    print(f"\n{'═' * 60}")
    print(f"  GENERATED QUERIES — {len(queries)} total")
    print(f"{'═' * 60}\n")
    for i, q in enumerate(queries, 1):
        print(f"  {i:2}. {q}")