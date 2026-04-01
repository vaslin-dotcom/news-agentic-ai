"""
build_profile_node.py
---------------------
Pure LLM synthesis node — no tools needed.
Takes raw GitHub data from state and infers structured professional profile fields.
"""

import json
import re
from llm import get_llm
from state import CollectionState

SYSTEM_PROMPT = """
You are a profile builder. Given a user's raw GitHub data, extract and infer a structured professional profile.

Rules:
- Infer `job` from bio, company, and repo patterns (e.g. "ML Engineer", "Full Stack Developer")
- Infer `industry` from company name + repo topics (e.g. "fintech", "devtools", "AI/ML")
- Infer `skills` as a flat list of technologies actually used (from languages + topics + repo names)
- Infer `tech_stack` as the top 10 primary technologies only
- Infer `interests` as topic areas the person seems curious about beyond their day job
- Infer `goals` as a short sentence — what they seem to be building toward
- Write `professional_context` as one dense paragraph summarising who this person is professionally
- If a field cannot be inferred, use empty string or empty list — never hallucinate

Respond ONLY with valid JSON. No markdown, no explanation, no backticks.

JSON shape:
{
  "github_username": "",
  "name": "",
  "bio": "",
  "job": "",
  "company": "",
  "industry": "",
  "location": "",
  "tech_stack": [],
  "skills": [],
  "interests": [],
  "goals": "",
  "professional_context": ""
}
"""


def build_profile_node(state: CollectionState) -> dict:
    raw = state["github_raw"]
    llm = get_llm(mode="think")

    user_message = f"""
Here is the raw GitHub data for this user:

PROFILE:
{json.dumps(raw.get("profile", {}), indent=2)}

REPOS (top 30):
{json.dumps(raw.get("repos", []), indent=2)}

LANGUAGES:
{json.dumps(raw.get("languages", []), indent=2)}

TOPICS:
{json.dumps(raw.get("topics", []), indent=2)}

Build the structured profile now.
"""

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ])

    try:
        profile_fields = json.loads(response.content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response.content, re.DOTALL)
        profile_fields = json.loads(match.group()) if match else {}

    return {"profile_fields": profile_fields}