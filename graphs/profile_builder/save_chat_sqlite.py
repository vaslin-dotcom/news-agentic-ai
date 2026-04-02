
#save_chat_sqlite
"""
save_chat_sqlite_node.py
-------------------------
Agentic node — LLM bound with SQLite tools only.
Saves all chat_fields into SQLite using upsert_profile.
"""

import json
from agent_utils import create_agent, invoke_agent
from mcp_tools import get_sqlite_tools
from state import CollectionState

SYSTEM_PROMPT = """
You are a database writer. Save personal profile fields into SQLite using upsert_profile.

Rules:
- Call upsert_profile once for EACH field — not all at once
- For list values (languages_spoken, personal_interests, news_exclusions),
  serialize them as JSON strings before saving
- For the "extra" dict, flatten each key with an "extra_" prefix
  e.g. extra = {"has_kids": true} → upsert_profile("extra_has_kids", "true")
- Skip any fields that are empty strings or empty lists
- Confirm total fields saved when done
"""


def save_chat_sqlite_node(state: CollectionState) -> dict:
    fields = state.get("chat_fields", {})
    tools = get_sqlite_tools()
    agent, smart_llm = create_agent(tools, SYSTEM_PROMPT, mode="think")
    messages = {"messages": [
        {"role": "user", "content": f"Save these personal profile fields to SQLite:\n\n{json.dumps(fields, indent=2)}"}
    ]}
    invoke_agent(agent, smart_llm, tools, messages, SYSTEM_PROMPT)
    return {"errors": []}