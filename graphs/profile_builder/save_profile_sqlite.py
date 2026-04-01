"""
save_profile_sqlite_node.py
----------------------------
Agentic node — LLM bound with SQLite tools only.
Saves all profile_fields into SQLite using upsert_profile.
"""

import json
from agent_utils import create_agent, invoke_agent
from mcp_tools import get_sqlite_tools
from state import CollectionState


SYSTEM_PROMPT = """
You are a database writer. Your job is to save profile fields into SQLite
using the upsert_profile tool.

Rules:
- Call upsert_profile once for EACH field — not all at once
- For list values (tech_stack, skills, interests), serialize them as JSON strings before saving
- Skip any fields that are empty strings or empty lists
- After saving all fields, confirm how many fields were saved
"""


def save_profile_sqlite_node(state: CollectionState) -> dict:
    fields = state.get("profile_fields", {})
    tools = get_sqlite_tools()
    agent, smart_llm = create_agent(tools, SYSTEM_PROMPT, mode="think")
    messages = {"messages": [
        {"role": "user", "content": f"Save these profile fields to SQLite:\n\n{json.dumps(fields, indent=2)}"}
    ]}
    invoke_agent(agent, smart_llm, tools, messages, SYSTEM_PROMPT)
    return {"errors": []}