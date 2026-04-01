"""
embed_profile_chroma_node.py
-----------------------------
Agentic node — LLM bound with Chroma tools only.
LLM chunks profile fields and upserts them into profile_vectors.

Chunk types: professional | company | goals | interests
"""

import json
from datetime import datetime
from agent_utils import create_agent, invoke_agent
from mcp_tools import get_chroma_tools
from state import CollectionState

SYSTEM_PROMPT = """
You are a vector database writer. Your job is to embed professional profile chunks
into Chroma using the upsert_embedding tool.

Create and save EXACTLY these chunks from the profile fields provided:

1. ID: "{username}:professional"
   Document: combine job + bio + skills + tech_stack into one descriptive sentence
   Metadata: {"type": "professional", "username": "{username}", "updated_at": "{now}"}

2. ID: "{username}:company"
   Document: combine company + industry + professional_context
   Metadata: {"type": "company", "username": "{username}", "updated_at": "{now}"}

3. ID: "{username}:goals"
   Document: the goals field as-is
   Metadata: {"type": "goals", "username": "{username}", "updated_at": "{now}"}

4. ID: "{username}:interests"
   Document: "Interests: " + comma-joined interests list
   Metadata: {"type": "interests", "username": "{username}", "updated_at": "{now}"}

Rules:
- Skip any chunk whose document would be empty
- Collection name is always: profile_vectors
- Do NOT pass an embedding field — the tool handles embedding internally
"""

def embed_profile_chroma_node(state: CollectionState) -> dict:
    username = state["username"]
    fields = state.get("profile_fields", {})
    now = datetime.utcnow().isoformat()
    tools = get_chroma_tools()
    agent, smart_llm = create_agent(tools, SYSTEM_PROMPT, mode="think")
    prompt = (
        f"Username: {username}\nNow (UTC): {now}\n\n"
        f"Profile fields:\n{json.dumps(fields, indent=2)}\n\n"
        f"Embed all non-empty chunks into profile_vectors now."
    )
    messages = {"messages": [{"role": "user", "content": prompt}]}
    invoke_agent(agent, smart_llm, tools, messages, SYSTEM_PROMPT)
    saved_ids = [f"{username}:{t}" for t in ["professional", "company", "goals", "interests"]]
    return {"chroma_ids": saved_ids, "errors": []}