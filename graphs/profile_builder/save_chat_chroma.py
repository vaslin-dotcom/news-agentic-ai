"""
embed_chat_chroma_node.py
--------------------------
Agentic node — LLM bound with Chroma tools only.
LLM chunks chat-collected personal fields and upserts into profile_vectors.

Chunk types: personal | lifestyle | exclusions | extra
"""

import json
from datetime import datetime
from agent_utils import create_agent, invoke_agent
from mcp_tools import run_with_chroma_tools
from state import CollectionState
import asyncio

SYSTEM_PROMPT = """
You are a vector database writer. Your job is to embed personal profile chunks
into Chroma using the upsert_embedding tool.

Create and save EXACTLY these chunks from the fields provided:

1. ID: "{username}:personal"
   Document: combine personal_interests + languages_spoken
   Metadata: {"type": "personal", "username": "{username}", "updated_at": "{now}"}

2. ID: "{username}:lifestyle"
   Document: combine locality + daily_habits + lifestyle_context + news_reading_time
   Metadata: {"type": "lifestyle", "username": "{username}", "updated_at": "{now}"}

3. ID: "{username}:exclusions"
   Document: "Avoid these topics in news: " + comma-joined news_exclusions list
   Metadata: {"type": "exclusions", "username": "{username}", "updated_at": "{now}"}

4. ID: "{username}:extra"  (only if extra dict is non-empty)
   Document: all extra key-value pairs joined as "key: value" sentences
   Metadata: {"type": "extra", "username": "{username}", "updated_at": "{now}"}

Rules:
- Skip any chunk whose document would be empty
- Collection name is always: profile_vectors
- Do NOT pass an embedding field — the tool handles embedding internally
"""


def embed_chat_chroma_node(state: CollectionState) -> dict:
    username = state["username"]
    fields   = state.get("chat_fields", {})
    tools    = asyncio.run(run_with_chroma_tools())
    agent, smart_llm = create_agent(tools, SYSTEM_PROMPT, mode="think")

    now    = datetime.utcnow().isoformat()
    prompt = (
        f"Username: {username}\n"
        f"Now (UTC): {now}\n\n"
        f"Chat fields:\n{json.dumps(fields, indent=2)}\n\n"
        f"Embed all non-empty chunks into profile_vectors now."
    )

    messages = {"messages": [{"role": "user", "content": prompt}]}
    invoke_agent(agent, smart_llm, tools, messages, SYSTEM_PROMPT, mode="think")

    saved_ids = [f"{username}:{t}" for t in ["personal", "lifestyle", "exclusions"]]
    if fields.get("extra"):
        saved_ids.append(f"{username}:extra")

    return {"chroma_ids": saved_ids, "errors": []}