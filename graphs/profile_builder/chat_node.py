"""
chat_node.py
------------
LLM-driven terminal interviewer.
100% MCP based — reads profile via get_profile tool.

Behaviour:
  - If profile is INCOMPLETE → asks only missing fields
  - If profile is COMPLETE   → shows existing profile, asks if anything changed
    User can say "no changes" to exit immediately, or describe changes
    which get merged/overwritten in state

Loop exits when LLM emits PROFILE_COMPLETE in its response.
"""

import asyncio
import json
import re
import sys
from llm import get_llm
from mcp_tools import get_tool
from state import CollectionState

# ---------------------------------------------------------------------------
# Key classification
# ---------------------------------------------------------------------------

GITHUB_KEYS = {
    "name", "github_username", "bio", "company", "location",
    "job", "industry", "skills", "tech_stack",
    "interests", "professional_context", "goals",
}

CHAT_KEYS = {
    "locality", "languages_spoken", "personal_interests",
    "daily_habits", "lifestyle_context", "news_reading_time",
    "news_exclusions",
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Used when some fields are still missing
INTERVIEWER_PROMPT = """
You are building a personal news relevance profile for a user.

ALREADY_KNOWN is shown below — do NOT ask about any of it again.
MISSING_FIELDS lists what still needs to be collected.

What to collect (only what's in MISSING_FIELDS):
- locality          : city, neighbourhood, region they live in
- languages_spoken  : human languages they speak
- personal_interests: hobbies, sports, food, cinema, music — outside of tech
- daily_habits      : morning reader? commuter? night owl?
- lifestyle_context : student / employee / freelancer / family person?
- news_reading_time : when they prefer to read news
- news_exclusions   : topics they actively want AVOIDED in their feed

Rules:
- Ask ONE question at a time — never multiple in one message
- Build naturally on their answers
- Keep questions short and conversational
- When all missing fields are covered, end your final message with:
  PROFILE_COMPLETE
"""

# Used when all fields are already filled — runs an update check
UPDATE_PROMPT = """
You are maintaining a personal news relevance profile for a user.
Their existing profile is shown below under CURRENT_PROFILE.

Your job:
1. Greet them briefly and show a short human-readable summary of their profile
   (not a JSON dump — write it naturally, like "You're based in Chennai, speak Tamil and English...")
2. Ask in a friendly way: has anything changed since we last spoke?
   (e.g. moved cities, new hobbies, topics to add/remove from news feed)
3. If they say nothing has changed (e.g. "no", "all good", "nothing"), 
   acknowledge it and end with PROFILE_COMPLETE immediately
4. If they describe changes, ask follow-up questions ONE at a time to clarify,
   then end with PROFILE_COMPLETE

Rules:
- Keep it conversational and brief — this is a check-in, not a full interview
- Never re-ask things they already confirmed are unchanged
- End your final message with PROFILE_COMPLETE on its own line
"""

EXTRACTION_PROMPT = """
Given a conversation where a user described updates to their personal profile,
extract only the fields that CHANGED or were newly mentioned.

Respond ONLY with valid JSON. No markdown, no backticks, no explanation.

JSON shape:
{
  "locality": "",
  "languages_spoken": [],
  "personal_interests": [],
  "daily_habits": "",
  "lifestyle_context": "",
  "news_reading_time": "",
  "news_exclusions": [],
  "extra": {}
}

- Use empty string or empty list for fields NOT mentioned or NOT changed
- Only populate fields where the user explicitly described something new or different
- Never hallucinate — if not mentioned, leave it empty
- If the user said "no changes", return all empty fields
"""

# ---------------------------------------------------------------------------
# MCP helpers
# ---------------------------------------------------------------------------

def _try_parse(value: str):
    if isinstance(value, str) and (value.startswith("[") or value.startswith("{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def _parse_tool_response(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        for block in raw:
            if isinstance(block, dict) and "text" in block:
                raw = block["text"]
                break
        else:
            return {}
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


async def _ainvoke_get_profile() -> dict:
    tool = get_tool("get_profile")
    raw = await tool.ainvoke({})
    return _parse_tool_response(raw)


def _load_profile_via_mcp() -> tuple[dict, dict]:
    flat = asyncio.run(_ainvoke_get_profile())

    github_fields: dict = {}
    chat_fields: dict   = {}
    extra: dict         = {}

    for key, value in flat.items():
        parsed = _try_parse(value)
        if key in GITHUB_KEYS:
            github_fields[key] = parsed
        elif key in CHAT_KEYS:
            chat_fields[key] = parsed
        elif key.startswith("extra_"):
            sub_key = key[len("extra_"):]
            extra[sub_key] = _try_parse(value)

    if extra:
        chat_fields["extra"] = extra

    return github_fields, chat_fields


def _missing_fields(chat_fields: dict) -> list[str]:
    return [
        key for key in CHAT_KEYS
        if not chat_fields.get(key) or chat_fields.get(key) in ("", [], {})
    ]

# ---------------------------------------------------------------------------
# System prompt builders
# ---------------------------------------------------------------------------

def _build_interviewer_system(github_fields: dict, chat_fields: dict, missing: list[str]) -> str:
    context = {
        "ALREADY_KNOWN": {
            "github_profile":   github_fields,
            "personal_profile": chat_fields,
        },
        "MISSING_FIELDS": missing,
    }
    return f"{INTERVIEWER_PROMPT}\n\n{json.dumps(context, indent=2)}"


def _build_update_system(github_fields: dict, chat_fields: dict) -> str:
    context = {
        "CURRENT_PROFILE": {
            "github_profile":   github_fields,
            "personal_profile": chat_fields,
        }
    }
    return f"{UPDATE_PROMPT}\n\n{json.dumps(context, indent=2)}"

# ---------------------------------------------------------------------------
# Shared chat loop
# ---------------------------------------------------------------------------

def _run_chat_loop(llm, system_message: str, chat_history: list) -> list:
    """Runs the conversation loop until LLM emits PROFILE_COMPLETE."""
    while True:
        messages = [{"role": "system", "content": system_message}] + chat_history
        response  = llm.invoke(messages)
        assistant = response.content

        profile_complete = "PROFILE_COMPLETE" in assistant
        display          = assistant.replace("PROFILE_COMPLETE", "").strip()

        print(f"\nAI: {display}\n")
        chat_history.append({"role": "assistant", "content": display})

        if profile_complete:
            break

        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Chat interrupted]")
            break

        if user_input:
            chat_history.append({"role": "user", "content": user_input})

    return chat_history

# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def chat_node(state: CollectionState) -> dict:
    # ── 1. Load existing profile via MCP ────────────────────────────────────
    github_fields, existing_chat_fields = _load_profile_via_mcp()

    # Merge anything already in state from this graph run
    state_chat = state.get("chat_fields") or {}
    for k, v in state_chat.items():
        if v and v not in ("", [], {}):
            existing_chat_fields[k] = v

    missing  = _missing_fields(existing_chat_fields)
    llm      = get_llm(mode="generation")
    chat_history = list(state.get("chat_history", []))

    # ── 2. Choose mode: first-time interview OR update check ─────────────────
    if missing:
        # Some fields still empty — run the interviewer
        print("\n" + "═" * 60)
        print("  Profile Builder — Personal Info Collection")
        known = [k for k in existing_chat_fields if k != "extra"]
        print(f"  Already known : {', '.join(known) if known else '(none)'}")
        print(f"  Still needed  : {', '.join(missing)}")
        print("═" * 60)

        system_message = _build_interviewer_system(github_fields, existing_chat_fields, missing)

    else:
        # Profile complete — run the update check
        print("\n" + "═" * 60)
        print("  Profile Builder — Update Check")
        print("  (Profile already complete — checking for changes)")
        print("═" * 60)

        system_message = _build_update_system(github_fields, existing_chat_fields)

    # ── 3. Run the conversation ──────────────────────────────────────────────
    chat_history = _run_chat_loop(llm, system_message, chat_history)

    print("\n" + "═" * 60)
    print("  Done. Extracting any changes...")
    print("═" * 60 + "\n")

    # ── 4. Extract fields from conversation ──────────────────────────────────
    conversation_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in chat_history
    )
    extraction = llm.invoke([
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user",   "content": f"CONVERSATION:\n{conversation_text}"},
    ])

    try:
        new_fields = json.loads(extraction.content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", extraction.content, re.DOTALL)
        new_fields = json.loads(match.group()) if match else {}

    # ── 5. Merge: new non-empty answers overwrite old ones ───────────────────
    final_fields: dict = dict(existing_chat_fields)

    for key in CHAT_KEYS:
        new_val = new_fields.get(key)
        if new_val and new_val not in ("", [], {}):
            final_fields[key] = new_val   # new answer overwrites

    existing_extra = existing_chat_fields.get("extra") or {}
    new_extra      = new_fields.get("extra") or {}
    if new_extra:
        final_fields["extra"] = {**existing_extra, **new_extra}

    return {
        "chat_fields":      final_fields,
        "chat_history":     chat_history,
        "profile_complete": True,
    }

# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  chat_node.py — standalone test")
    print("═" * 60)

    print("\n[1] Loading profile from SQLite via MCP...\n")

    try:
        github_fields, chat_fields = _load_profile_via_mcp()
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    print("GitHub fields:")
    for k, v in github_fields.items():
        print(f"  {k}: {v}")

    print("\nChat fields:")
    if chat_fields:
        for k, v in chat_fields.items():
            print(f"  {k}: {v}")
    else:
        print("  (none)")

    missing = _missing_fields(chat_fields)
    print(f"\nMissing : {missing if missing else '(none — will run update check)'}")
    print(f"Mode    : {'INTERVIEW' if missing else 'UPDATE CHECK'}")

    if "--chat" in sys.argv:
        print("\n[2] Starting chat...\n")
        dummy_state = {
            "username":       github_fields.get("github_username", "unknown"),
            "profile_fields": github_fields,
            "chat_history":   [],
            "chat_fields":    chat_fields,
        }
        result = chat_node(dummy_state)
        print("\n[3] Final chat_fields:")
        print(json.dumps(result.get("chat_fields", {}), indent=2, default=str))
    else:
        print("\nRun with --chat to start:")
        print("  python graphs/profile_builder/chat_node.py --chat")