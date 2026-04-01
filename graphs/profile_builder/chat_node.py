"""
chat_node.py
------------
LLM-driven terminal interviewer.
No tools needed — pure conversation loop.
Reads the already-built profile, identifies gaps GitHub can't fill,
and asks one targeted question at a time.

Loop exits when LLM emits PROFILE_COMPLETE anywhere in its response.
Full conversation is then structured into chat_fields by a second LLM call.
"""

import json
import re
from llm import get_llm
from state import CollectionState

INTERVIEWER_PROMPT = """
You are building a personal news relevance profile for a user.
You already know their professional background from GitHub (shown below).
Your job is to learn everything about them that GitHub cannot tell you.

What you need to uncover (be natural — don't follow a rigid order):
- Where they live (city, neighbourhood, region)
- Human languages they speak
- Personal interests outside of tech (sports, food, cinema, music, hobbies)
- Daily habits and routines (morning reader? commuter? night owl?)
- Lifestyle context (student, employee, freelancer, family person?)
- When they prefer to read news
- Topics they actively want to AVOID in their news feed
- Anything else useful for personalising news relevance

Rules:
- Ask ONE question at a time — never multiple in one message
- Build naturally on their answers — don't follow a script
- Never ask about things already known from GitHub (shown below)
- Keep questions short and conversational
- When you feel you have a complete picture of this person's life context,
  end your final message with the exact token on its own line:
  PROFILE_COMPLETE
"""

EXTRACTION_PROMPT = """
Given a conversation where a user described themselves personally,
extract structured profile fields.

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

- Use empty string or empty list for fields not mentioned — never hallucinate
- Put anything useful that doesn't fit above into "extra" as key-value pairs
"""


def _build_system(profile_fields: dict) -> str:
    return f"{INTERVIEWER_PROMPT}\n\nALREADY KNOWN FROM GITHUB:\n{json.dumps(profile_fields, indent=2)}"


def chat_node(state: CollectionState) -> dict:
    profile_fields = state.get("profile_fields", {})
    chat_history   = list(state.get("chat_history", []))
    llm            = get_llm(mode="generation")
    system_message = _build_system(profile_fields)

    print("\n" + "═" * 60)
    print("  Profile Builder — Personal Info Collection")
    print("  (Chat naturally. The AI will ask what it needs to know.)")
    print("═" * 60 + "\n")

    while True:
        messages = [{"role": "system", "content": system_message}] + chat_history

        response        = llm.invoke(messages)
        assistant_msg   = response.content
        profile_complete = "PROFILE_COMPLETE" in assistant_msg
        display_msg     = assistant_msg.replace("PROFILE_COMPLETE", "").strip()

        print(f"AI: {display_msg}\n")
        chat_history.append({"role": "assistant", "content": display_msg})

        if profile_complete:
            break

        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Chat interrupted]")
            break

        if user_input:
            chat_history.append({"role": "user", "content": user_input})

    print("\n" + "═" * 60)
    print("  Done. Saving your profile...")
    print("═" * 60 + "\n")

    # Extract structured fields from full conversation
    conversation_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in chat_history
    )
    extraction_response = llm.invoke([
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user",   "content": f"CONVERSATION:\n{conversation_text}"},
    ])

    try:
        chat_fields = json.loads(extraction_response.content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", extraction_response.content, re.DOTALL)
        chat_fields = json.loads(match.group()) if match else {}

    return {
        "chat_fields":      chat_fields,
        "chat_history":     chat_history,
        "profile_complete": True,
    }