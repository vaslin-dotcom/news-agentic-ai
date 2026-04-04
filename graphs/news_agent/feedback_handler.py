"""
feedback_handler.py
--------------------
Always-on background process that listens for Telegram button taps
and writes feedback signals to SQLite + Chroma.

Compatible with langchain-mcp-adapters 0.1.0+
Uses client.session(server_name) per call.

Run:
  python feedback_handler.py
"""

import asyncio
import json
import re
import traceback
from datetime import datetime

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# ---------------------------------------------------------------------------
# MCP servers
# ---------------------------------------------------------------------------

MCP_SERVERS = {
    "telegram": {
        "command":   "python",
        "args":      ["../../mcp/telegram.py"],
        "transport": "stdio",
    },
    "sqlite": {
        "command":   "python",
        "args":      ["../../mcp/sqlite.py"],
        "transport": "stdio",
    },
    "chroma": {
        "url"      : "http://localhost:8001/mcp/",
        "transport": "streamable_http"  # ← change transport too
    },
}

POLL_INTERVAL = 30  # seconds

client = MultiServerMCPClient(MCP_SERVERS)

# ---------------------------------------------------------------------------
# Exception group unwrapper — Python 3.11 uses ExceptionGroups
# ---------------------------------------------------------------------------

def _unwrap_exception(e: BaseException) -> str:
    """Recursively unwrap ExceptionGroup to get the real error message."""
    if isinstance(e, BaseExceptionGroup):
        msgs = [_unwrap_exception(sub) for sub in e.exceptions]
        return " | ".join(msgs)
    return f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Helper: call a single tool on a named server
# ---------------------------------------------------------------------------

async def _call(server: str, tool_name: str, args: dict):
    async with client.session(server) as session:
        tools    = await load_mcp_tools(session)
        tool_map = {t.name: t for t in tools}
        if tool_name not in tool_map:
            raise ValueError(f"Tool '{tool_name}' not found on '{server}'. "
                             f"Available: {list(tool_map.keys())}")
        raw = await tool_map[tool_name].ainvoke(args)
        return _unwrap_response(raw)


def _unwrap_response(raw) -> str:
    """Normalise MCP tool response to plain string."""
    if isinstance(raw, list):
        return " ".join(
            b["text"] if isinstance(b, dict) and "text" in b else str(b)
            for b in raw
        )
    if hasattr(raw, "text"):
        return raw.text
    return str(raw)

# ---------------------------------------------------------------------------
# Callback line parser
# ---------------------------------------------------------------------------

def _parse_callback_line(line: str) -> dict | None:
    try:
        signal_match     = re.search(r"signal=(\w+)", line)
        article_id_match = re.search(r"article_id=(\d+)", line)
        from_match       = re.search(r"from=@(\w+)", line)

        if not signal_match or not article_id_match:
            return None

        return {
            "signal":     signal_match.group(1),
            "article_id": int(article_id_match.group(1)),
            "from_user":  from_match.group(1) if from_match else "",
        }
    except Exception:
        return None

# ---------------------------------------------------------------------------
# One poll cycle
# ---------------------------------------------------------------------------

async def _handle_callbacks():
    # ── 1. Poll Telegram ─────────────────────────────────────────────────────
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Polling Telegram...")

    try:
        raw = await _call("telegram", "get_callbacks", {"limit": 20})
    except BaseException as e:
        print(f"  ❌ get_callbacks failed: {_unwrap_exception(e)}")
        traceback.print_exc()
        return

    print(f"  Raw response: {raw[:200]}")

    if not raw or "No new callbacks" in raw or "No callbacks found" in raw:
        print("  No new taps.")
        return

    print(f"  Callbacks:\n{raw}")

    # ── 2. Parse ──────────────────────────────────────────────────────────────
    callbacks = [
        cb for line in raw.strip().splitlines()
        if (cb := _parse_callback_line(line.strip()))
    ]

    if not callbacks:
        print("  No parseable callbacks.")
        return

    # ── 3. Process each ───────────────────────────────────────────────────────
    for cb in callbacks:
        signal     = cb["signal"]
        article_id = cb["article_id"]
        from_user  = cb["from_user"]

        print(f"\n  → signal={signal}  article_id={article_id}  from=@{from_user}")

        # 3a. Save to SQLite
        try:
            result = await _call("sqlite", "save_feedback", {
                "article_id": article_id,
                "signal":     signal,
                "from_user":  from_user,
            })
            print(f"     ✅ SQLite: {result}")
        except BaseException as e:
            print(f"     ❌ SQLite save_feedback: {_unwrap_exception(e)}")
            continue

        # 3b. Only embed negative signals
        if signal not in ("irrelevant", "skip"):
            print(f"     ℹ Chroma: skipped ('{signal}' is positive)")
            continue

        # Fetch article from SQLite
        try:
            query_raw = await _call("sqlite", "execute_query", {
                "sql":    "SELECT title, summary, url FROM articles WHERE id = ?",
                "params": [article_id],
            })
            rows = json.loads(query_raw)

            if not rows:
                print(f"     ⚠ article_id {article_id} not found in SQLite")
                continue

            article = rows[0]
            title   = article.get("title", "")
            summary = article.get("summary", "")
            url     = article.get("url", "")

        except BaseException as e:
            print(f"     ❌ SQLite execute_query: {_unwrap_exception(e)}")
            continue

        # Upsert into Chroma
        try:
            result = await _call("chroma", "upsert_embedding", {
                "collection": "feedback_vectors",
                "id":         f"{url}:{signal}",
                "document":   f"{title} {summary}".strip(),
                "metadata":   {
                    "signal":      signal,
                    "username":    from_user,
                    "article_id":  str(article_id),
                    "received_at": datetime.utcnow().isoformat(),
                },
            })
            print(f"     ✅ Chroma: {result}")

        except BaseException as e:
            print(f"     ❌ Chroma upsert_embedding: {_unwrap_exception(e)}")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run():
    print("\n" + "═" * 60)
    print("  Feedback Handler — starting")
    print(f"  Poll interval : {POLL_INTERVAL}s")
    print("  Press Ctrl+C to stop")
    print("═" * 60 + "\n")
    print("Listening for Telegram button taps...\n")

    while True:
        try:
            await _handle_callbacks()
        except BaseException as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Poll cycle error: {_unwrap_exception(e)}")
            traceback.print_exc()

        await asyncio.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n\nFeedback handler stopped.")