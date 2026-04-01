"""
Telegram MCP Server (FastMCP version)
--------------------------------------
Two-way Telegram integration for the news agent.

Tools exposed:
  - send_alert     : send a formatted news alert to your chat
  - send_message   : send any plain text / HTML message
  - get_updates    : poll for new messages / feedback replies
  - get_chat_id    : one-time setup helper to find your chat ID

Setup (run once before anything else):
  1. Message @BotFather on Telegram → /newbot → copy the token
  2. Send your new bot any message (so it has an update to return)
  3. python mcps/telegram_mcp.py --get-chat-id
  4. Copy both values into your .env:
       TELEGRAM_BOT_TOKEN=...
       TELEGRAM_CHAT_ID=...
"""

import asyncio
import os
import sys
from datetime import datetime

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("telegram-mcp")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
API_BASE  = f"https://api.telegram.org/bot{BOT_TOKEN}"

# Tracks the last seen update_id so get_updates never returns duplicates
_last_update_id: int = 0


# ---------------------------------------------------------------------------
# Low-level helpers (sync — called inside async via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _post_sync(endpoint: str, payload: dict) -> dict:
    """Synchronous POST to Telegram Bot API."""
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")
    with httpx.Client(timeout=15.0) as client:
        resp = client.post(f"{API_BASE}/{endpoint}", json=payload)
        resp.raise_for_status()
        data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error: {data.get('description', 'unknown')}")
    return data


def _get_sync(endpoint: str, params: dict = None) -> dict:
    """Synchronous GET from Telegram Bot API."""
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(f"{API_BASE}/{endpoint}", params=params or {})
        resp.raise_for_status()
        data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error: {data.get('description', 'unknown')}")
    return data


def _build_alert_text(article: dict, urgency: int, reasoning: str) -> str:
    """Format a news alert as HTML for Telegram."""
    urgency_bar = "🔴" * urgency + "⚪" * (5 - urgency)
    title       = article.get("title", "No title")
    source      = article.get("source", "")
    date        = article.get("published_at", "")[:10]
    summary     = article.get("summary", "")[:300]
    url         = article.get("url", "")

    return (
        f"<b>News Alert</b> — Urgency {urgency}/5 {urgency_bar}\n\n"
        f"<b>{title}</b>\n"
        f"<i>{source}</i> · {date}\n\n"
        f"{summary}...\n\n"
        f"<b>Why this matters to you:</b>\n{reasoning}\n\n"
        f'<a href="{url}">Read full article</a>\n\n'
        f"<i>Reply 'skip' to mark as not relevant.</i>"
    )


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def send_alert(article: dict, urgency: int, reasoning: str) -> str:
    """
    Send a formatted news alert to your Telegram chat.
    Includes title, source, summary, urgency score, and chain-of-thought reasoning.

    :param article: Dict with keys: title, url, summary, source, published_at
    :param urgency: Urgency score 1 (low) to 5 (critical)
    :param reasoning: Explanation of why this article matters to you
    """
    if not CHAT_ID:
        raise RuntimeError("TELEGRAM_CHAT_ID not set. Run get_chat_id first.")

    urgency = max(1, min(5, int(urgency)))
    text = _build_alert_text(article, urgency, reasoning)

    await asyncio.to_thread(_post_sync, "sendMessage", {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": False,
    })

    return f"Alert sent: {article.get('title', 'unknown')}"


@mcp.tool()
async def send_message(text: str, parse_mode: str = "HTML") -> str:
    """
    Send a plain text or HTML message to your Telegram chat.

    :param text: Message content. Supports HTML tags like <b>, <i>, <a href>.
    :param parse_mode: 'HTML' or 'Markdown'. Default is 'HTML'.
    """
    if not CHAT_ID:
        raise RuntimeError("TELEGRAM_CHAT_ID not set.")

    await asyncio.to_thread(_post_sync, "sendMessage", {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": parse_mode,
    })
    return "Message sent."


@mcp.tool()
async def get_updates(limit: int = 10) -> str:
    """
    Poll Telegram for new messages sent to your bot.
    Automatically deduplicates — only returns messages since the last call.
    Use this in the feedback node to check for 'skip' replies.

    :param limit: Max number of updates to fetch. Default 10.
    """
    global _last_update_id

    params = {"limit": limit, "timeout": 0}
    if _last_update_id > 0:
        params["offset"] = _last_update_id + 1

    data = await asyncio.to_thread(_get_sync, "getUpdates", params)
    updates = data.get("result", [])

    if not updates:
        return "No new messages."

    _last_update_id = max(u["update_id"] for u in updates)

    lines = []
    for update in updates:
        msg  = update.get("message", {})
        text = msg.get("text", "").strip()
        if not text:
            continue
        user = msg.get("from", {}).get("username", "unknown")
        date = datetime.fromtimestamp(msg.get("date", 0)).strftime("%Y-%m-%d %H:%M")
        lines.append(f"[{date}] @{user}: {text}")

    return "\n".join(lines) if lines else "No text messages found."


@mcp.tool()
async def get_chat_id() -> str:
    """
    One-time setup helper. Fetches recent Telegram updates to find your chat ID.
    Send any message to your bot first, then call this tool.
    Copy the printed chat ID into your .env as TELEGRAM_CHAT_ID.
    """
    data = await asyncio.to_thread(_get_sync, "getUpdates", {"limit": 5, "timeout": 0})
    updates = data.get("result", [])

    if not updates:
        return (
            "No updates found.\n"
            "Send any message to your bot on Telegram first, then call this again."
        )

    lines = ["Found chats:\n"]
    seen = set()
    for update in updates:
        chat = update.get("message", {}).get("chat", {})
        cid  = chat.get("id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        lines.append(
            f"  Chat ID  : {cid}\n"
            f"  Name     : {chat.get('first_name') or chat.get('title')}\n"
            f"  Username : @{chat.get('username', 'N/A')}\n"
            f"  → Add to .env: TELEGRAM_CHAT_ID={cid}\n"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Direct Python API (called from LangGraph nodes — no MCP protocol overhead)
# ---------------------------------------------------------------------------

async def alert(article: dict, urgency: int, reasoning: str):
    """Send a formatted alert directly from a node."""
    if not CHAT_ID:
        raise RuntimeError("TELEGRAM_CHAT_ID not set.")
    text = _build_alert_text(article, max(1, min(5, urgency)), reasoning)
    await asyncio.to_thread(_post_sync, "sendMessage", {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    })


async def get_new_messages(limit: int = 10) -> list[dict]:
    """
    Poll for new messages. Returns list of {text, from_user, date} dicts.
    Called by the feedback node to detect 'skip' replies.
    """
    global _last_update_id

    params = {"limit": limit, "timeout": 0}
    if _last_update_id > 0:
        params["offset"] = _last_update_id + 1

    data = await asyncio.to_thread(_get_sync, "getUpdates", params)
    updates = data.get("result", [])
    if not updates:
        return []

    _last_update_id = max(u["update_id"] for u in updates)

    messages = []
    for update in updates:
        msg  = update.get("message", {})
        text = msg.get("text", "").strip()
        if text:
            messages.append({
                "text":      text,
                "from_user": msg.get("from", {}).get("username", ""),
                "date":      msg.get("date", 0),
            })
    return messages


# ---------------------------------------------------------------------------
# CLI helper: python mcps/telegram_mcp.py --get-chat-id
# ---------------------------------------------------------------------------

async def _cli_get_chat_id():
    print("Fetching updates to find your chat ID...")
    data = await asyncio.to_thread(_get_sync, "getUpdates", {"limit": 5, "timeout": 0})
    updates = data.get("result", [])
    if not updates:
        print("No updates found. Send a message to your bot on Telegram first.")
        return
    for update in updates:
        chat = update.get("message", {}).get("chat", {})
        if chat:
            print(f"\nChat ID  : {chat['id']}")
            print(f"Name     : {chat.get('first_name') or chat.get('title')}")
            print(f"Username : @{chat.get('username', 'N/A')}")
            print(f"\nAdd to .env → TELEGRAM_CHAT_ID={chat['id']}")
            break


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--get-chat-id" in sys.argv:
        asyncio.run(_cli_get_chat_id())
    else:
        mcp.run(transport="stdio")