"""
Telegram MCP Server (FastMCP version)
--------------------------------------
Two-way Telegram integration for the news agent.

Tools exposed:
  - send_alert       : formatted news alert WITH inline feedback buttons
  - send_message     : plain text / HTML message
  - get_updates      : poll for new text messages
  - get_callbacks    : poll for inline button taps (feedback signals)
  - get_chat_id      : one-time setup helper

Feedback buttons on each alert:
  ✅ Useful  |  ❌ Not Useful  |  🚫 Skip Topic

Callback data format: "{signal}:{article_id}"
  e.g. "useful:42", "irrelevant:17", "skip:9"

Setup:
  1. Message @BotFather → /newbot → copy token
  2. Send bot any message
  3. python mcp/telegram.py --get-chat-id
  4. Add to .env: TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=...
"""

import asyncio
import os
import sys
from datetime import datetime
import json
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

_last_update_id: int = 0

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _post_sync(endpoint: str, payload: dict) -> dict:
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
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(f"{API_BASE}/{endpoint}", params=params or {})
        resp.raise_for_status()
        data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error: {data.get('description', 'unknown')}")
    return data


def _build_alert_text(
    article: dict,
    urgency: int,
    reasoning: str,
    connected_to: list = None,
) -> str:
    urgency_bar = "🔴" * urgency + "⚪" * (5 - urgency)
    title   = article.get("title", "No title")
    source  = article.get("source", "")
    date    = article.get("published_at", "")[:10]
    summary = article.get("summary", "")[:300]
    url     = article.get("url", "")

    text = (
        f"<b>📰 News Alert</b> — Urgency {urgency}/5 {urgency_bar}\n\n"
        f"<b>{title}</b>\n"
        f"<i>{source}</i> · {date}\n\n"
        f"{summary}...\n\n"
        f"<b>💡 Why this matters to you:</b>\n{reasoning}\n"
    )

    if connected_to:
        text += "\n<b>🔗 Connected to:</b>\n"
        for past in connected_to[:3]:
            past_title = past.get("title", "")
            past_date  = past.get("published_at", "")[:10]
            if past_title:
                text += f"  • {past_title} ({past_date})\n"

    text += f'\n<a href="{url}">🌐 Read full article</a>'
    return text


def _build_keyboard(article_id: int) -> dict:
    return {
        "inline_keyboard": [[
            {"text": "✅ Useful",      "callback_data": f"useful:{article_id}"},
            {"text": "❌ Not Useful",  "callback_data": f"irrelevant:{article_id}"},
            {"text": "🚫 Skip Topic", "callback_data": f"skip:{article_id}"},
        ]]
    }


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def send_alert(
    article     : str,         # JSON string — MCP tools pass primitives only
    urgency     : int,
    reasoning   : str,
    article_id  : int,
    connected_to: str = "[]",  # JSON string list
) -> str:
    """
    Send a formatted news alert with inline ✅ ❌ 🚫 feedback buttons.
    Button taps are readable via get_callbacks tool.

    :param article: JSON string with title, url, summary, source, published_at
    :param urgency: Score 1–5
    :param reasoning: Why this matters to the user (from sequential thinking)
    :param article_id: SQLite article row ID — links button tap back to article
    :param connected_to: JSON string list of connected past articles [{title, published_at}]
    """
    if not CHAT_ID:
        raise RuntimeError("TELEGRAM_CHAT_ID not set.")

    # Safely parse — handle both str and already-parsed dict/list
    if isinstance(article, str):
        article = json.loads(article)
    if isinstance(connected_to, str):
        connected_to = json.loads(connected_to)
    if not isinstance(connected_to, list):
        connected_to = []

    urgency  = max(1, min(5, int(urgency)))
    text     = _build_alert_text(article, urgency, reasoning, connected_to)
    keyboard = _build_keyboard(article_id)

    await asyncio.to_thread(_post_sync, "sendMessage", {
        "chat_id":                  CHAT_ID,
        "text":                     text,
        "parse_mode":               "HTML",
        "disable_web_page_preview": False,
        "reply_markup":             keyboard,
    })

    return f"Alert sent: '{article.get('title', 'unknown')}' (article_id={article_id})"


@mcp.tool()
async def send_message(text: str, parse_mode: str = "HTML") -> str:
    """
    Send a plain text or HTML message to your Telegram chat.

    :param text: Message content. Supports HTML tags like <b>, <i>, <a href>.
    :param parse_mode: 'HTML' or 'Markdown'. Default HTML.
    """
    if not CHAT_ID:
        raise RuntimeError("TELEGRAM_CHAT_ID not set.")

    await asyncio.to_thread(_post_sync, "sendMessage", {
        "chat_id":    CHAT_ID,
        "text":       text,
        "parse_mode": parse_mode,
    })
    return "Message sent."


@mcp.tool()
async def get_updates(limit: int = 10) -> str:
    """
    Poll for new TEXT messages sent to your bot.
    Auto-deduplicates — only returns messages since last call.

    :param limit: Max updates to fetch. Default 10.
    """
    global _last_update_id

    params = {"limit": limit, "timeout": 0}
    if _last_update_id > 0:
        params["offset"] = _last_update_id + 1

    data    = await asyncio.to_thread(_get_sync, "getUpdates", params)
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
async def get_callbacks(limit: int = 10) -> str:
    """
    Poll for inline button taps (callback queries) from feedback buttons.
    Auto-acknowledges each tap so Telegram removes the loading spinner.
    Auto-deduplicates — only returns taps since last call.

    Returns one line per tap:
      signal=useful      article_id=42  from=@username  date=2026-04-02 08:00
      signal=skip        article_id=17  from=@username  date=2026-04-02 08:01

    Signals: useful | irrelevant | skip

    :param limit: Max updates to fetch. Default 10.
    """
    global _last_update_id

    params = {"limit": limit, "timeout": 0}
    if _last_update_id > 0:
        params["offset"] = _last_update_id + 1

    data    = await asyncio.to_thread(_get_sync, "getUpdates", params)
    updates = data.get("result", [])

    if not updates:
        return "No new callbacks."

    _last_update_id = max(u["update_id"] for u in updates)

    lines = []
    for update in updates:
        cb = update.get("callback_query")
        if not cb:
            continue

        callback_data = cb.get("data", "")
        callback_id   = cb.get("id", "")
        user          = cb.get("from", {}).get("username", "unknown")
        date          = datetime.now().strftime("%Y-%m-%d %H:%M")

        if ":" not in callback_data:
            continue

        signal, article_id = callback_data.split(":", 1)

        # Acknowledge immediately — removes spinner from button.
        # Telegram only allows ~60s to ack — after that it returns 400.
        # The callback data is still valid so we always continue processing.
        try:
            await asyncio.to_thread(_post_sync, "answerCallbackQuery", {
                "callback_query_id": callback_id,
                "text": f"Marked as: {signal}",
            })
        except Exception:
            pass  # expired callback ID — safe to ignore

        lines.append(
            f"signal={signal}  article_id={article_id}  "
            f"from=@{user}  date={date}"
        )

    return "\n".join(lines) if lines else "No callbacks found."


@mcp.tool()
async def get_chat_id() -> str:
    """
    One-time setup helper. Fetches recent updates to find your chat ID.
    Send any message to your bot on Telegram first, then call this tool.
    Copy the printed chat ID into your .env as TELEGRAM_CHAT_ID.
    """
    data    = await asyncio.to_thread(_get_sync, "getUpdates", {"limit": 5, "timeout": 0})
    updates = data.get("result", [])

    if not updates:
        return "No updates found. Send a message to your bot on Telegram first."

    lines = ["Found chats:\n"]
    seen  = set()
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
# CLI helper
# ---------------------------------------------------------------------------

async def _cli_get_chat_id():
    print("Fetching updates to find your chat ID...")
    data    = await asyncio.to_thread(_get_sync, "getUpdates", {"limit": 5, "timeout": 0})
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