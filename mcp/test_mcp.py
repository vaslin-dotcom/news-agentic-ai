"""
test_mcp.py — Test all MCP servers via the actual MCP protocol
---------------------------------------------------------------
Spins up each server as a subprocess and calls tools through
the real MCP stdio transport — exactly how LangGraph will use them.

Usage:
    python mcp/test_mcp.py              # test all servers
    python mcp/test_mcp.py ddg          # test one server
    python mcp/test_mcp.py chroma
    python mcp/test_mcp.py sqlite
    python mcp/test_mcp.py github
    python mcp/test_mcp.py telegram
"""

import asyncio
import json
import sys
import traceback
import uuid
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ── Colour helpers ──────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")
def info(msg): print(f"  {CYAN}→{RESET} {msg}")

def section(name):
    print(f"\n{BOLD}{'─' * 50}{RESET}")
    print(f"{BOLD}{CYAN} {name}{RESET}")
    print(f"{BOLD}{'─' * 50}{RESET}")

# Path to the mcp/ folder (same folder this file lives in)
MCP_DIR = Path(__file__).parent

DUMMY_VEC = [0.01 * (i % 10) for i in range(384)]


# ── Helper: connect to a server and run a test coroutine ────────────────────

async def with_server(script: str, test_fn):
    """
    Starts `python mcp/<script>.py` as a subprocess MCP server,
    opens a ClientSession, runs test_fn(session), then shuts down cleanly.
    """
    server_params = StdioServerParameters(
        command="python",
        args=[str(MCP_DIR / f"{script}.py")],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await test_fn(session)


# ── Helper: call a tool and return parsed result ─────────────────────────────

async def call(session: ClientSession, tool: str, **kwargs) -> str:
    result = await session.call_tool(tool, arguments=kwargs)
    if result.isError:
        error_text = result.content[0].text if result.content else "Unknown error"
        raise RuntimeError(f"Tool {tool} returned error: {error_text}")
    return result.content[0].text


# ═══════════════════════════════════════════════════════════════════════════
# 1. DDG News
# ═══════════════════════════════════════════════════════════════════════════

async def test_ddg():
    section("DDG News MCP server")

    async def run(session: ClientSession):
        info("Listing tools...")
        tools = await session.list_tools()
        tool_names = [t.name for t in tools.tools]
        assert "fetch_news" in tool_names, f"fetch_news not found, got: {tool_names}"
        ok(f"Tools registered: {tool_names}")

        info("Calling fetch_news(query='Python AI', max_results=3)...")
        raw = await call(session, "fetch_news", query="Python AI", max_results=3)
        assert "articles" in raw.lower() or "found" in raw.lower(), f"Unexpected response: {raw[:100]}"
        ok(f"fetch_news responded ({len(raw)} chars)")
        info(f"Preview: {raw[:120].strip()}...")

    await with_server("ddg", run)
    print(f"\n  {GREEN}{BOLD}DDG — SERVER TEST PASSED{RESET}")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# 2. Chroma
# ═══════════════════════════════════════════════════════════════════════════

async def test_chroma():
    section("Chroma MCP server")

    async def run(session: ClientSession):
        info("Listing tools...")
        tools = await session.list_tools()
        tool_names = [t.name for t in tools.tools]
        expected = {"upsert_embedding", "query_similar", "delete_embedding", "list_collections"}
        missing = expected - set(tool_names)
        assert not missing, f"Missing tools: {missing}"
        ok(f"Tools registered: {tool_names}")

        info("Calling list_collections...")
        raw = await call(session, "list_collections")
        ok(f"list_collections: {raw.strip()}")

        info("Calling upsert_embedding (no embedding — auto-generated)...")
        raw = await call(
            session, "upsert_embedding",
            collection="profile_vectors",
            id="test-server-001",
            document="Testing the Chroma MCP server via protocol",
            metadata={"type": "test"},
        )
        assert "Upserted" in raw, f"Unexpected: {raw}"
        ok(f"upsert_embedding: {raw.strip()}")

        info("Calling query_similar (query_text — auto-embedded)...")
        raw = await call(
            session, "query_similar",
            collection="profile_vectors",
            query_text="Testing the Chroma MCP server via protocol",
            n_results=1,
        )
        assert "test-server-001" in raw, f"Expected ID in results: {raw[:200]}"
        ok("query_similar returned correct ID")

        info("Calling delete_embedding...")
        raw = await call(session, "delete_embedding",
                         collection="profile_vectors", id="test-server-001")
        assert "Deleted" in raw, f"Unexpected: {raw}"
        ok(f"delete_embedding: {raw.strip()}")

    await with_server("chroma", run)
    print(f"\n  {GREEN}{BOLD}Chroma — SERVER TEST PASSED{RESET}")
    return True

# ═══════════════════════════════════════════════════════════════════════════
# 3. SQLite
# ═══════════════════════════════════════════════════════════════════════════

async def test_sqlite():
    section("SQLite MCP server")

    # Generate unique identifiers for this test run to avoid leftovers
    unique_suffix = uuid.uuid4().hex[:8]
    test_url = f"https://example.com/test-mcp-server-article-{unique_suffix}"
    test_key = f"test_job_{unique_suffix}"

    async def run(session: ClientSession):
        info("Listing tools...")
        tools = await session.list_tools()
        tool_names = [t.name for t in tools.tools]
        expected = {
            "execute_query", "upsert_profile", "get_profile",
            "save_article", "save_alert", "save_feedback",
            "get_articles", "url_exists",
        }
        missing = expected - set(tool_names)
        assert not missing, f"Missing tools: {missing}"
        ok(f"Tools registered: {tool_names}")

        # upsert_profile
        info("Calling upsert_profile...")
        raw = await call(session, "upsert_profile", key=test_key, value="MCP Tester")
        assert "saved" in raw.lower(), f"Unexpected: {raw}"
        ok(f"upsert_profile: {raw.strip()}")

        # get_profile
        info("Calling get_profile...")
        raw = await call(session, "get_profile")
        profile = json.loads(raw)
        assert test_key in profile, f"{test_key} missing from profile: {profile}"
        ok(f"get_profile returned {len(profile)} fields, {test_key}='{profile[test_key]}'")

        # url_exists (before save)
        info("Calling url_exists (before save)...")
        raw = await call(session, "url_exists", url=test_url)
        data = json.loads(raw)
        assert data["exists"] is False
        ok("url_exists correctly returns False before save")

        # save_article
        info("Calling save_article...")
        raw = await call(
            session, "save_article",
            url=test_url,
            title="MCP Server Test Article",
            source="test_mcp.py",
            summary="Testing the SQLite MCP server via the MCP protocol.",
            published_at="2024-11-01",
            relevance_score=0.9,
            urgency=4,
            reasoning="Test reasoning",
            embedding_id="test-embed-server",
        )
        data = json.loads(raw)
        assert data["status"] in ("saved", "duplicate")
        article_id = data["article_id"]
        ok(f"save_article: status={data['status']}, article_id={article_id}")

        # url_exists (after save)
        info("Calling url_exists (after save)...")
        raw = await call(session, "url_exists", url=test_url)
        data = json.loads(raw)
        assert data["exists"] is True
        assert data["article_id"] == article_id
        ok(f"url_exists correctly returns True, article_id={data['article_id']}")

        # save_alert
        info("Calling save_alert...")
        raw = await call(session, "save_alert",
                         article_id=article_id, urgency=4, reasoning="Test alert")
        assert "recorded" in raw.lower(), f"Unexpected: {raw}"
        ok(f"save_alert: {raw.strip()}")

        # save_feedback
        info("Calling save_feedback...")
        raw = await call(session, "save_feedback",
                         article_id=article_id, signal="skip", from_user="tester")
        assert "saved" in raw.lower(), f"Unexpected: {raw}"
        ok(f"save_feedback: {raw.strip()}")

        # get_articles
        info("Calling get_articles...")
        raw = await call(session, "get_articles", limit=5, min_urgency=0)
        articles = json.loads(raw)
        assert isinstance(articles, list)
        ok(f"get_articles returned {len(articles)} rows")

        # execute_query (SELECT only)
        info("Calling execute_query (SELECT)...")
        raw = await call(session, "execute_query",
                         sql=f"SELECT key, value FROM profile WHERE key = '{test_key}'")
        rows = json.loads(raw)
        assert isinstance(rows, list)
        ok(f"execute_query returned {len(rows)} rows")

        # execute_query blocked for non-SELECT
        info("Calling execute_query with DELETE (should be blocked)...")
        raw = await call(session, "execute_query",
                         sql=f"DELETE FROM profile WHERE key='{test_key}'")
        assert "only SELECT" in raw, f"Expected block message, got: {raw}"
        ok("Non-SELECT query correctly blocked")

    await with_server("sqlite", run)
    print(f"\n  {GREEN}{BOLD}SQLite — SERVER TEST PASSED{RESET}")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# 4. GitHub
# ═══════════════════════════════════════════════════════════════════════════

async def test_github():
    section("GitHub MCP server")

    async def run(session: ClientSession):
        info("Listing tools...")
        tools = await session.list_tools()
        tool_names = [t.name for t in tools.tools]
        expected = {"get_user_profile", "get_repositories", "get_languages", "get_pinned_topics"}
        missing = expected - set(tool_names)
        assert not missing, f"Missing tools: {missing}"
        ok(f"Tools registered: {tool_names}")

        info("Calling get_user_profile(username='vaslin-dotcom')...")
        raw = await call(session, "get_user_profile", username="vaslin-dotcom")
        profile = json.loads(raw)
        assert profile["username"] == "vaslin-dotcom"
        ok(f"get_user_profile: {profile['name']}, repos={profile['public_repos']}")

        info("Calling get_repositories(username='vaslin-dotcom', max_repos=3)...")
        raw = await call(session, "get_repositories", username="vaslin-dotcom", max_repos=3)
        repos = json.loads(raw)
        assert isinstance(repos, list) and len(repos) > 0
        ok(f"get_repositories: {len(repos)} repos, first='{repos[0]['name']}'")

        info("Calling get_languages(username='torvalds', max_repos=5)...")
        raw = await call(session, "get_languages", username="torvalds", max_repos=5)
        langs = json.loads(raw)
        assert isinstance(langs, list)
        ok(f"get_languages: {[l['language'] for l in langs[:3]]}")

        info("Calling get_pinned_topics(username='torvalds', max_repos=5)...")
        raw = await call(session, "get_pinned_topics", username="torvalds", max_repos=5)
        topics = json.loads(raw)
        assert isinstance(topics, list)
        ok(f"get_pinned_topics: {topics[:3] if topics else '(none — normal)'}")

    await with_server("github", run)
    print(f"\n  {GREEN}{BOLD}GitHub — SERVER TEST PASSED{RESET}")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# 5. Telegram
# ═══════════════════════════════════════════════════════════════════════════

async def test_telegram():
    section("Telegram MCP server")

    import os
    from dotenv import load_dotenv
    load_dotenv()

    token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not token or token == "your_bot_token_here":
        warn("TELEGRAM_BOT_TOKEN not configured — skipping")
        warn("Set it in .env then run: python mcp/test_mcp.py telegram")
        return None

    if not chat_id or chat_id == "your_chat_id_here":
        warn("TELEGRAM_CHAT_ID not configured — skipping")
        warn("Run: python mcp/telegram.py --get-chat-id")
        return None

    async def run(session: ClientSession):
        info("Listing tools...")
        tools = await session.list_tools()
        tool_names = [t.name for t in tools.tools]
        expected = {"send_alert", "send_message", "get_updates", "get_chat_id"}
        missing = expected - set(tool_names)
        assert not missing, f"Missing tools: {missing}"
        ok(f"Tools registered: {tool_names}")

        info("Calling send_message...")
        raw = await call(
            session, "send_message",
            text="<b>MCP server test</b> — send_message tool is working.",
        )
        assert "sent" in raw.lower(), f"Unexpected: {raw}"
        ok(f"send_message: {raw.strip()}")

        info("Calling send_alert...")
        raw = await call(
            session, "send_alert",
            article={
                "title":        "MCP Server Test Alert",
                "url":          "https://example.com",
                "summary":      "This verifies send_alert works via the MCP protocol.",
                "source":       "test_mcp.py",
                "published_at": "2024-11-01",
            },
            urgency=2,
            reasoning="Test message — MCP server protocol is working correctly.",
        )
        assert "sent" in raw.lower(), f"Unexpected: {raw}"
        ok(f"send_alert: {raw.strip()}")
        ok("Check your Telegram — you should have 2 new messages")

        info("Calling get_updates...")
        raw = await call(session, "get_updates", limit=5)
        ok(f"get_updates: {raw[:80].strip()}...")

    await with_server("telegram", run)
    print(f"\n  {GREEN}{BOLD}Telegram — SERVER TEST PASSED{RESET}")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════

async def main(targets: list[str]):
    tests = {
        "ddg":      ("DDG News",  test_ddg),
        "chroma":   ("Chroma",    test_chroma),
        "sqlite":   ("SQLite",    test_sqlite),
        "github":   ("GitHub",    test_github),
        "telegram": ("Telegram",  test_telegram),
    }

    to_run = {k: v for k, v in tests.items() if not targets or k in targets}
    results = {}

    for key, (name, fn) in to_run.items():
        try:
            result = await fn()
            if result is None:
                results[key] = "SKIP"
            elif result is True:
                results[key] = "PASS"
            else:
                results[key] = "FAIL"
        except AssertionError as e:
            fail(f"Assertion failed: {e}")
            results[key] = "FAIL"
        except Exception as e:
            fail(f"{type(e).__name__}: {e}")
            traceback.print_exc()
            results[key] = "FAIL"

    # Summary
    print(f"\n{BOLD}{'═' * 50}{RESET}")
    print(f"{BOLD} Summary{RESET}")
    print(f"{BOLD}{'═' * 50}{RESET}")
    for key, status in results.items():
        name = tests[key][0]
        if status == "PASS":
            print(f"  {GREEN}✓ PASS{RESET}  {name}")
        elif status == "SKIP":
            print(f"  {YELLOW}⚠ SKIP{RESET}  {name}")
        else:
            print(f"  {RED}✗ FAIL{RESET}  {name}")

    if any(v == "FAIL" for v in results.values()):
        print(f"\n{RED}Fix failing servers before building nodes.{RESET}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}{BOLD}All servers working. Ready to build nodes.{RESET}")


if __name__ == "__main__":
    targets = [a.lower() for a in sys.argv[1:]]
    asyncio.run(main(targets))