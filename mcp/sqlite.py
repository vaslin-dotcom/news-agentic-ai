"""
SQLite MCP Server (FastMCP version)
-------------------------------------
Relational storage for the news agent.

Tables managed:
  - profile    : your structured profile (job, interests, goals, exclusions)
  - articles   : every processed article with embedding ref
  - alerts     : articles that crossed the urgency threshold
  - feedback   : your 'skip' replies from Telegram

Tools exposed:
  - execute_query   : run any SELECT (read-only safe queries)
  - upsert_profile  : insert or update your profile fields
  - save_article    : store a processed article
  - save_alert      : store an alert that was sent
  - save_feedback   : store a feedback signal (skip / irrelevant)
  - get_profile     : retrieve your full profile as a dict
  - get_articles    : fetch recent articles with optional filters
  - url_exists      : check if an article URL was already processed
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("sqlite-mcp")

# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

DB_PATH = str(Path(__file__).parent.parent / "db" / "news_agent.db")
os.makedirs(Path(DB_PATH).parent, exist_ok=True)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    conn.execute("PRAGMA journal_mode=WAL")  # safe for concurrent reads
    return conn


def _init_db():
    """Create all tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS profile (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS articles (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            url             TEXT UNIQUE NOT NULL,
            title           TEXT,
            source          TEXT,
            summary         TEXT,
            published_at    TEXT,
            fetched_at      TEXT NOT NULL,
            relevance_score REAL,
            urgency         INTEGER,
            reasoning       TEXT,
            embedding_id    TEXT
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id  INTEGER REFERENCES articles(id),
            sent_at     TEXT NOT NULL,
            urgency     INTEGER,
            reasoning   TEXT
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id  INTEGER REFERENCES articles(id),
            signal      TEXT NOT NULL,
            received_at TEXT NOT NULL,
            from_user   TEXT
        );
    """)
    conn.commit()
    conn.close()


_init_db()


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def execute_query(sql: str, params: list = None) -> str:
    """
    Execute a read-only SQL SELECT query and return results as JSON.
    Use this for any custom lookups not covered by other tools.

    :param sql: A SELECT statement. INSERT/UPDATE/DELETE are blocked.
    :param params: Optional list of positional parameters for the query.
    """
    if not sql.strip().upper().startswith("SELECT"):
        return "Error: only SELECT queries are allowed via execute_query."

    conn = _get_conn()
    try:
        cursor = conn.execute(sql, params or [])
        rows = [dict(row) for row in cursor.fetchall()]
        return json.dumps(rows, indent=2, default=str)
    except sqlite3.Error as e:
        return f"SQL error: {e}"
    finally:
        conn.close()


@mcp.tool()
def upsert_profile(key: str, value: str) -> str:
    """
    Insert or update a single profile field.
    Keys are free-form strings e.g. 'job', 'interests', 'goals', 'exclusions', 'github_username'.

    :param key: Profile field name
    :param value: Field value — store lists/dicts as JSON strings
    """
    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO profile (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, value, now),
        )
        conn.commit()
        return f"Profile field '{key}' saved."
    finally:
        conn.close()


@mcp.tool()
def get_profile() -> str:
    """
    Retrieve the full profile as a JSON dict.
    Returns all key-value pairs stored via upsert_profile.
    """
    conn = _get_conn()
    try:
        rows = conn.execute("SELECT key, value FROM profile").fetchall()
        profile = {row["key"]: row["value"] for row in rows}
        return json.dumps(profile, indent=2)
    finally:
        conn.close()


@mcp.tool()
def save_article(
    url: str,
    title: str,
    source: str,
    summary: str,
    published_at: str,
    relevance_score: float = 0.0,
    urgency: int = 0,
    reasoning: str = "",
    embedding_id: str = "",
) -> str:
    """
    Store a processed article. Silently ignored if the URL already exists.
    Returns JSON with status and article_id.

    :param url: Article URL (unique key for deduplication)
    :param title: Article headline
    :param source: Publisher name
    :param summary: Short text summary
    :param published_at: ISO date string
    :param relevance_score: Cosine similarity score from Chroma (0.0–1.0)
    :param urgency: Urgency score 1–5 from reasoning step
    :param reasoning: Chain-of-thought reasoning text
    :param embedding_id: ID used in Chroma article_vectors collection
    """
    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    try:
        cursor = conn.execute(
            """
            INSERT INTO articles
                (url, title, source, summary, published_at, fetched_at,
                 relevance_score, urgency, reasoning, embedding_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO NOTHING
            """,
            (url, title, source, summary, published_at, now,
             relevance_score, urgency, reasoning, embedding_id),
        )
        conn.commit()
        if cursor.lastrowid:
            return json.dumps({"status": "saved", "article_id": cursor.lastrowid})
        row = conn.execute("SELECT id FROM articles WHERE url=?", (url,)).fetchone()
        return json.dumps({"status": "duplicate", "article_id": row["id"]})
    finally:
        conn.close()


@mcp.tool()
def save_alert(article_id: int, urgency: int, reasoning: str) -> str:
    """
    Record that an alert was sent for an article.

    :param article_id: Row ID from the articles table
    :param urgency: Urgency score that triggered the alert
    :param reasoning: Reasoning text that was included in the alert
    """
    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO alerts (article_id, sent_at, urgency, reasoning) VALUES (?, ?, ?, ?)",
            (article_id, now, urgency, reasoning),
        )
        conn.commit()
        return f"Alert recorded for article_id {article_id}."
    finally:
        conn.close()


@mcp.tool()
def save_feedback(article_id: int, signal: str, from_user: str = "") -> str:
    """
    Store a feedback signal received via Telegram.

    :param article_id: Row ID from the articles table
    :param signal: Feedback string e.g. 'skip', 'relevant', 'irrelevant'
    :param from_user: Telegram username who sent the feedback
    """
    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO feedback (article_id, signal, received_at, from_user) VALUES (?, ?, ?, ?)",
            (article_id, signal, now, from_user),
        )
        conn.commit()
        return f"Feedback '{signal}' saved for article_id {article_id}."
    finally:
        conn.close()


@mcp.tool()
def get_articles(limit: int = 20, min_urgency: int = 0, source: str = "") -> str:
    """
    Fetch recent processed articles with optional filters.

    :param limit: Max rows to return. Default 20.
    :param min_urgency: Only return articles with urgency >= this value.
    :param source: Filter by publisher name (partial match).
    """
    conn = _get_conn()
    try:
        sql = "SELECT * FROM articles WHERE urgency >= ?"
        params: list = [min_urgency]
        if source:
            sql += " AND source LIKE ?"
            params.append(f"%{source}%")
        sql += " ORDER BY fetched_at DESC LIMIT ?"
        params.append(limit)
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
        return json.dumps(rows, indent=2, default=str)
    finally:
        conn.close()


@mcp.tool()
def url_exists(url: str) -> str:
    """
    Check if an article URL has already been processed.
    Returns JSON: {"exists": true/false, "article_id": <id or null>}

    :param url: The article URL to check
    """
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT id FROM articles WHERE url=?", (url,)
        ).fetchone()
        if row:
            return json.dumps({"exists": True, "article_id": row["id"]})
        return json.dumps({"exists": False, "article_id": None})
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Direct Python API (called from LangGraph nodes)
# ---------------------------------------------------------------------------

def db_upsert_profile(key: str, value: str):
    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO profile (key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, value, now),
    )
    conn.commit()
    conn.close()


def db_get_profile() -> dict:
    conn = _get_conn()
    rows = conn.execute("SELECT key, value FROM profile").fetchall()
    conn.close()
    return {row["key"]: row["value"] for row in rows}


def db_save_article(
    article: dict,
    relevance_score: float,
    urgency: int,
    reasoning: str,
    embedding_id: str,
) -> int:
    """Saves article and returns its article_id (new or existing)."""
    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    cursor = conn.execute(
        """
        INSERT INTO articles
            (url, title, source, summary, published_at, fetched_at,
             relevance_score, urgency, reasoning, embedding_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(url) DO NOTHING
        """,
        (
            article.get("url"), article.get("title"), article.get("source"),
            article.get("summary"), article.get("published_at"), now,
            relevance_score, urgency, reasoning, embedding_id,
        ),
    )
    conn.commit()
    if cursor.lastrowid:
        article_id = cursor.lastrowid
    else:
        article_id = conn.execute(
            "SELECT id FROM articles WHERE url=?", (article.get("url"),)
        ).fetchone()["id"]
    conn.close()
    return article_id


def db_save_alert(article_id: int, urgency: int, reasoning: str):
    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO alerts (article_id, sent_at, urgency, reasoning) VALUES (?, ?, ?, ?)",
        (article_id, now, urgency, reasoning),
    )
    conn.commit()
    conn.close()


def db_save_feedback(article_id: int, signal: str, from_user: str = ""):
    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO feedback (article_id, signal, received_at, from_user) VALUES (?, ?, ?, ?)",
        (article_id, signal, now, from_user),
    )
    conn.commit()
    conn.close()


def db_url_exists(url: str) -> tuple[bool, int | None]:
    """Returns (exists: bool, article_id: int | None)."""
    conn = _get_conn()
    row = conn.execute("SELECT id FROM articles WHERE url=?", (url,)).fetchone()
    conn.close()
    if row:
        return True, row["id"]
    return False, None


def db_get_recent_articles(limit: int = 20, min_urgency: int = 0) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM articles WHERE urgency >= ? ORDER BY fetched_at DESC LIMIT ?",
        (min_urgency, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")