# 🤖 News Agentic AI

A fully agentic, self-improving personalised news system built with **LangGraph** and **MCP (Model Context Protocol)**. It learns your profile, fetches news from multiple sources, reasons deeply about each article's relevance to your specific life, and delivers Telegram alerts with inline feedback buttons — getting smarter with every interaction.

---

## How It Works

The system runs as two independent LangGraph graphs:

**Graph 1 — Profile Builder**: Builds a rich user profile by fetching GitHub data, synthesising professional context via LLM, then having a conversational interview to fill in personal details. Everything is stored in SQLite and embedded into ChromaDB.

**Graph 2 — News Collector**: Reads the profile, generates search queries, fetches news, filters by semantic relevance, reasons deeply about each article using Sequential Thinking MCP, saves scored articles, and sends Telegram alerts for anything above urgency 3.

A **Feedback Handler** runs continuously alongside the graph, polling Telegram for button clicks (✅ Useful / ❌ Not Useful / 🚫 Skip Topic) and feeding them back into ChromaDB — penalising future similar articles automatically.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   GRAPH 1 — Profile Builder             │
│                                                         │
│  fetch_github → build_profile → [save_sqlite            │
│                                  embed_chroma]          │
│              → chat_node    → [save_sqlite              │
│                                  embed_chroma]          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  GRAPH 2 — News Collector               │
│                                                         │
│  load_profile → generate_queries → fetch_news           │
│              → relevance_filter → deep_reasoning        │
│              → [save_node ‖ embed_node] → alert_node    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│           FEEDBACK HANDLER (always-on process)          │
│                                                         │
│  poll Telegram → save_feedback → embed into             │
│  feedback_vectors → penalise future similar articles    │
└─────────────────────────────────────────────────────────┘
```

---

## News Collector — Node by Node

### Node 1 — `load_profile_node`
Reads the full profile from SQLite and all profile vector chunks from ChromaDB. These chunks are passed to the deep reasoning node so the LLM has rich semantic context for each article.

### Node 2 — `generate_queries_node`
Uses an LLM with a structured template-based prompt to generate targeted DuckDuckGo search queries covering every dimension of the profile: company, job role, skills, tech stack, personal interests, languages, location, and career growth. Unknown profile fields are handled gracefully by LLM inference.

### Node 3 — `fetch_news_node`
Fetches news for all queries using the DDG MCP server. Deduplicates by URL and title across all query results. Filters already-seen URLs via SQLite. Uses **DuckDuckGo as primary** and **Google News RSS as fallback** when DDG returns empty results or hits rate limits.

### Node 4 — `relevance_filter_node`
For each article:
- Queries `profile_vectors` in Chroma to get a base relevance score
- Queries `feedback_vectors` to apply penalties (skip = −0.4, irrelevant = −0.2)
- Hard-drops articles matching `news_exclusions`
- Keeps articles with `final_score >= 0.3`

### Node 5 — `deep_reasoning_node`
The core of the system. For each article, using the **Sequential Thinking MCP server**:
1. Understands what the article is actually about
2. Maps it to specific dimensions of the person's profile
3. Considers the person's exact role, company, and location
4. Checks connected past articles from `article_vectors` for bigger picture
5. Applies feedback history to adjust urgency
6. Assesses real-world impact on this specific person
7. Outputs: `urgency (1–5)`, `relevance_score`, `reasoning`, `connected_to`

**Urgency scale:**
| Level | Meaning |
|-------|---------|
| 1 | Mildly interesting, no action needed |
| 2 | Worth saving, no alert |
| 3 | Notable — alert sent |
| 4 | Directly affects your work or life — alert sent |
| 5 | Act on this now — alert sent |

### Nodes 6a & 6b — `save_node` + `embed_node` (parallel)
Save all scored articles to SQLite and embed them into `article_vectors` in ChromaDB simultaneously.

### Node 7 — `alert_node`
Sends Telegram alerts for all articles with urgency ≥ 3. Each alert includes title, source, date, summary, personalised reasoning, connected past articles, a read link, and three inline feedback buttons.

```
📰 News Alert — Urgency 4/5 🔴🔴🔴🔴⚪

Infosys announces AI-first strategy at Investor Day 2026
Economic Times · 2026-04-05

Infosys management outlined a complete pivot to AI-led delivery...

💡 Why this matters to you:
You are an AI/ML engineer at Infosys Mysore. This directly 
affects your team's roadmap and project priorities.

🔗 Connected to:
• Anthropic-Infosys partnership announced (2026-02-17)
• Infosys Q2 earnings: AI projects at 4,600 (2026-01-15)

🌐 Read full article

✅ Useful  |  ❌ Not Useful  |  🚫 Skip Topic
```

---

## How the System Improves Over Time

```
Run 1   → feedback_vectors empty → no penalties applied
          article_vectors empty  → no connections found

Run 10  → light feedback history → some topics penalised
          article_vectors growing → connections found in reasoning

Run 30+ → rich feedback history → filter drops most irrelevant articles
          dense article_vectors  → multi-hop connections in reasoning
          alerts feel deeply personal → system has learned your taste
```

The system self-corrects without any manual tuning. Only improves. Never degrades.

---

## MCP Servers

| Server | Transport | Tools |
|--------|-----------|-------|
| `mcp/sqlite.py` | stdio | `get_profile`, `save_article`, `save_alert`, `save_feedback`, `url_exists`, `execute_query`, `upsert_profile`, `get_articles` |
| `mcp/chroma.py` | stdio / streamable_http | `upsert_embedding`, `query_similar`, `delete_embedding`, `list_collections` |
| `mcp/ddg_news.py` | stdio | `fetch_news` (DDG primary + Google RSS fallback), `fetch_article_content` |
| `mcp/telegram.py` | stdio | `send_alert`, `send_message`, `get_updates`, `get_callbacks`, `get_chat_id` |
| `mcp/github.py` | stdio | `get_user_profile`, `get_repositories`, `get_languages`, `get_pinned_topics` |
| `sequential-thinking` | stdio (npx) | `think` (multi-step reasoning chain) |

---

## Database Schema

**SQLite** (`db/news_agent.db`)
- `profile` — flat key-value store (key, value, updated_at)
- `articles` — url, title, source, summary, published_at, fetched_at, relevance_score, urgency, reasoning, embedding_id
- `alerts` — article_id FK, sent_at, urgency, reasoning
- `feedback` — article_id FK, signal, received_at, from_user

**ChromaDB** (`db/chroma/`)
- `profile_vectors` — professional, company, goals, interests, personal, lifestyle, exclusions chunks
- `article_vectors` — all processed articles (id = URL)
- `feedback_vectors` — skipped/irrelevant articles with signal metadata

---



## Project Structure

```
news-agentic-ai/
├── graphs/
│   ├── profile_builder/
│   │   ├── graph.py
│   │   ├── state.py
│   │   ├── mcp_tools.py
│   │   ├── agent_utils.py
│   │   └── nodes/
│   │       ├── fetch_github_node.py
│   │       ├── build_profile_node.py
│   │       ├── save_profile_sqlite_node.py
│   │       ├── embed_profile_chroma_node.py
│   │       ├── chat_node.py
│   │       ├── save_chat_sqlite_node.py
│   │       └── embed_chat_chroma_node.py
│   └── news_collector/
│       ├── graph.py
│       ├── state.py
│       ├── feedback_handler.py
│       └── nodes/
│           ├── load_profile_node.py
│           ├── generate_queries_node.py
│           ├── fetch_news_node.py
│           ├── relevance_filter_node.py
│           ├── deep_reasoning_node.py
│           ├── save_node.py
│           ├── embed_node.py
│           └── alert_node.py
├── mcp/
│   ├── sqlite.py
│   ├── chroma.py
│   ├── ddg_news.py
│   ├── telegram.py
│   └── github.py
├── config.py
├── llm.py
├── db_test.py
├── workflow.txt
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.11+
- Node.js (for Sequential Thinking MCP)
- Telegram bot token + chat ID
- NVIDIA NIM API key
- Groq API key
- GitHub personal access token (for profile builder)

### Install

```bash
git clone https://github.com/vaslin-dotcom/news-agentic-ai
cd news-agentic-ai
pip install langchain langgraph langchain-openai langchain-mcp-adapters \
            chromadb sentence-transformers mcp fastmcp \
            ddgs feedparser httpx beautifulsoup4 python-telegram-bot
```

Install the Sequential Thinking MCP server:
```bash
npm install -g @modelcontextprotocol/server-sequential-thinking
```

### Configure

Create a `config.py` with your keys:

```python
# NVIDIA NIM
NVIDIA_API_KEY   = "your-nvidia-key"
NVIDIA_BASE_URL  = "https://integrate.api.nvidia.com/v1"
NVIDIA_THINK_MODEL = "nvidia/llama-3.1-nemotron-ultra-253b-v1"
NVIDIA_GEN_MODEL   = "meta/llama-3.3-70b-instruct"

# Telegram
TELEGRAM_BOT_TOKEN = "your-bot-token"
TELEGRAM_CHAT_ID   = "your-chat-id"

# GitHub
GITHUB_TOKEN = "your-github-token"
```

### Run

**Step 1 — Build your profile:**
```bash
cd graphs/profile_builder
python graph.py your-github-username
```

**Step 2 — Run the news collector:**
```bash
cd graphs/news_collector
python graph.py
```

**Step 3 — Start the feedback handler and chroma MCP (in a separate terminal):**
```bash
cd graphs/news_collector
python feedback_handler.py

and

cd mcp
python chroma.py
```

Schedule `graph.py` to run every morning via Task Scheduler (Windows) or cron (Linux/Mac).

---

## Tech Stack

Python · LangGraph · LangChain · MCP (FastMCP) · ChromaDB · SQLite · DuckDuckGo Search · Google News RSS · Telegram Bot API · NVIDIA NIM · Groq · Sequential Thinking MCP · sentence-transformers (all-MiniLM-L6-v2) · feedparser · BeautifulSoup4 · httpx
