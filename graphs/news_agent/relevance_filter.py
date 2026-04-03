"""
relevance_filter_node.py
------------------------
Filters raw articles by relevance using Chroma vector similarity.
For each article:
  1. Score against profile_vectors (how relevant is it?)
  2. Penalty from feedback_vectors (has user skipped similar before?)
  3. Hard exclusion check (drop if matches news_exclusions)
  4. Threshold: final_score >= 0.3 → KEEP, else DROP

Note on Chroma distances:
  Chroma returns L2 distances, not cosine similarities.
  Lower distance = more similar.
  We use distance < 0.7 as threshold (equivalent to similarity > 0.3).
  Penalty is applied as distance increase.
"""

import json
import asyncio
import re
from langchain_mcp_adapters.client import MultiServerMCPClient
from state import NewsState

MCP_SERVERS = {
    "chroma": {
        "command": "python",
        "args": ["../../mcp/chroma.py"],
        "transport": "stdio"
    }
}


def _unwrap(raw) -> str:
    item = raw[0] if isinstance(raw, list) else raw
    return item["text"] if isinstance(item, dict) else item.text


def _parse_chroma_text(text: str) -> list[dict]:
    """Parse Chroma query_similar text output into list of dicts."""
    chunks = []
    entries = re.split(r"\n\d+\.\s+ID:", text)
    for entry in entries:
        entry = entry.strip()
        if not entry or entry.startswith("Top "):
            continue
        chunk = {}
        chunk["id"] = entry.splitlines()[0].strip()

        sim = re.search(r"Similarity:\s*([-\d.]+)", entry)
        if sim:
            chunk["similarity"] = float(sim.group(1))

        doc = re.search(r"Document:\s*(.+?)(?=\n\s+Metadata:)", entry, re.DOTALL)
        if doc:
            chunk["document"] = doc.group(1).strip().rstrip(".")

        meta = re.search(r"Metadata:\s*(\{.+?\})", entry, re.DOTALL)
        if meta:
            try:
                chunk["metadata"] = json.loads(meta.group(1))
            except json.JSONDecodeError:
                chunk["metadata"] = {}

        if chunk.get("id"):
            chunks.append(chunk)
    return chunks


def _check_exclusions(article: dict, exclusions: list[str]) -> bool:
    """Returns True if article should be EXCLUDED."""
    text = (article.get("title", "") + " " + article.get("summary", "")).lower()
    for excl in exclusions:
        # check both singular and plural forms
        excl_clean = excl.lower().rstrip("s")
        if excl_clean in text:
            return True
    return False


async def _filter(state: NewsState) -> dict:
    articles   = state["raw_articles"]
    profile    = state["profile"]
    exclusions = profile.get("news_exclusions", [])
    if isinstance(exclusions, str):
        exclusions = json.loads(exclusions)
    exclusions = [e.lower() for e in exclusions]

    client = MultiServerMCPClient(MCP_SERVERS)
    tools  = {t.name: t for t in await client.get_tools()}

    filtered   = []
    dropped    = 0

    print(f"\n  Filtering {len(articles)} articles...")

    for i, article in enumerate(articles, 1):
        title   = article.get("title", "")
        summary = article.get("summary", "")
        query_text = f"{title} {summary}"

        # ── Step 1: hard exclusion check ──────────────────────────
        if _check_exclusions(article, exclusions):
            dropped += 1
            continue

        # ── Step 2: profile similarity ────────────────────────────
        try:
            raw = await tools["query_similar"].ainvoke({
                "collection" : "profile_vectors",
                "query_text" : query_text,
                "n_results"  : 3,
            })
            profile_results = _parse_chroma_text(_unwrap(raw))
        except Exception as e:
            profile_results = []

        # get best distance (lowest = most similar)
        # Chroma similarity field = 1 - distance, so distance = 1 - similarity
        # We stored similarity in _parse_chroma_text
        if profile_results:
            # similarity is negative in your Chroma setup (L2 distance issue)
            # use the least negative = most relevant
            best_similarity = max(r.get("similarity", -999) for r in profile_results)
            top_chunks      = profile_results
        else:
            best_similarity = -999
            top_chunks      = []

        # ── Step 3: feedback penalty ──────────────────────────────
        penalty = 0.0
        similar_feedback = []
        try:
            raw = await tools["query_similar"].ainvoke({
                "collection" : "feedback_vectors",
                "query_text" : query_text,
                "n_results"  : 3,
            })
            text = _unwrap(raw)
            if "is empty" not in text:
                feedback_results = _parse_chroma_text(text)
                for fb in feedback_results:
                    meta   = fb.get("metadata", {})
                    signal = meta.get("signal", "")
                    if signal == "skip":
                        penalty = max(penalty, 0.4)
                    elif signal == "irrelevant":
                        penalty = max(penalty, 0.2)
                    similar_feedback.append(fb)
        except Exception:
            pass

        # ── Step 4: threshold check ───────────────────────────────
        # best_similarity ranges roughly -1 to 0 in your Chroma setup
        # -0.3 or higher = reasonably relevant (less negative = more similar)
        final_score = best_similarity - penalty

        THRESHOLD = -0.8   # keep articles with similarity > -0.8

        if final_score < THRESHOLD:
            dropped += 1
            continue

        # ── KEEP ──────────────────────────────────────────────────
        filtered.append({
            **article,
            "similarity_score"        : best_similarity,
            "feedback_penalty"        : penalty,
            "final_score"             : final_score,
            "matching_profile_chunks" : top_chunks,
            "similar_feedback_articles": similar_feedback,
        })

        print(f"  [{i:2}] KEEP  score={final_score:.3f}  {title[:60]}")

    print(f"\n  Kept : {len(filtered)}")
    print(f"  Dropped : {dropped}")

    return {"filtered_articles": filtered}


def relevance_filter_node(state: NewsState) -> dict:
    return asyncio.run(_filter(state))


# ── Test block ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # use the 43 articles from fetch_news test — paste a few here
    sample_articles = [
  {
    "title": "Tech layoffs surpass 45,000 in early 2026",
    "url": "https://www.networkworld.com/article/4143749/tech-layoffs-surpass-45000-in-early-2026.html",
    "summary": "Tech companies have announced more than 45,000 layoffs since the start of 2026, as firms across the ...",
    "source": "Network World",
    "published_at": "2026-03-10T18:30:00+00:00"
  },
  {
    "title": "90 days, 1 lakh layoffs: What happens to workers after the pink slip?",
    "url": "https://www.msn.com/en-in/money/economy/90-days-1-lakh-layoffs-what-happens-to-workers-after-the-pink-slip/ar-AA1ZSi55?ocid=BingNewsVerp",
    "summary": "Nearly one lakh professionals are facing uncertain job prospects as large-scale layoffs continue thr...",
    "source": "India Today on MSN",
    "published_at": "2026-04-01T08:36:08+00:00"
  },
  {
    "title": "Indian shares trail regional peers on $68.6 billion IT rout over AI concerns",
    "url": "https://finance.yahoo.com/news/indian-shares-trail-regional-peers-032238530.html",
    "summary": "Indian shares have lagged their Asian and emerging market peers so far in February, pressured by a $...",
    "source": "Reuters via Yahoo Finance",
    "published_at": "2026-03-04T08:36:09+00:00"
  },
  {
    "title": "Stock Market Today, Jan. 14: Greenland Takeover Chatter Drowned Out by Bank...",
    "url": "https://finance.yahoo.com/news/stock-market-today-jan-14-182646036.html",
    "summary": "This live blog is refreshed periodically throughout the day with the latest updates from the......",
    "source": "TheStreet via Yahoo Finance",
    "published_at": "2026-01-03T08:36:09+00:00"
  },
  {
    "title": "Best Generative AI Libraries for Developers in 2026",
    "url": "https://www.analyticsinsight.net/generative-ai/best-generative-ai-libraries-for-developers-in-2026",
    "summary": "Overview: Generative AI development now involves layered stacks combining training, orchestration, m...",
    "source": "Analytics Insight",
    "published_at": "2026-03-21T08:36:12+00:00"
  },
  {
    "title": "IPL 2026 schedule this week (30th March - 5th April): Full list of matches, dates, timings",
    "url": "https://sports.yahoo.com/articles/ipl-2026-schedule-week-30th-093700450.html",
    "summary": "IPL 2026 kicked off with Virat Kohli leading Royal Challengers Bengaluru to a win, while Rohit Sharm...",
    "source": "Yahoo Sports",
    "published_at": "2026-03-31T08:36:17+00:00"
  },
  {
    "title": "IPL 2026 complete schedule announced: Dates, time, venues, updated squads, live streaming info - All you need to know",
    "url": "https://www.msn.com/en-in/sports/cricket/ipl-2026-complete-schedule-announced-dates-time-venues-updated-squads-live-streaming-info-all-you-need-to-know/ar-AA1ZsnuQ?ocid=BingNewsVerp",
    "summary": "Three-time champions Kolkata Knight Riders are the most affected franchise in IPL 2026, having lost ...",
    "source": "Mint on MSN",
    "published_at": "2026-03-27T08:36:17+00:00"
  },
  {
    "title": "RR IPL 2026 full schedule: Check dates, venues and home-away fixtures of Rajasthan Royals",
    "url": "https://www.msn.com/en-in/sports/cricket/rr-ipl-2026-full-schedule-check-dates-venues-and-home-away-fixtures-of-rajasthan-royals/ar-AA1ZHQsl?ocid=BingNewsVerp",
    "summary": "Rajasthan Royals will kick off their IPL 2026 campaign on March 30 against Chennai Super Kings in Gu...",
    "source": "The Times of India on MSN",
    "published_at": "2026-03-31T08:36:17+00:00"
  },
  {
    "title": "BCCI announces schedule for second phase of TATA IPL 2026",
    "url": "https://www.iplt20.com/news/4256/bcci-announces-schedule-for-second-phase-of-tata-ipl-2026",
    "summary": "Stay updated with the latest IPL news, player announcements, match updates, team changes, and offici...",
    "source": "Indian Premier League Official Website",
    "published_at": "2026-03-27T08:36:17+00:00"
  },
  {
    "title": "IPL 2026 schedule: BCCI reveals full second phase fixtures - check out",
    "url": "https://www.msn.com/en-us/sports/other/ipl-2026-schedule-bcci-reveals-full-second-phase-fixtures-check-out/ar-AA1Zsgig?ocid=BingNewsVerp",
    "summary": "The IPL 2026 season resumes on April 13, with the Board of Control for Cricket in India unveiling th...",
    "source": "The Times of India on MSN",
    "published_at": "2026-03-27T08:36:17+00:00"
  },
  {
    "title": "IPL 2026 weekend schedule: New season kicks off! Check matches, venues, timings",
    "url": "https://www.msn.com/en-us/sports/other/ipl-2026-weekend-schedule-new-season-kicks-off-check-matches-venues-timings/ar-AA1Zw9tn?ocid=BingNewsVerp",
    "summary": "IPL 2026 ignites this weekend with defending champions RCB facing SRH in Bengaluru, a rematch of the...",
    "source": "The Times of India on MSN",
    "published_at": "2026-03-27T08:36:17+00:00"
  },
  {
    "title": "GT IPL 2026 full schedule: Check dates, venues and home-away fixtures of Gujarat Titans",
    "url": "https://www.msn.com/en-in/sports/other/gt-ipl-2026-full-schedule-check-dates-venues-and-home-away-fixtures-of-gujarat-titans/ar-AA1ZOzJ1?ocid=BingNewsVerp",
    "summary": "Gujarat Titans will launch their IPL 2026 campaign against Punjab Kings on March 31st in New Chandig...",
    "source": "The Times of India on MSN",
    "published_at": "2026-04-01T08:36:17+00:00"
  },
  {
    "title": "CSK IPL 2026 full schedule: Check dates, venues and home-away fixtures of Chennai Super Kings",
    "url": "https://sports.yahoo.com/articles/csk-ipl-2026-full-schedule-153800982.html",
    "summary": "Chennai Super Kings will kick off their IPL 2026 campaign against Rajasthan Royals on March 30 in Gu...",
    "source": "Yahoo Sports",
    "published_at": "2026-03-31T08:36:17+00:00"
  },
  {
    "title": "RR IPL 2026 schedule: Full match list, dates & venues",
    "url": "https://www.msn.com/en-in/sports/cricket/rr-ipl-2026-schedule-full-match-list-dates-venues/ar-AA1ZF8r1?ocid=BingNewsVerp",
    "summary": "Rajasthan Royals (RR), the winners of the inaugural edition of IPL in 2008, start their IPL 2026 jou...",
    "source": "Jagran Josh on MSN",
    "published_at": "2026-03-30T08:36:17+00:00"
  },
  {
    "title": "Rajasthan Royals IPL 2026 full schedule: Date, time & venues of RR matches",
    "url": "https://www.msn.com/en-us/sports/general/rajasthan-royals-ipl-2026-full-schedule-date-time-venues-of-rr-matches/ar-AA1Zx0Eb?ocid=BingNewsVerp",
    "summary": "Rajasthan Royals (RR) kickstart their IPL 2026 campaign with a fresh leadership group and a schedule...",
    "source": "Cricket Times on MSN",
    "published_at": "2026-03-28T08:36:17+00:00"
  },
  {
    "title": "IPL 2026 schedule - fixtures and results",
    "url": "https://sports.yahoo.com/articles/ipl-2026-schedule-fixtures-start-154247905.html",
    "summary": "Royal Challengers Bengaluru are the defendingIPLchampions [Getty Images] The fullschedulefor the2026...",
    "source": "BBC via Yahoo Sports",
    "published_at": "2026-04-01T08:36:17+00:00"
  },
  {
    "title": "MI IPL 2026 full schedule: Check dates, venues and home-away fixtures of Mumbai...",
    "url": "https://sports.yahoo.com/articles/mi-ipl-2026-full-schedule-161600406.html",
    "summary": "Mumbai Indians, aiming for a record sixthIPLtitle, kick off their2026campaign against Kolkata......",
    "source": "Willow Sports via Yahoo Sports",
    "published_at": "2026-04-02T16:36:17+00:00"
  },
  {
    "title": "RCB IPL 2026 full schedule: Check dates, venues and home-away fixtures of Royal...",
    "url": "https://sports.yahoo.com/articles/rcb-ipl-2026-full-schedule-153800665.html",
    "summary": "Defending champions Royal Challengers Bengaluru begin theirIPL 2026campaign on March 28 against......",
    "source": "Willow Sports via Yahoo Sports",
    "published_at": "2026-03-27T08:36:17+00:00"
  },
  {
    "title": "IPL 2026 Preview: KKR vs SRH match predicted playing XI, pitch report, weather...",
    "url": "https://sports.yahoo.com/articles/ipl-2026-preview-kkr-vs-120000179.html",
    "summary": "Kolkata Knight Riders host Sunrisers Hyderabad inIPL 2026clash at Eden Gardens. Both teams,......",
    "source": "Willow Sports via Yahoo Sports",
    "published_at": "2026-04-02T11:36:17+00:00"
  },
  {
    "title": "IPL 2026 KKR vs SRH Live Streaming: How to watch Kolkata Knight Riders vs...",
    "url": "https://sports.yahoo.com/articles/ipl-2026-kkr-vs-srh-130900068.html",
    "summary": "Kolkata Knight Riders confront bowling weaknesses, especially spin, as they host Sunrisers......",
    "source": "Willow Sports via Yahoo Sports",
    "published_at": "2026-04-02T13:36:17+00:00"
  },
  {
    "title": "IPL 2026: After snub, RCB star heads to court to secure playing rights",
    "url": "https://sports.yahoo.com/articles/ipl-2026-snub-rcb-star-133100544.html",
    "summary": "Sri Lankan fast bowler Nuwan Thushara has taken Sri Lanka Cricket (SLC) to court, seeking a No......",
    "source": "Willow Sports via Yahoo Sports",
    "published_at": "2026-04-02T13:36:17+00:00"
  },
  {
    "title": "IPL schedule 2026: Full list of fixtures for all 10 teams",
    "url": "https://sports.yahoo.com/articles/ipl-schedule-2026-full-list-092300209.html",
    "summary": "The BCCI has unveiled theIPL 2026 schedule, with the 19th season commencing on March 28 and......",
    "source": "Willow Sports via Yahoo Sports",
    "published_at": "2026-03-27T08:36:17+00:00"
  },
  {
    "title": "How to watch IPL 2026: Cricket live streams, TV channels, start times, full...",
    "url": "https://sports.yahoo.com/articles/watch-ipl-2026-cricket-live-100001101.html",
    "summary": "Indian Premier League2026 schedule: Date, match timings Match No. Date Match Venue Time (IST) Time.....",
    "source": "The Sporting News via Yahoo Sports",
    "published_at": "2026-03-28T08:36:17+00:00"
  },
  {
    "title": "Observational memory cuts AI agent costs 10x and outscores RAG on long-context benchmarks",
    "url": "https://venturebeat.com/data/observational-memory-cuts-ai-agent-costs-10x-and-outscores-rag-on-long",
    "summary": "RAG isn't always fast enough or intelligent enough for modern agentic AI workflows. As teams move fr...",
    "source": "VentureBeat",
    "published_at": "2026-02-09T18:30:00+00:00"
  },
  {
    "title": "Snowflake builds new intelligence that goes beyond RAG to query and aggregate thousands of documents at once",
    "url": "https://venturebeat.com/data-infrastructure/snowflake-builds-new-intelligence-that-goes-beyond-rag-to-query-and",
    "summary": "Enterprise AI has a data problem. Despite billions in investment and increasingly capable language m...",
    "source": "VentureBeat",
    "published_at": "2025-11-03T18:30:00+00:00"
  },
  {
    "title": "Claude Code RAG Masterclass : 8 Steps Proves Structure, Not Size, Drives Real Answers",
    "url": "https://www.geeky-gadgets.com/rag-stack-tutorial-2026/",
    "summary": "What if you could build an AI system that not only retrieves information with pinpoint accuracy but ...",
    "source": "Geeky Gadgets",
    "published_at": "2026-01-29T18:30:00+00:00"
  },
  {
    "title": "Contextual AI launches Agent Composer to turn enterprise RAG into production-ready AI agents",
    "url": "https://venturebeat.com/technology/contextual-ai-launches-agent-composer-to-turn-enterprise-rag-into-production",
    "summary": "In the race to bring artificial intelligence into the enterprise, a small but well-funded startup is...",
    "source": "VentureBeat",
    "published_at": "2026-01-26T18:30:00+00:00"
  },
  {
    "title": "Why Chroma’s New Context-1 20B AI Model is Beating ChatGPT 5 at Search",
    "url": "https://www.geeky-gadgets.com/agentic-loops-retrieval/",
    "summary": "Chroma’s Context-1 is a 20B retrieval-augmented model that beats ChatGPT 5 on search, using agentic ...",
    "source": "Geeky Gadgets",
    "published_at": "2026-04-02T08:36:21+00:00"
  },
  {
    "title": "Tamil OTT Releases: Netflix, JioHotstar, Amazon Prime Video Set April 2026 Line-Up With Youth, Happy Raj & More Movies- Check Your Watchlist",
    "url": "https://www.newsx.com/entertainment/tamil-ott-releases-netflix-jiohotstar-amazon-prime-video-set-april-2026-line-up-with-youth-happy-raj-more-movies-check-your-watchlist-186328/",
    "summary": "As March draws to a close, the focus shifts to an exciting slate of Tamil films gearing up for their...",
    "source": "NewsX",
    "published_at": "2026-03-22T08:36:28+00:00"
  },
  {
    "title": "From youth to Thaai Kizhavi: Top 10 highest-grossing Tamil films of 2026",
    "url": "https://www.msn.com/en-in/entertainment/movies/from-youth-to-thaai-kizhavi-top-10-highest-grossing-tamil-films-of-2026/ar-AA1ZSq1t?ocid=BingNewsVerp",
    "summary": "Small-budget Tamil films are dominating the box office this year, outperforming big-budget releases....",
    "source": "Asianet Newsable on MSN",
    "published_at": "2026-04-01T08:36:28+00:00"
  }
]

    sample_profile = {'company': 'Infosys',
 'daily_habits': 'morning tv news while stretching, push-ups, sit-ups; youtube '
                 'during breakfast, lunch, dinner; 1 hr weight training with '
                 'resistance band at night',
 'github_username': 'vaslin-dotcom',
 'goals': 'Building production-grade AI agents and RAG systems that interact '
          'naturally across voice, text and hardware.',
 'industry': 'IT services',
 'interests': ['conversational AI',
               'assistive hardware',
               'neuromorphic computing',
               'autonomous agents',
               'career-tech fusion'],
 'job': 'AI/ML Engineer',
 'languages_spoken': ['tamil', 'english', 'malayalam', 'telugu'],
 'lifestyle_context': 'system engineer trainee, trained in oracle 21c but not '
                      'interested, building own projects',
 'locality': 'infosys campus',
 'location': 'Mysore',
 'name': 'ANTONY VASLIN',
 'news_exclusions': ['gossips'],
 'news_reading_time': 'morning',
 'personal_interests': ['cricket', 'movies of other languages with subtitles'],
 'professional_context': 'Early-career AI/ML Engineer at Infosys in Mysore who '
                         'has spent the last 18 months rapidly prototyping '
                         'intelligent systems—ranging from voice-enabled RAG '
                         'chatbots and multi-agent debaters to Raspberry-Pi '
                         'smart glasses and neuromorphic drowsiness '
                         'detection—showcasing a clear focus on conversational '
                         'AI, retrieval-augmented generation and edge '
                         'inference.',
 'skills': ['Python',
            'LangGraph',
            'LangChain',
            'RAG pipelines',
            'Telegram bots',
            'OCR',
            'Raspberry Pi',
            'Spiking Neural Networks',
            'multi-agent systems',
            'real-time inference'],
 'tech_stack': ['Python',
                'LangGraph',
                'LangChain',
                'RAG',
                'Telegram API',
                'OCR',
                'Raspberry Pi',
                'Spiking Neural Networks',
                'FastAPI',
                'Pandas']}


    result = relevance_filter_node({
        "profile"          : sample_profile,
        "profile_chunks"   : [],
        "search_queries"   : [],
        "raw_articles"     : sample_articles,
        "filtered_articles": [],
        "scored_articles"  : [],
        "alert_articles"   : [],
        "errors"           : [],
    })

    articles = result["filtered_articles"]
    print(f"\n{'═' * 60}")
    print(f"  FILTERED ARTICLES — {len(articles)} passed")
    print(f"{'═' * 60}\n")
    for art in articles:
        print(f"  Title      : {art['title'][:70]}")
        print(f"  Similarity : {art['similarity_score']:.4f}")
        print(f"  Penalty    : {art['feedback_penalty']}")
        print(f"  Final Score: {art['final_score']:.4f}")
        print(f"  Top Chunks : {[c['id'] for c in art['matching_profile_chunks']]}")
        print()