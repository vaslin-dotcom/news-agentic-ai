"""
graph.py — News Collector Graph
--------------------------------
Wires all 7 nodes into a LangGraph StateGraph.

Flow:
  load_profile
      ↓
  generate_queries
      ↓
  fetch_news
      ↓
  relevance_filter
      ↓
  deep_reasoning
      ↓
  save_node ──┬── embed_node
              └── (both finish)
                      ↓
                  alert_node
                      ↓
                     END

Nodes 6a (save) and 6b (embed) run in parallel via Send API.
alert_node runs after both complete because it needs article_id
from save_node (written back into scored_articles).

Run:
  python graph.py
"""

import operator
import asyncio
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.types import Send

from load_profile    import load_profile_node
from generate_query import generate_queries_node
from fetch_news      import fetch_news_node
from relevance_filter import relevance_filter_node
from deep_thinking  import deep_reasoning_node
from save_sqlite            import save_node
from save_chroma           import embed_node
from telegram_alert           import alert_node
from state                import NewsState

# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(NewsState)

    graph.add_node("load_profile",     load_profile_node)
    graph.add_node("generate_queries", generate_queries_node)
    graph.add_node("fetch_news",       fetch_news_node)
    graph.add_node("relevance_filter", relevance_filter_node)
    graph.add_node("deep_reasoning",   deep_reasoning_node)
    graph.add_node("save_node",        save_node)
    graph.add_node("embed_node",       embed_node)
    graph.add_node("alert_node",       alert_node)

    graph.set_entry_point("load_profile")
    graph.add_edge("load_profile",     "generate_queries")
    graph.add_edge("generate_queries", "fetch_news")
    graph.add_edge("fetch_news",       "relevance_filter")
    graph.add_edge("relevance_filter", "deep_reasoning")

    # parallel: both run after deep_reasoning
    graph.add_edge("deep_reasoning", "save_node")
    graph.add_edge("deep_reasoning", "embed_node")

    # both feed into alert_node — LangGraph waits for both
    graph.add_edge("save_node",  "alert_node")
    graph.add_edge("embed_node", "alert_node")

    graph.add_edge("alert_node", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run():
    print("\n" + "═" * 60)
    print("  News Collector Graph — starting")
    print("═" * 60 + "\n")

    app = build_graph()

    initial_state: NewsState = {
        "profile"          : {},
        "profile_chunks"   : [],
        "search_queries"   : [],
        "raw_articles"     : [],
        "filtered_articles": [],
        "scored_articles"  : [],
        "alert_articles"   : [],
        "errors"           : [],
    }

    final_state = app.invoke(initial_state)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  RUN COMPLETE")
    print("═" * 60)
    print(f"  Queries generated  : {len(final_state.get('search_queries', []))}")
    print(f"  Raw articles       : {len(final_state.get('raw_articles', []))}")
    print(f"  After filter       : {len(final_state.get('filtered_articles', []))}")
    print(f"  Scored             : {len(final_state.get('scored_articles', []))}")
    print(f"  Alerts sent        : {len(final_state.get('alert_articles', []))}")
    errors = final_state.get("errors", [])
    if errors:
        print(f"  Errors             : {len(errors)}")
        for e in errors:
            print(f"    - {e}")
    print()

    return final_state


# ---------------------------------------------------------------------------
# Visualise the graph (optional — prints ASCII or saves PNG)
# ---------------------------------------------------------------------------

def visualise():
    app = build_graph()
    try:
        # requires pygraphviz or mermaid
        png = app.get_graph().draw_mermaid_png()
        with open("news_collector.png", "wb") as f:
            f.write(png)
        print("Graph saved to graph.png")
    except Exception:
        # fallback: print mermaid text
        print(app.get_graph().draw_mermaid())


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if "--visualise" in sys.argv or "--visualize" in sys.argv:
        visualise()
    else:
        run()