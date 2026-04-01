"""
graph.py
--------
Wires all collection nodes into a LangGraph StateGraph.

Flow:
  fetch_github
      │
  build_profile
      │
  ┌───┴───┐  (parallel)
  │       │
save_   embed_
sqlite  chroma   ← GitHub profile phase
  │       │
  └───┬───┘
      │
  chat_node        ← terminal interview
      │
  ┌───┴───┐  (parallel)
  │       │
save_   embed_
sqlite  chroma   ← chat fields phase
  │       │
  └───┬───┘
      │
     END
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from langgraph.graph import StateGraph, END
from state import CollectionState
import nest_asyncio
nest_asyncio.apply()

from github_fetch       import fetch_github_node
from build_profile      import build_profile_node
from save_profile_sqlite import save_profile_sqlite_node
from save_profile_chroma import embed_profile_chroma_node
from chat_node               import chat_node
from save_chat_sqlite   import save_chat_sqlite_node
from save_chat_chroma  import embed_chat_chroma_node


def build_graph() -> StateGraph:
    graph = StateGraph(CollectionState)

    # ── Register nodes ────────────────────────────────────────────
    graph.add_node("fetch_github",       fetch_github_node)
    graph.add_node("build_profile",      build_profile_node)
    graph.add_node("save_profile_sqlite", save_profile_sqlite_node)
    graph.add_node("embed_profile_chroma", embed_profile_chroma_node)
    graph.add_node("chat",               chat_node)
    graph.add_node("save_chat_sqlite",   save_chat_sqlite_node)
    graph.add_node("embed_chat_chroma",  embed_chat_chroma_node)

    # ── Edges ─────────────────────────────────────────────────────

    # Linear: fetch → build
    graph.set_entry_point("fetch_github")
    graph.add_edge("fetch_github", "build_profile")

    # Parallel: build → save_sqlite + embed_chroma (GitHub phase)
    graph.add_edge("build_profile", "save_profile_sqlite")
    graph.add_edge("build_profile", "embed_profile_chroma")

    # Both parallel nodes feed into chat
    # LangGraph waits for all incoming edges before running a node
    graph.add_edge("save_profile_sqlite",  "chat")
    graph.add_edge("embed_profile_chroma", "chat")

    # Linear: chat → parallel save + embed (chat phase)
    graph.add_edge("chat", "save_chat_sqlite")
    graph.add_edge("chat", "embed_chat_chroma")

    # Both parallel nodes → END
    graph.add_edge("save_chat_sqlite",  END)
    graph.add_edge("embed_chat_chroma", END)

    return graph.compile()


# ── Entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    username = 'vaslin-dotcom'

    app = build_graph()

    final_state = app.invoke({
        "username":        username,
        "github_raw":      {},
        "profile_fields":  {},
        "chat_fields":     {},
        "chat_history":    [],
        "profile_complete": False,
        "chroma_ids":      [],
        "errors":          [],
    })

    print("\n── Collection Summary ──────────────────────────────────")
    print(f"Chroma vectors saved : {final_state['chroma_ids']}")
    if final_state["errors"]:
        print(f"Errors               : {final_state['errors']}")
    else:
        print("No errors.")
    print("────────────────────────────────────────────────────────\n")