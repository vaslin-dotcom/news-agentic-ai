import operator
from typing import Annotated
from typing_extensions import TypedDict


class NewsState(TypedDict):
    profile           : dict          # full flat profile from SQLite
    profile_chunks    : list[dict]    # all profile_vectors from Chroma
    search_queries    : list[str]     # LLM-generated DDG queries
    raw_articles      : list[dict]    # fetched + deduped + unseen
    filtered_articles : list[dict]    # passed relevance filter
    scored_articles   : list[dict]    # after deep reasoning
    alert_articles    : list[dict]    # urgency >= 3
    errors            : Annotated[list[str], operator.add]