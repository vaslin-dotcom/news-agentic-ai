from typing import TypedDict, Annotated
import operator

class CollectionState(TypedDict):
    # --- Input ---
    username: str

    # --- GitHub raw data ---
    github_raw: dict        # {profile, repos, languages, topics}

    # --- LLM synthesized fields ---
    profile_fields: dict    # professional: job, skills, company, industry etc.
    chat_fields: dict       # personal: locality, habits, interests, exclusions etc.

    # --- Chat loop ---
    chat_history: list[dict]    # [{role, content}, ...]
    profile_complete: bool

    # --- Tracking ---
    chroma_ids: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]