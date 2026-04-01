"""
Chroma Vector DB MCP Server (FastMCP version)
----------------------------------------------
Wraps ChromaDB into an MCP-compatible server using FastMCP.
Handles storage and retrieval of embeddings for:
  - profile_vectors  : your interests, goals, skills
  - article_vectors  : processed articles (for deduplication)
  - feedback_vectors : articles you marked irrelevant

Embedding model: all-MiniLM-L6-v2 (free, local, 384-dim)
  → pip install sentence-transformers
  Embedding is handled INTERNALLY — callers just pass text.

Tools exposed:
  - upsert_embedding   : add or update a vector with metadata (text in, stored as vector)
  - query_similar      : find top-k similar vectors by query text
  - delete_embedding   : remove a vector by ID
  - list_collections   : list all Chroma collections
"""

import json
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

mcp = FastMCP("chroma-mcp")

# ---------------------------------------------------------------------------
# Embedding model — loaded once at startup
# ---------------------------------------------------------------------------

_model = SentenceTransformer("all-MiniLM-L6-v2")

def _embed(text: str) -> list[float]:
    return _model.encode(text, normalize_embeddings=True).tolist()

# ---------------------------------------------------------------------------
# ChromaDB client — persists to disk in ./db/chroma/
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIR = str(Path(__file__).parent.parent / "db" / "chroma")
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

_client = chromadb.PersistentClient(
    path=CHROMA_PERSIST_DIR,
    settings=Settings(anonymized_telemetry=False),
)

# Pre-create the three collections we'll always need
for _col in ["profile_vectors", "article_vectors", "feedback_vectors"]:
    _client.get_or_create_collection(name=_col)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_collection(name: str) -> chromadb.Collection:
    return _client.get_or_create_collection(name=name)


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def upsert_embedding(
    collection: str,
    id: str,
    document: str,
    metadata: dict = None,
) -> str:
    """
    Add or update a vector embedding in a Chroma collection.
    Embedding is generated internally from the document text.
    If the ID already exists it will be overwritten.

    :param collection: One of 'profile_vectors', 'article_vectors', 'feedback_vectors'
    :param id: Unique identifier for this vector (e.g. 'username:professional')
    :param document: The text to embed and store
    :param metadata: Optional dict of extra metadata (e.g. {"type": "professional", "username": "..."})
    """
    embedding = _embed(document)
    col = _get_collection(collection)
    col.upsert(
        ids=[id],
        embeddings=[embedding],
        documents=[document],
        metadatas=[metadata or {}],
    )
    return f"Upserted '{id}' into '{collection}'. Collection now has {col.count()} vectors."


@mcp.tool()
def query_similar(
    collection: str,
    query_text: str,
    n_results: int = 5,
    where: dict = None,
) -> str:
    """
    Find the top-k most similar vectors to a query text.
    Query is embedded internally before searching.
    Returns IDs, similarity scores, documents, and metadata.

    :param collection: Collection to search in
    :param query_text: The text to search for similar vectors
    :param n_results: Number of nearest neighbours to return. Default 5.
    :param where: Optional metadata filter e.g. {"type": "professional"}
    """
    col = _get_collection(collection)
    count = col.count()

    if count == 0:
        return f"Collection '{collection}' is empty."

    n_results = min(n_results, count)
    query_embedding = _embed(query_text)

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = col.query(**kwargs)

    lines = [f"Top {n_results} results from '{collection}':\n"]
    for i, (rid, dist, doc, meta) in enumerate(zip(
        results["ids"][0],
        results["distances"][0],
        results["documents"][0],
        results["metadatas"][0],
    ), 1):
        similarity = round(1 - dist, 4)
        lines.append(
            f"{i}. ID: {rid}\n"
            f"   Similarity: {similarity}\n"
            f"   Document: {doc[:150]}...\n"
            f"   Metadata: {json.dumps(meta)}\n"
        )

    return "\n".join(lines)


@mcp.tool()
def delete_embedding(collection: str, id: str) -> str:
    """
    Remove a vector from a collection by its ID.

    :param collection: Collection name
    :param id: ID of the vector to delete
    """
    _get_collection(collection).delete(ids=[id])
    return f"Deleted '{id}' from '{collection}'."


@mcp.tool()
def list_collections() -> str:
    """
    List all Chroma collections and how many vectors are in each.
    """
    all_cols = _client.list_collections()
    if not all_cols:
        return "No collections found."

    lines = ["Collections:\n"]
    for col_obj in all_cols:
        col = _client.get_collection(col_obj.name)
        lines.append(f"  - {col_obj.name}: {col.count()} vectors")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Direct Python API (called from LangGraph nodes — no MCP protocol overhead)
# ---------------------------------------------------------------------------

def upsert(collection: str, doc_id: str, document: str, metadata: dict = None):
    """Direct upsert — generates embedding internally, bypasses MCP protocol."""
    col = _get_collection(collection)
    col.upsert(
        ids=[doc_id],
        embeddings=[_embed(document)],
        documents=[document],
        metadatas=[metadata or {}],
    )


def query(collection: str, query_text: str, n_results: int = 5, where: dict = None) -> list[dict]:
    """
    Direct query by text — embeds internally, returns clean list of dicts:
    [{"id": ..., "similarity": ..., "document": ..., "metadata": ...}, ...]
    """
    col = _get_collection(collection)
    count = col.count()
    if count == 0:
        return []

    n_results = min(n_results, count)
    kwargs = {
        "query_embeddings": [_embed(query_text)],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = col.query(**kwargs)
    return [
        {
            "id": rid,
            "similarity": round(1 - dist, 4),
            "document": doc,
            "metadata": meta,
        }
        for rid, dist, doc, meta in zip(
            results["ids"][0],
            results["distances"][0],
            results["documents"][0],
            results["metadatas"][0],
        )
    ]


def delete(collection: str, doc_id: str):
    """Direct delete by ID."""
    _get_collection(collection).delete(ids=[doc_id])


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")