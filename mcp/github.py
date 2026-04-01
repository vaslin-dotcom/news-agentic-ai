"""
GitHub MCP Server (FastMCP version)
-------------------------------------
Fetches your public GitHub profile data for the profile builder.

Tools exposed:
  - get_user_profile  : bio, location, company, follower count
  - get_repositories  : your public repos with language, stars, description
  - get_languages     : aggregated language usage across all repos
  - get_pinned_topics : topics/tags from your repos (signals interests)

Requires: GITHUB_PAT in .env
  → https://github.com/settings/tokens
  → Scopes needed: read:user, public_repo (no write permissions needed)
"""

import json
import os
from collections import Counter

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("github-mcp")

GITHUB_PAT   = os.getenv("GITHUB_PAT", "")
GITHUB_API   = "https://api.github.com"


# ---------------------------------------------------------------------------
# Low-level helper
# ---------------------------------------------------------------------------

def _gh_get(endpoint: str, params: dict = None) -> dict | list:
    """Authenticated GET to GitHub API."""
    if not GITHUB_PAT:
        raise RuntimeError("GITHUB_PAT is not set in .env")

    headers = {
        "Authorization": f"Bearer {GITHUB_PAT}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(f"{GITHUB_API}/{endpoint}", headers=headers, params=params or {})
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_user_profile(username: str) -> str:
    """
    Fetch a GitHub user's public profile.
    Returns bio, company, location, blog, public repo count, followers.

    :param username: GitHub username e.g. 'torvalds'
    """
    data = _gh_get(f"users/{username}")

    profile = {
        "username":     data.get("login"),
        "name":         data.get("name"),
        "bio":          data.get("bio"),
        "company":      data.get("company"),
        "location":     data.get("location"),
        "blog":         data.get("blog"),
        "public_repos": data.get("public_repos"),
        "followers":    data.get("followers"),
        "following":    data.get("following"),
        "created_at":   data.get("created_at"),
    }
    return json.dumps(profile, indent=2)


@mcp.tool()
def get_repositories(username: str, max_repos: int = 30) -> str:
    """
    Fetch a user's public repositories sorted by most recently updated.
    Returns name, description, primary language, stars, topics.

    :param username: GitHub username
    :param max_repos: Max repos to fetch. Default 30, max 100.
    """
    max_repos = max(1, min(max_repos, 100))
    data = _gh_get(
        f"users/{username}/repos",
        params={"sort": "updated", "per_page": max_repos, "type": "owner"},
    )

    repos = []
    for r in data:
        repos.append({
            "name":        r.get("name"),
            "description": r.get("description"),
            "language":    r.get("language"),
            "stars":       r.get("stargazers_count", 0),
            "forks":       r.get("forks_count", 0),
            "topics":      r.get("topics", []),
            "updated_at":  r.get("updated_at"),
            "url":         r.get("html_url"),
        })

    return json.dumps(repos, indent=2)


@mcp.tool()
def get_languages(username: str, max_repos: int = 30) -> str:
    """
    Aggregate programming language usage across all public repos.
    Returns languages ranked by number of repos using them.
    Useful for inferring technical skills/interests.

    :param username: GitHub username
    :param max_repos: How many repos to scan. Default 30.
    """
    max_repos = max(1, min(max_repos, 100))
    data = _gh_get(
        f"users/{username}/repos",
        params={"sort": "updated", "per_page": max_repos, "type": "owner"},
    )

    language_counts: Counter = Counter()
    for repo in data:
        lang = repo.get("language")
        if lang:
            language_counts[lang] += 1

    ranked = [
        {"language": lang, "repo_count": count}
        for lang, count in language_counts.most_common()
    ]
    return json.dumps(ranked, indent=2)


@mcp.tool()
def get_pinned_topics(username: str, max_repos: int = 30) -> str:
    """
    Collect all unique topics/tags from a user's repos.
    Topics are tags like 'machine-learning', 'fastapi', 'react' that
    signal interests and technology areas.

    :param username: GitHub username
    :param max_repos: How many repos to scan. Default 30.
    """
    max_repos = max(1, min(max_repos, 100))
    data = _gh_get(
        f"users/{username}/repos",
        params={"sort": "updated", "per_page": max_repos, "type": "owner"},
    )

    topic_counts: Counter = Counter()
    for repo in data:
        for topic in repo.get("topics", []):
            topic_counts[topic] += 1

    ranked = [
        {"topic": topic, "repo_count": count}
        for topic, count in topic_counts.most_common()
    ]
    return json.dumps(ranked, indent=2)


# ---------------------------------------------------------------------------
# Direct Python API (called from LangGraph profile builder node)
# ---------------------------------------------------------------------------

def gh_get_profile(username: str) -> dict:
    """Returns clean profile dict directly."""
    data = _gh_get(f"users/{username}")
    return {
        "username":     data.get("login"),
        "name":         data.get("name"),
        "bio":          data.get("bio"),
        "company":      data.get("company"),
        "location":     data.get("location"),
        "public_repos": data.get("public_repos"),
    }


def gh_get_languages(username: str, max_repos: int = 30) -> list[dict]:
    """Returns [{language, repo_count}, ...] sorted by count."""
    data = _gh_get(
        f"users/{username}/repos",
        params={"sort": "updated", "per_page": max_repos, "type": "owner"},
    )
    counts: Counter = Counter()
    for repo in data:
        lang = repo.get("language")
        if lang:
            counts[lang] += 1
    return [{"language": l, "repo_count": c} for l, c in counts.most_common()]


def gh_get_topics(username: str, max_repos: int = 30) -> list[str]:
    """Returns flat list of unique topics across all repos."""
    data = _gh_get(
        f"users/{username}/repos",
        params={"sort": "updated", "per_page": max_repos, "type": "owner"},
    )
    topics: set = set()
    for repo in data:
        for t in repo.get("topics", []):
            topics.add(t)
    return sorted(topics)


def gh_get_repos(username: str, max_repos: int = 30) -> list[dict]:
    """Returns list of repo dicts with name, description, language, stars, topics."""
    data = _gh_get(
        f"users/{username}/repos",
        params={"sort": "updated", "per_page": max_repos, "type": "owner"},
    )
    return [
        {
            "name":        r.get("name"),
            "description": r.get("description"),
            "language":    r.get("language"),
            "stars":       r.get("stargazers_count", 0),
            "topics":      r.get("topics", []),
        }
        for r in data
    ]


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")