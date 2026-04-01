"""
fetch_github_node.py
--------------------
Agentic node — LLM bound with GitHub tools only.
Uses agent_utils to handle SmartLLM fallback with create_react_agent.
"""

import json
from agent_utils import create_agent, invoke_agent
from mcp_tools import get_github_tools
from state import CollectionState

SYSTEM_PROMPT = """
You are a GitHub data collector.
Your job is to fetch ALL available data for the given GitHub username using the tools provided.

You MUST call ALL of these tools — do not skip any:
1. get_user_profile   — fetch bio, company, location, name, followers
2. get_repositories   — fetch all public repos (use max_repos=30)
3. get_languages      — fetch language usage across repos (use max_repos=30)
4. get_pinned_topics  — fetch all topics/tags from repos (use max_repos=30)

After calling all 4 tools, return ONLY a JSON object with keys:
profile, repos, languages, topics.
No explanation, no markdown, no backticks.
"""


def fetch_github_node(state: CollectionState) -> dict:
    tools = get_github_tools()
    agent, smart_llm = create_agent(tools, SYSTEM_PROMPT, mode="think")
    messages = {"messages": [
        {"role": "user", "content": f"Fetch all GitHub data for username: {state['username']}"}
    ]}
    result = invoke_agent(agent, smart_llm, tools, messages, SYSTEM_PROMPT)
    final_message = result["messages"][-1].content

    try:
        github_raw = json.loads(final_message)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", final_message, re.DOTALL)
        github_raw = json.loads(match.group()) if match else {
            "profile": {}, "repos": [], "languages": [], "topics": []
        }
    return {"github_raw": github_raw}