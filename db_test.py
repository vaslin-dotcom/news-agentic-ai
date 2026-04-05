"""
test_dbs.py
-----------
Tests SQLite and Chroma DBs via MCP tools.
Run directly: python test_dbs.py
"""

import asyncio
import json
import sys
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

BASE_DIR = Path(__file__).parent
MCP_SCRIPTS = {
    "sqlite": BASE_DIR / "mcp" / "sqlite.py",
    "chroma": BASE_DIR / "mcp" / "chroma.py",
}

for name, path in MCP_SCRIPTS.items():
    if not path.exists():
        print(f"Error: {name} script not found at {path}")
        sys.exit(1)

MCP_SERVERS = {
    "sqlite": {
        "command": sys.executable,
        "args": [str(MCP_SCRIPTS["sqlite"])],
        "transport": "stdio"
    },
    "chroma": {
        "url"      : "http://localhost:8001/mcp/",
        "transport": "streamable_http"  # ← change transport too
    }
}


async def main():
    client = MultiServerMCPClient(MCP_SERVERS)
    try:
        tools = {t.name: t for t in await client.get_tools()}

        # ── SQLite — Profile Table ──────────────────────────────────────
        print("\n" + "═" * 60)
        print("  SQLite — Profile Table")
        print("═" * 60)

        result = await tools["get_profile"].ainvoke({})
        # result is a list of message dicts with 'type', 'text', etc.
        # Extract text from the first message
        if isinstance(result, list) and len(result) > 0 and 'text' in result[0]:
            content = result[0]['text']
            # Try to parse as JSON (profile dictionary)
            try:
                profile = json.loads(content)
                if not profile:
                    print("  ⚠ No profile data found.")
                else:
                    for key, value in profile.items():
                        # Try to parse value as JSON if it's a string
                        if isinstance(value, str):
                            try:
                                parsed = json.loads(value)
                                # Pretty print with indentation
                                value_str = json.dumps(parsed, indent=2)
                                # Replace newlines with newline + spaces for indentation
                                value_str = value_str.replace('\n', '\n      ')
                                print(f"  {key}: {value_str}")
                            except (json.JSONDecodeError, TypeError):
                                print(f"  {key}: {value}")
                        else:
                            print(f"  {key}: {value}")
            except json.JSONDecodeError:
                print(f"  Raw content: {content}")
        else:
            print(f"  Unexpected response structure: {result}")

        # ── Chroma — Collections ──────────────────────────────────
        print("\n" + "═" * 60)
        print("  Chroma — Collections")
        print("═" * 60)

        result = await tools["list_collections"].ainvoke({})
        if isinstance(result, list) and len(result) > 0 and 'text' in result[0]:
            content = result[0]['text']
            # The text is already formatted nicely
            print(content)
        else:
            print(result)

        # ── Chroma — profile_vectors ──────────────────────────────
        print("\n" + "═" * 60)
        print("  Chroma — profile_vectors (query: 'professional background skills interests')")
        print("═" * 60)

        result = await tools["query_similar"].ainvoke({
            "collection": "profile_vectors",
            "query_text": "professional background skills interests",
            "n_results": 10,
        })
        if isinstance(result, list) and len(result) > 0 and 'text' in result[0]:
            content = result[0]['text']
            print(content)
        else:
            print(result)

    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        if hasattr(client, 'close') and callable(client.close):
            try:
                await client.close()
            except:
                pass


if __name__ == "__main__":
    asyncio.run(main())