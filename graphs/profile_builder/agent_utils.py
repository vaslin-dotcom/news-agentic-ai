"""
agent_utils.py
--------------
Creates and invokes ReAct agents with SmartLLM fallback.
All invocations are async (ainvoke) to support async MCP tools.
"""

import asyncio
from langgraph.prebuilt import create_react_agent
from openai import RateLimitError, InternalServerError
from llm import get_llm


def create_agent(tools: list, system_prompt: str, mode: str = "think"):
    smart_llm = get_llm(mode=mode)
    agent = create_react_agent(
        smart_llm.primary.bind_tools(tools), tools, prompt=system_prompt
    )
    return agent, smart_llm


async def invoke_agent_async(
    agent, smart_llm, tools, messages, system_prompt, mode="think"
) -> dict:
    try:
        return await agent.ainvoke(messages)

    except (RateLimitError, InternalServerError):
        print("[Rate limit / 503] Primary failed — switching to alt LLM")
        await asyncio.sleep(3)
        alt_agent = create_react_agent(
            smart_llm.alt.bind_tools(tools), tools, prompt=system_prompt
        )
        try:
            return await alt_agent.ainvoke(messages)
        except (RateLimitError, InternalServerError):
            print("[Alt failed] switching to fallback LLM")
            await asyncio.sleep(3)
            fallback_agent = create_react_agent(
                smart_llm.fallback.bind_tools(tools), tools, prompt=system_prompt
            )
            try:
                return await fallback_agent.ainvoke(messages)
            except (RateLimitError, InternalServerError):
                print("[All models failed] waiting 30s...")
                await asyncio.sleep(30)
                return await fallback_agent.ainvoke(messages)