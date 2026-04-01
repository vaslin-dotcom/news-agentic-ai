# agent_utils.py
import asyncio
import time
from langgraph.prebuilt import create_react_agent
from openai import RateLimitError, InternalServerError
from llm import get_llm


def create_agent(tools: list, system_prompt: str, mode: str = "think"):
    smart_llm = get_llm(mode=mode)
    agent = create_react_agent(
        smart_llm.primary.bind_tools(tools), tools, prompt=system_prompt
    )
    return agent, smart_llm


async def _invoke_async(agent, smart_llm, tools, messages, system_prompt) -> dict:
    try:
        return await agent.ainvoke(messages)
    except (RateLimitError, InternalServerError):
        print("[Rate limit] switching to alt LLM")
        await asyncio.sleep(3)
        alt_agent = create_react_agent(
            smart_llm.alt.bind_tools(tools), tools, prompt=system_prompt
        )
        try:
            return await alt_agent.ainvoke(messages)
        except (RateLimitError, InternalServerError):
            print("[Alt failed] switching to fallback")
            await asyncio.sleep(3)
            fallback_agent = create_react_agent(
                smart_llm.fallback.bind_tools(tools), tools, prompt=system_prompt
            )
            return await fallback_agent.ainvoke(messages)


def invoke_agent(agent, smart_llm, tools, messages, system_prompt, mode="think") -> dict:
    return asyncio.run(_invoke_async(agent, smart_llm, tools, messages, system_prompt))