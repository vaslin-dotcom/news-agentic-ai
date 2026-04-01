import time
from langchain_openai import ChatOpenAI
from openai import RateLimitError, InternalServerError
from config import (
    GROQ_API_KEY, GROQ_BASE_URL,
    THINK_MODEL, THINK_MODEL_ALT,
    GENERATION_MODEL, GENERATION_MODEL_ALT,
    NVIDIA_API_KEY, NVIDIA_BASE_URL,
    NVIDIA_THINK_MODEL, NVIDIA_GEN_MODEL
)

def _build_llm(model: str, api_key: str, base_url: str):
    return ChatOpenAI(
        model=model,
        temperature=0,
        api_key=api_key,
        base_url=base_url,
        max_retries=2,
        request_timeout=60
    )

class SmartLLM:
    def __init__(self, primary, alt, fallback):
        self.primary  = primary
        self.alt      = alt
        self.fallback = fallback

    # ── NEW: bind tools to all three LLMs ──
    def bind_tools(self, tools):
        return SmartLLM(
            primary  = self.primary.bind_tools(tools),
            alt      = self.alt.bind_tools(tools),
            fallback = self.fallback.bind_tools(tools)
        )

    def invoke(self, prompt):
        try:
            return self.primary.invoke(prompt)
        except RateLimitError:
            print("[Rate limit] NVIDIA hit 40 RPM — switching to Groq primary")
            time.sleep(3)
            return self._invoke_alt(prompt)
        except InternalServerError:
            print("[503] NVIDIA unavailable — switching to Groq primary")
            time.sleep(3)
            return self._invoke_alt(prompt)

    def _invoke_alt(self, prompt):
        try:
            return self.alt.invoke(prompt)
        except (RateLimitError, InternalServerError):
            print("[Groq primary failed] switching to Groq alt")
            time.sleep(3)
            return self._invoke_fallback(prompt)

    def _invoke_fallback(self, prompt):
        try:
            return self.fallback.invoke(prompt)
        except (RateLimitError, InternalServerError):
            print("[Groq alt failed] waiting 30s and retrying...")
            time.sleep(30)
            return self.fallback.invoke(prompt)
        except Exception as e:
            print(f"[All models failed] {type(e).__name__}: {e}")
            raise


def get_llm(output_schema=None, mode='think'):
    time.sleep(1.5)

    is_generation = (mode == 'generation')

    nvidia_model   = NVIDIA_GEN_MODEL     if is_generation else NVIDIA_THINK_MODEL
    alt_model      = GENERATION_MODEL     if is_generation else THINK_MODEL
    fallback_model = GENERATION_MODEL_ALT if is_generation else THINK_MODEL_ALT

    nvidia_llm   = _build_llm(nvidia_model,   NVIDIA_API_KEY, NVIDIA_BASE_URL)
    alt_llm      = _build_llm(alt_model,      GROQ_API_KEY,   GROQ_BASE_URL)
    fallback_llm = _build_llm(fallback_model, GROQ_API_KEY,   GROQ_BASE_URL)

    if output_schema:
        primary_final  = nvidia_llm.with_structured_output(output_schema)
        alt_final      = alt_llm.with_structured_output(output_schema, method="function_calling")
        fallback_final = fallback_llm.with_structured_output(output_schema)
    else:
        primary_final  = nvidia_llm
        alt_final      = alt_llm
        fallback_final = fallback_llm

    return SmartLLM(
        primary  = primary_final,
        alt      = alt_final,
        fallback = fallback_final
    )