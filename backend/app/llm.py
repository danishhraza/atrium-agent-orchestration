"""Model access: every call goes to OpenRouter through LiteLLM, using the calling user's own key.

Messages and tools use the OpenAI chat format. Model ids are OpenRouter slugs such as
"anthropic/claude-sonnet-5.5" or "openai/gpt-5.6-terra".
"""
import json
from dataclasses import dataclass

import httpx
import litellm

from . import security, telemetry
from .config import TIERS, settings
from .db import User, session

litellm.drop_params = True  # ignore params a given model does not support instead of erroring
OPENROUTER = "https://openrouter.ai/api/v1"


class MissingKey(Exception):
    """The user has not saved an OpenRouter key yet."""


@dataclass
class Creds:
    """What a run needs to know about the user it runs for."""
    user_id: str
    key: str
    tiers: dict[str, str]
    judge_enabled: bool
    router_mode: str
    explore_rate: float


def creds(user_id: str) -> Creds:
    with session() as s:
        user = s.get(User, user_id)
    key = security.decrypt(user.openrouter_key) if user and user.openrouter_key else ""
    if not key:
        raise MissingKey("Add your OpenRouter API key in Settings to wake your team up.")
    tiers = {t: getattr(user, f"tier_{t}") or settings.default_tiers[t] for t in TIERS}
    return Creds(user_id, key, tiers, user.judge_enabled, user.router_mode, user.explore_rate)


async def check_key(key: str) -> bool:
    """Ask OpenRouter whether a key is valid, before we store it."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{OPENROUTER}/key", headers={"Authorization": f"Bearer {key}"})
    return resp.status_code == 200


async def complete(key: str, model: str, messages: list[dict], tools: list[dict] | None = None, name: str = "llm") -> dict:
    """One LLM turn. Returns {content, tool_calls: [{id, name, arguments}], raw, usage}.

    `raw` is the assistant message to append to history unchanged: it carries provider-specific
    fields (such as reasoning blocks) that must be replayed on the next turn of a tool loop.
    `usage` is {input, output, cost} with cost in USD as billed by OpenRouter.
    """
    with telemetry.observation(name, as_type="generation", model=model, input=messages) as gen:
        resp = await litellm.acompletion(
            model=f"openrouter/{model}", api_key=key, messages=messages, tools=tools or None,
            max_tokens=8000, num_retries=2,
        )
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_invalid_json": tc.function.arguments}
            calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})

        u = getattr(resp, "usage", None)
        usage = {"input": getattr(u, "prompt_tokens", 0) or 0, "output": getattr(u, "completion_tokens", 0) or 0,
                 "cost": getattr(u, "cost", None)}
        if usage["cost"] is None:  # OpenRouter did not report it: fall back to LiteLLM's price map
            try:
                usage["cost"] = litellm.completion_cost(completion_response=resp)
            except Exception:
                usage["cost"] = 0.0
        raw = msg.model_dump(exclude_none=True)
        gen.update(output=raw, usage_details={"input": usage["input"], "output": usage["output"]},
                   cost_details={"total": float(usage["cost"])})
        return {"content": msg.content or "", "tool_calls": calls, "raw": raw, "usage": usage}


async def embed(key: str, texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    async with httpx.AsyncClient(timeout=60) as client:
        for i in range(0, len(texts), 64):
            resp = await client.post(f"{OPENROUTER}/embeddings", headers={"Authorization": f"Bearer {key}"},
                                     json={"model": settings.embedding_model, "input": texts[i:i + 64]})
            resp.raise_for_status()
            out += [d["embedding"] for d in sorted(resp.json()["data"], key=lambda d: d["index"])]
    return out


def friendly_error(e: Exception) -> str:
    """Turn provider errors into something a user can act on."""
    if isinstance(e, MissingKey):
        return str(e)
    if isinstance(e, litellm.AuthenticationError):
        return "OpenRouter rejected your API key. Check it in Settings."
    if "402" in str(e) or "credits" in str(e).lower():
        return ("Your OpenRouter key does not have enough credit for this request. Add credits at "
                "openrouter.ai/credits, then press Retry on the failed task (or ask to try again).")
    if isinstance(e, litellm.RateLimitError):
        return "OpenRouter rate-limited this request, or the key is out of credit. Try again shortly."
    if isinstance(e, litellm.NotFoundError):
        return "That model id was not found on OpenRouter. Check the model tiers in Settings."
    return f"Something went wrong: {type(e).__name__}: {str(e)[:300]}"
