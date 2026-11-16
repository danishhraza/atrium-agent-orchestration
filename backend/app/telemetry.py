"""Thin Langfuse wrapper. With no LANGFUSE_* keys every call here is a no-op, so the app runs untraced.

Trace model (see docs/LANGFUSE_GUIDE.md):
  session  = one mission (a user request and everything it fans out into)
  trace    = one agent run (orchestrate / task / synthesize / chat)
  agent    observation = the run, and nested consults of other agents
  generation observation = each LLM call (model, tokens, cost)
  tool / retriever observation = each tool call / knowledge-box search
"""
import os
from contextlib import contextmanager

ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
_client = None
if ENABLED:
    from langfuse import get_client, propagate_attributes

    _client = get_client()


class _Noop:
    def update(self, **_):
        pass


@contextmanager
def observation(name: str, as_type: str = "span", **kwargs):
    if not _client:
        yield _Noop()
        return
    with _client.start_as_current_observation(name=name, as_type=as_type, **kwargs) as obs:
        yield obs


@contextmanager
def trace_attrs(*, user_id: str, session_id: str, tags: list[str], trace_name: str, metadata: dict | None = None):
    """Attributes set here are stamped on every observation created inside the block."""
    if not _client:
        yield
        return
    meta = {k: str(v)[:200] for k, v in (metadata or {}).items()}
    with propagate_attributes(user_id=user_id, session_id=session_id, tags=tags, trace_name=trace_name, metadata=meta):
        yield


def current_trace_id() -> str | None:
    return _client.get_current_trace_id() if _client else None


def score(trace_id: str | None, name: str, value: float | str, comment: str | None = None) -> None:
    if _client and trace_id:
        _client.create_score(trace_id=trace_id, name=name, value=value, comment=comment)


def prompt(name: str, fallback: str) -> str:
    """Langfuse prompt management: edit prompts in the UI without redeploying. Falls back to the code default."""
    if not _client:
        return fallback
    try:
        return _client.get_prompt(name, fallback=fallback, cache_ttl_seconds=60).prompt
    except Exception:
        return fallback


def flush() -> None:
    if _client:
        _client.flush()
