"""A scripted stand-in for llm.complete, used only by the test-suite.

It follows fixed rules so the platform mechanics (delegation, handoff, isolation, lifecycle) can be
tested deterministically and for free. The application itself has no mock mode.
"""
import json
import re

USAGE = {"input": 10, "output": 5, "cost": 0.001}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{4,}", text.lower()))


def say(content: str) -> dict:
    return {"content": content, "tool_calls": [], "raw": {"role": "assistant", "content": content}, "usage": dict(USAGE)}


def call(tool: str, **args) -> dict:
    tc = {"id": "call_1", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
    return {"content": "", "tool_calls": [{"id": "call_1", "name": tool, "arguments": args}],
            "raw": {"role": "assistant", "content": None, "tool_calls": [tc]}, "usage": dict(USAGE)}


async def complete(key, model, messages, tools=None, name="llm") -> dict:
    names = {t["function"]["name"] for t in tools or []}
    system = messages[0]["content"]
    if name == "judge":
        return say('{"score": 0.9, "reason": "scripted"}')
    last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
    text = messages[last_user]["content"]
    results = [m["content"] for m in messages[last_user + 1:] if m["role"] == "tool"]
    if results:  # second turn: wrap up from what the tools returned
        return say("[fake] " + " | ".join(r[:400] for r in results))
    if "team_status" in names and re.search(r"\b(status|update|pending|progress)\b", text.lower()):
        return call("team_status")
    if "delegate_task" in names:
        roster = re.findall(r"^- (.+?) \(id: (\w+)\) — (.*)$", system, re.M)
        best = max(((len(_words(text) & _words(f"{n} {role}")), aid) for n, aid, role in roster), default=(0, ""))
        if best[0] > 0:
            return call("delegate_task", agent=best[1], title=text[:60], brief=text)
    if "search_knowledge" in names:
        return call("search_knowledge", query=text[:300])
    return say(f"[fake] {text[:400]}")
