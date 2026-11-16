"""Offline eval suite. Run from backend/ (calls real models, so it needs a key):

  OPENROUTER_API_KEY=sk-or-... uv run python -m evals.run [routing] [rag] [mission]

Three suites, each a list of {input, expected_output} items plus evaluators:
  routing  - does the orchestrator hand each request to the right agent?
  rag      - does retrieval find the right document, and never leak a box the agent is not assigned?
  mission  - end to end: does the team finish the request, and how good is the final report (LLM judge)?

With LANGFUSE_* keys set, each suite runs as a Langfuse experiment (traced, scored, comparable
run-over-run in the UI). Without keys it runs locally and prints the same scores.
Exits non-zero when a gate fails, so it can block a merge in CI.
"""
import asyncio
import os
import sys
import uuid
from pathlib import Path

# Evals get their own database so they never touch real user data.
os.environ.setdefault("ATRIUM_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data" / "evals"))
os.environ.setdefault("ATRIUM_DEV_AUTH", "1")  # evals run locally, outside any login

from langfuse import Evaluation  # noqa: E402
from sqlmodel import select  # noqa: E402

from app import rag, security, telemetry  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import Agent, Task, User, ensure_user, init_db, session  # noqa: E402
from app.judge import judge  # noqa: E402
from app.runtime import runtime  # noqa: E402
from app.seed import seed_demo  # noqa: E402

ROUTING = [
    {"input": "Draft outreach emails for the East region call list prospects", "expected_output": "Eli"},
    {"input": "Who is our contact at Pacific Solar in the West region call list?", "expected_output": "Wren"},
    {"input": "Do the math: what does Growth cost for 40 seats with the annual pricing discount?", "expected_output": "Quinn"},
    {"input": "Plan an outreach campaign and coordinate the regional reps", "expected_output": "Morgan"},
    {"input": "Hello, who are you?", "expected_output": "self"},
]

RAG = [
    {"input": {"agent": "Eli", "query": "Harbor Logistics competitor contract"}, "expected_output": "call-list-east.txt"},
    {"input": {"agent": "Wren", "query": "Pacific Solar SSO uptime SLA"}, "expected_output": "call-list-west.txt"},
    {"input": {"agent": "Quinn", "query": "Growth plan price per seat annual discount"}, "expected_output": "product-docs.txt"},
    # Isolation probes: the agent asks about a box it is not assigned. Nothing from that box may come back.
    {"input": {"agent": "Eli", "query": "Redwood Robotics Lena Fischer"}, "expected_output": None, "metadata": {"forbidden": "Call List West"}},
    {"input": {"agent": "Quinn", "query": "Northwind Traders Dana Whitfield"}, "expected_output": None, "metadata": {"forbidden": "Call List East"}},
]

MISSION = [
    {"input": "Draft a short outreach email for each prospect on the East region call list.",
     "expected_output": "One email each for Northwind Traders (Dana Whitfield), Harbor Logistics (Sam Okafor) and Brightline Dental Group (Priya Raman), tailored to their situation."},
    {"input": "Which West region prospect should we pitch the Enterprise plan to, and why?",
     "expected_output": "Pacific Solar Co: 200 reps and they need SSO and an uptime SLA, which only Enterprise includes."},
]


KEY = os.getenv("OPENROUTER_API_KEY", "")


async def fresh_team() -> dict[str, Agent]:
    """Every item runs against its own freshly seeded team, so items cannot affect each other."""
    user = f"eval-{uuid.uuid4().hex[:8]}"
    ensure_user(user)
    with session() as s:
        me = s.get(User, user)
        me.openrouter_key, me.judge_enabled, me.explore_rate = security.encrypt(KEY), False, 0.0
        s.add(me)
        s.commit()
    await seed_demo(user, KEY)
    with session() as s:
        return {a.name: a for a in s.exec(select(Agent).where(Agent.user_id == user)).all()}


def _get(item, key):
    return item[key] if isinstance(item, dict) else getattr(item, key)


# ------------------------------ tasks (what is being evaluated) ------------------------------

async def routing_task(*, item, **_):
    team = await fresh_team()
    orch = next(a for a in team.values() if a.is_orchestrator)
    await runtime.chat(orch, _get(item, "input"))
    await runtime.idle()
    with session() as s:
        subs = s.exec(select(Task).where(Task.user_id == orch.user_id, Task.created_by == orch.id)).all()
    names = {a.id: a.name for a in team.values()}
    return names[subs[0].assignee_id] if subs else "self"


async def rag_task(*, item, **_):
    team = await fresh_team()
    inp = _get(item, "input")
    hits = await rag.search(KEY, team[inp["agent"]].id, inp["query"], k=3)
    return [{"document": h["document"], "box": h["box"]} for h in hits]


async def mission_task(*, item, **_):
    team = await fresh_team()
    orch = next(a for a in team.values() if a.is_orchestrator)
    reply = await runtime.chat(orch, _get(item, "input"))
    await runtime.idle()
    with session() as s:
        mission = s.exec(select(Task).where(Task.user_id == orch.user_id, Task.parent_id == None)).first()  # noqa: E711
    if not mission:  # the orchestrator answered directly
        return {"status": "done", "answer": reply.content}
    return {"status": mission.status, "answer": mission.result}


# ------------------------------ evaluators (how it is scored) ------------------------------

def routed_correctly(*, output, expected_output, **_):
    return Evaluation(name="routing_accuracy", value=float(output == expected_output), comment=f"routed to {output}")


def retrieval_hit(*, output, expected_output, **_):
    if expected_output is None:
        return []
    return Evaluation(name="retrieval_hit", value=float(bool(output) and output[0]["document"] == expected_output))


def no_leak(*, output, metadata=None, **_):
    forbidden = (metadata or {}).get("forbidden")
    if not forbidden:
        return []
    return Evaluation(name="box_leak", value=float(any(h["box"] == forbidden for h in output)))


def completed(*, output, **_):
    return Evaluation(name="completed", value=float(output["status"] == "done"))


async def quality(*, input, output, expected_output, **_):
    value, reason = await judge(KEY, settings.tier_smart, input, output["answer"], reference=expected_output)
    return Evaluation(name="quality", value=value, comment=reason) if value is not None else []


# name -> (items, task, evaluators, gates {score: (comparison, threshold)})
SUITES = {
    "routing": (ROUTING, routing_task, [routed_correctly], {"routing_accuracy": (">=", 0.8)}),
    "rag": (RAG, rag_task, [retrieval_hit, no_leak], {"retrieval_hit": (">=", 0.99), "box_leak": ("<=", 0.0)}),
    "mission": (MISSION, mission_task, [completed, quality], {"completed": (">=", 0.99), "quality": (">=", 0.7)}),
}


async def run_local(items, task, evaluators) -> dict[str, list[float]]:
    scores: dict[str, list[float]] = {}
    for item in items:
        output = await task(item=item)
        for ev in evaluators:
            res = ev(input=item["input"], output=output, expected_output=item.get("expected_output"), metadata=item.get("metadata"))
            res = await res if asyncio.iscoroutine(res) else res
            for e in res if isinstance(res, list) else [res]:
                scores.setdefault(e.name, []).append(float(e.value))
    return scores


def run_langfuse(name, items, task, evaluators) -> dict[str, list[float]]:
    from langfuse import get_client

    result = get_client().run_experiment(
        name=f"atrium-{name}", data=items, task=task, evaluators=evaluators, max_concurrency=2,
        metadata=settings.default_tiers,
    )
    print(result.format())
    scores: dict[str, list[float]] = {}
    for item_result in result.item_results:
        for e in item_result.evaluations:
            scores.setdefault(e.name, []).append(float(e.value))
    return scores


def main() -> int:
    if not KEY:
        print("Set OPENROUTER_API_KEY to run the evals (they call real models).")
        return 2
    init_db()
    wanted = sys.argv[1:] or list(SUITES)
    print(f"models: {settings.default_tiers}")
    print(f"langfuse: {'on - runs recorded as experiments' if telemetry.ENABLED else 'off - local run only'}\n")
    failed = False
    for name in wanted:
        items, task, evaluators, gates = SUITES[name]
        scores = run_langfuse(name, items, task, evaluators) if telemetry.ENABLED else asyncio.run(run_local(items, task, evaluators))
        print(f"== {name} ({len(items)} items)")
        for metric, values in scores.items():
            mean = sum(values) / len(values)
            op, threshold = gates.get(metric, (">=", 0.0))
            ok = mean >= threshold if op == ">=" else mean <= threshold
            failed |= not ok
            print(f"   {metric:18} {mean:.2f}   gate {op} {threshold}   {'PASS' if ok else 'FAIL'}   (n={len(values)})")
    telemetry.flush()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
