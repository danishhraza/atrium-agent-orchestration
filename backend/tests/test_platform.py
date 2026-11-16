"""Deterministic checks of the platform mechanics, run against a scripted LLM (tests/fake_llm.py).

These guard invariants that must never depend on model behaviour: knowledge-box isolation, task
lifecycle, handoff limits, auth, key handling. Model *quality* is measured separately in evals/.
"""
import random

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlmodel import select

from app import llm, rag, router, security
from app.config import settings
from app.db import Agent, Message, RunLog, Task, User, ensure_user, session
from app.main import app
from app.runtime import runtime
from app.seed import seed_demo
from app.tools import Ctx, calculator, message_agent, web_fetch
from evals.router_bench import evaluate
from tests import fake_llm

KEY = "sk-or-test"


def give_key(user: str, judge: bool = False) -> None:
    ensure_user(user)
    with session() as s:
        me = s.get(User, user)
        me.openrouter_key, me.judge_enabled, me.explore_rate = security.encrypt(KEY), judge, 0.0
        s.add(me)
        s.commit()


async def team(user: str, judge: bool = False) -> dict[str, Agent]:
    give_key(user, judge)
    await seed_demo(user, KEY)
    with session() as s:
        return {a.name: a for a in s.exec(select(Agent).where(Agent.user_id == user)).all()}


def tasks_of(user: str) -> list[Task]:
    with session() as s:
        return list(s.exec(select(Task).where(Task.user_id == user)).all())


# ------------------------------ knowledge isolation ------------------------------

async def test_knowledge_boxes_are_isolated_per_agent():
    t = await team("u-rag")
    query = "Redwood Robotics Lena Fischer workflow automation"

    west_hits = await rag.search(KEY, t["Wren"].id, query)
    assert west_hits[0]["box"] == "Call List West"

    east_hits = await rag.search(KEY, t["Eli"].id, query)
    assert all(h["box"] != "Call List West" for h in east_hits)
    assert not any("Redwood" in h["text"] for h in east_hits)

    assert await rag.search(KEY, t["Jarvis"].id, query) == []  # the orchestrator is assigned no boxes


# ------------------------------ teamwork ------------------------------

async def test_orchestrator_delegates_then_reports_back():
    t = await team("u-mission")
    reply = await runtime.chat(t["Jarvis"], "Draft outreach for the East region call list prospects")
    assert "Delegated to Eli" in reply.content
    await runtime.idle()

    mission, sub = sorted(tasks_of("u-mission"), key=lambda x: x.parent_id is not None)
    assert (mission.status, sub.status) == ("done", "done")
    assert sub.assignee_id == t["Eli"].id and sub.parent_id == mission.id
    assert "Northwind" in sub.result  # Eli answered from the East call list

    with session() as s:
        kinds = [m.kind for m in s.exec(select(Message).where(Message.user_id == "u-mission").order_by(Message.created_at)).all()]
        logs = s.exec(select(RunLog).where(RunLog.user_id == "u-mission")).all()
    assert kinds == ["chat", "delegate", "chat", "result", "report"]
    assert sorted(l.mode for l in logs) == ["chat", "synthesize", "task"]  # every run is costed
    assert all(l.cost > 0 for l in logs)
    assert all(st["state"] == "idle" for st in runtime.states.values())


async def test_lead_delegates_to_its_own_reports():
    t = await team("u-lead")
    await runtime.chat(t["Morgan"], "West region call list outreach prospects")
    await runtime.idle()
    mission, sub = sorted(tasks_of("u-lead"), key=lambda x: x.parent_id is not None)
    assert mission.assignee_id == t["Morgan"].id and mission.status == "done"
    assert sub.assignee_id == t["Wren"].id and sub.created_by == t["Morgan"].id


async def test_handoff_moves_the_task_to_a_new_owner(monkeypatch):
    t = await team("u-handoff")

    async def scripted(key, model, messages, tools=None, name="llm"):
        if "You are Quinn" in messages[0]["content"] and messages[-1]["role"] == "user":
            return fake_llm.call("handoff_task", agent="Eli", notes="This is an East prospect question.")
        return await fake_llm.complete(key, model, messages, tools, name)

    monkeypatch.setattr(llm, "complete", scripted)
    task = Task(user_id="u-handoff", title="Who is the contact at Harbor Logistics?", assignee_id=t["Quinn"].id)
    runtime.save_task(task)
    await runtime.execute(task.id)
    await runtime.idle()

    done = tasks_of("u-handoff")[0]
    assert (done.status, done.assignee_id, done.handoffs) == ("done", t["Eli"].id, 1)
    assert "Sam Okafor" in done.result
    assert "[Handoff from Quinn]" in done.brief


async def test_consult_returns_the_teammates_answer_and_bills_the_asker():
    t = await team("u-consult")
    ctx = Ctx(rt=runtime, creds=llm.creds("u-consult"), agent=t["Quinn"], text="", mode="chat")
    answer = await message_agent(ctx, agent="Wren", message="Which prospect needs SSO and an uptime SLA?")
    assert "Pacific Solar" in answer
    assert ctx.usage["cost"] == pytest.approx(0.002)  # Wren's two turns, charged to Quinn's run
    with session() as s:
        kinds = {m.kind for m in s.exec(select(Message).where(Message.user_id == "u-consult")).all()}
    assert kinds == {"consult", "reply"}


async def test_finished_tasks_are_judged_and_scored():
    t = await team("u-judge", judge=True)
    await runtime.chat(t["Jarvis"], "Draft outreach for the East region call list prospects")
    await runtime.idle()
    with session() as s:
        scores = {l.mode: l.score for l in s.exec(select(RunLog).where(RunLog.user_id == "u-judge")).all()}
    assert scores == {"chat": None, "task": 0.9, "synthesize": 0.9}


async def test_retry_resumes_only_the_failed_subtask(monkeypatch):
    t = await team("u-retry")
    broken = {"on": True}
    eli_runs = []

    async def flaky(key, model, messages, tools=None, name="llm"):
        if "You are Eli" in messages[0]["content"]:
            eli_runs.append(1)
            if broken["on"]:
                raise RuntimeError("OpenrouterException 402: would exceed your available credits")
        return await fake_llm.complete(key, model, messages, tools, name)

    monkeypatch.setattr(llm, "complete", flaky)
    await runtime.chat(t["Jarvis"], "Draft outreach for the East region call list prospects")
    await runtime.idle()
    mission, sub = sorted(tasks_of("u-retry"), key=lambda x: x.parent_id is not None)
    assert sub.status == "failed" and "enough credit" in sub.result

    broken["on"] = False
    c = TestClient(app)
    assert c.post(f"/api/tasks/{mission.id}/retry", headers={"X-User": "u-retry"}).json() == {"restarted": 1}
    await runtime.idle()

    mission, sub = sorted(tasks_of("u-retry"), key=lambda x: x.parent_id is not None)
    assert (mission.status, sub.status) == ("done", "done")
    assert "Northwind" in sub.result and "Northwind" in mission.result  # the report was rewritten from the new result
    assert len(tasks_of("u-retry")) == 2                                 # same tasks resumed, nothing re-delegated
    assert c.post(f"/api/tasks/{sub.id}/retry", headers={"X-User": "u-retry"}).status_code == 422


# ------------------------------ keys and auth ------------------------------

async def test_without_a_key_the_agent_asks_for_one():
    orch = ensure_user("u-nokey")
    reply = await runtime.chat(orch, "hello")
    assert "OpenRouter API key" in reply.content


def test_api_key_is_encrypted_and_never_returned(monkeypatch):
    async def accept(key):
        return key == "sk-or-good"

    monkeypatch.setattr(llm, "check_key", accept)
    c = TestClient(app)
    h = {"X-User": "u-key"}
    assert c.put("/api/settings", json={"openrouter_key": "sk-or-bad"}, headers=h).status_code == 422
    view = c.put("/api/settings", json={"openrouter_key": "sk-or-good", "tier_fast": "openai/gpt-5.6-luna"}, headers=h).json()
    assert view["has_key"] and view["key_hint"] == "good" and view["tiers"]["fast"] == "openai/gpt-5.6-luna"

    assert "sk-or-good" not in c.get("/api/state", headers=h).text
    with session() as s:
        stored = s.get(User, "u-key").openrouter_key
    assert "sk-or-good" not in stored and security.decrypt(stored) == "sk-or-good"
    assert llm.creds("u-key").tiers["fast"] == "openai/gpt-5.6-luna"


def test_login_is_required_when_dev_auth_is_off(monkeypatch):
    monkeypatch.setattr(settings, "dev_auth", False)
    monkeypatch.setattr(settings, "clerk_publishable_key", "pk_test_ZXhhbXBsZS5jbGVyay5hY2NvdW50cy5kZXYk")
    assert security.clerk_issuer() == "https://example.clerk.accounts.dev"
    c = TestClient(app)
    assert c.get("/api/state").status_code == 401
    assert c.get("/api/state", headers={"X-User": "someone"}).status_code == 401  # the dev header is ignored
    assert c.get("/api/state", headers={"Authorization": "Bearer not-a-jwt"}).status_code == 401
    with pytest.raises(HTTPException):
        security.authenticate(None)


def test_api_agent_crud_cycle_guard_and_tenant_isolation():
    give_key("u-api")
    c = TestClient(app)
    h = {"X-User": "u-api"}
    a = c.post("/api/agents", json={"name": "A", "role": "first"}, headers=h).json()
    b = c.post("/api/agents", json={"name": "B", "role": "second", "parent_id": a["id"]}, headers=h).json()
    assert b["parent_id"] == a["id"]
    assert c.patch(f"/api/agents/{a['id']}", json={"parent_id": b["id"]}, headers=h).status_code == 422

    box = c.post("/api/boxes", json={"name": "Notes"}, headers=h).json()
    up = c.post(f"/api/boxes/{box['id']}/files", files={"file": ("n.txt", b"The launch code word is heliotrope.")}, headers=h)
    assert up.json()["n_chunks"] == 1
    c.patch(f"/api/agents/{a['id']}", json={"box_ids": [box["id"]]}, headers=h)

    reply = c.post("/api/chat", json={"agent_id": a["id"], "content": "launch code word"}, headers=h).json()
    assert "heliotrope" in reply["content"]
    other = c.post("/api/chat", json={"agent_id": b["id"], "content": "launch code word"}, headers=h).json()
    assert "heliotrope" not in other["content"]

    assert [x["name"] for x in c.get("/api/state", headers=h).json()["agents"]] == ["Jarvis", "A", "B"]
    stranger = {"X-User": "someone-else"}
    assert c.get("/api/state", headers=stranger).json()["boxes"] == []
    assert c.patch(f"/api/agents/{a['id']}", json={"name": "hijacked"}, headers=stranger).status_code == 404


# ------------------------------ tools ------------------------------

async def test_tool_safety():
    assert await calculator(None, "(1200 * 0.15) / 12") == "15.0"
    assert "Error" in await web_fetch(None, "http://127.0.0.1:8000/api/state")
    assert "Error" in await web_fetch(None, "file:///etc/passwd")


# ------------------------------ routing ------------------------------

EASY = ["What is the capital of {}?", "Translate {} into Spanish", "Give a synonym for {}", "Convert {} to Celsius"]
HARD = ["Design a go-to-market strategy for {} and analyze the trade-offs",
        "Analyze the root cause of {} and plan a migration strategy",
        "Evaluate and compare architectures for {} then forecast the risk"]
NOUNS = ["apples", "the pipeline", "dental clinics", "our CRM", "the east region", "pricing", "onboarding", "churn"]


def synthetic(n: int, seed: int = 0) -> list[dict]:
    """A world where the fast tier passes easy tasks only, and the other tiers pass everything."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        hard = i % 2 == 0
        text = rng.choice(HARD if hard else EASY).format(rng.choice(NOUNS))
        rows.append({"text": text, "hard": hard, "results": {
            "fast": {"score": 0.3 if hard else 0.9, "cost": 1.0},
            "balanced": {"score": 0.9, "cost": 5.0},
            "smart": {"score": 0.95, "cost": 20.0}}})
    return rows


def test_heuristic_router():
    assert router.heuristic("What is the Starter price?") == "fast"
    assert router.heuristic("Compare Growth and Enterprise for a 40 seat team") == "balanced"
    assert router.heuristic("Analyze our pipeline and design a strategy for Q3") == "smart"


def test_learned_router_picks_the_cheapest_tier_that_passes():
    rows = synthetic(80)
    model = router.train_from([(r["text"], t, r["results"][t]["score"]) for r in rows for t in ("fast", "balanced", "smart")])
    assert router.learned(model, "What is the capital of France?") == "fast"
    assert router.learned(model, "Design a strategy for churn and analyze the trade-offs") == "balanced"
    assert router.learned({}, "anything") is None  # no data yet: caller falls back to the heuristic


def test_router_trains_from_a_users_judged_runs_and_is_used():
    give_key("u-router")
    with session() as s:
        for r in synthetic(60, seed=1):
            for tier in ("fast", "balanced"):
                s.add(RunLog(user_id="u-router", agent_id="x", mode="task", tier=tier, routed_by="explore",
                             model="m", text=r["text"], score=r["results"][tier]["score"]))
        me = s.get(User, "u-router")
        me.router_mode = "learned"
        s.add(me)
        s.commit()
    assert router.train("u-router") == {"samples": 120, "tiers_trained": ["balanced", "fast"], "usable": True}

    auto = Agent(user_id="u-router", name="Auto", tier="auto")
    creds = llm.creds("u-router")
    assert router.pick(auto, "Translate hello into Spanish", creds) == (settings.tier_fast, "fast", "learned")
    assert router.pick(auto, "Analyze the root cause of churn and plan a strategy", creds)[1:] == ("balanced", "learned")
    assert router.pick(Agent(user_id="u-router", name="Pinned", tier="smart"), "hi", creds)[1:] == ("smart", "fixed")


def test_benchmark_replay_scores_policies_correctly():
    res = evaluate(synthetic(60, seed=2))
    assert res["always-smart"]["cost_vs_smart"] == 0
    assert res["always-fast"]["pass_rate"] == 0.5          # fast fails every hard task
    assert res["oracle"]["mix"] == {"fast": 30, "balanced": 30, "smart": 0}
    # The cross-validated learned router should land on (or next to) the oracle: all passes, far cheaper.
    assert res["learned"]["pass_rate"] >= 0.95
    assert res["learned"]["cost_vs_smart"] < -0.7
    assert res["learned"]["cost"] <= res["heuristic"]["cost"]
