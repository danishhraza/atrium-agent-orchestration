"""HTTP + WebSocket API. Every route resolves the caller through Clerk (security.current_user)
and every query is scoped to that user id."""
import asyncio
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import delete, select

from . import llm, rag, router, security, telemetry
from .config import TIERS, settings
from .db import Agent, AgentBox, Box, Chunk, Document, Message, RunLog, Task, User, ensure_user, init_db, session
from .runtime import runtime
from .seed import seed_demo
from .tools import OPTIONAL

MAX_UPLOAD = 20 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.dev_auth and not settings.clerk_publishable_key:
        raise RuntimeError("Set ATRIUM_CLERK_PUBLISHABLE_KEY, or ATRIUM_DEV_AUTH=1 for local development.")
    security.encrypt("startup-check")  # fail now, not on first key save, if ATRIUM_SECRET_KEY is missing
    init_db()
    yield
    telemetry.flush()


app = FastAPI(title="Atrium", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins.split(","), allow_methods=["*"], allow_headers=["*"])


def current_user(user: str = Depends(security.current_user)) -> str:
    ensure_user(user)
    return user


def owned(model, id_: str, user_id: str):
    with session() as s:
        obj = s.get(model, id_)
    if not obj or obj.user_id != user_id:
        raise HTTPException(404, f"{model.__name__} not found")
    return obj


class AgentIn(BaseModel):
    name: str | None = None
    role: str | None = None
    instructions: str | None = None
    tier: str | None = None
    model: str | None = None
    tools: list[str] | None = None
    parent_id: str | None = None
    box_ids: list[str] | None = None


class BoxIn(BaseModel):
    name: str
    description: str = ""


class ChatIn(BaseModel):
    agent_id: str | None = None  # None = the orchestrator
    content: str


class FeedbackIn(BaseModel):
    value: int  # 1 = good, 0 = bad


class SettingsIn(BaseModel):
    openrouter_key: str | None = None   # "" removes the saved key
    tier_fast: str | None = None
    tier_balanced: str | None = None
    tier_smart: str | None = None
    judge_enabled: bool | None = None
    router_mode: str | None = None
    explore_rate: float | None = None


def _settings_view(user: User) -> dict:
    """What the client may see. The API key itself never leaves the server."""
    return {"has_key": bool(user.openrouter_key), "key_hint": user.key_hint,
            "tiers": {t: getattr(user, f"tier_{t}") or settings.default_tiers[t] for t in TIERS},
            "judge_enabled": user.judge_enabled, "router_mode": user.router_mode, "explore_rate": user.explore_rate}


def _apply(agent: Agent, data: AgentIn, user_id: str) -> None:
    fields = data.model_dump(exclude_unset=True, exclude={"box_ids"})
    if fields.get("tier") and fields["tier"] not in ("auto", *TIERS):
        raise HTTPException(422, "tier must be auto, fast, balanced or smart")
    with session() as s:
        if "parent_id" in fields and fields["parent_id"]:
            seen, cur = {agent.id}, s.get(Agent, fields["parent_id"])
            if not cur or cur.user_id != user_id or cur.is_orchestrator:
                fields["parent_id"] = None  # reporting to the orchestrator is the default, stored as None
            while cur and not cur.is_orchestrator:  # walk up: refuse reporting cycles
                if cur.id in seen:
                    raise HTTPException(422, "that would create a reporting cycle")
                seen.add(cur.id)
                cur = s.get(Agent, cur.parent_id) if cur.parent_id else None
        for k, v in fields.items():
            setattr(agent, k, v or None if k in ("model", "parent_id") else v)
        s.merge(agent)
        if data.box_ids is not None:
            mine = set(s.exec(select(Box.id).where(Box.user_id == user_id)).all())
            s.exec(delete(AgentBox).where(AgentBox.agent_id == agent.id))
            for bid in set(data.box_ids) & mine:
                s.add(AgentBox(agent_id=agent.id, box_id=bid))
        s.commit()


@app.get("/api/state")
def state(user: str = Depends(current_user)):
    with session() as s:
        agents = s.exec(select(Agent).where(Agent.user_id == user).order_by(Agent.created_at)).all()
        boxes = s.exec(select(Box).where(Box.user_id == user)).all()
        docs = s.exec(select(Document).where(Document.user_id == user)).all()
        links = s.exec(select(AgentBox).where(AgentBox.agent_id.in_([a.id for a in agents]))).all()
        tasks = s.exec(select(Task).where(Task.user_id == user).order_by(Task.created_at.desc()).limit(100)).all()
        msgs = s.exec(select(Message).where(Message.user_id == user).order_by(Message.created_at.desc()).limit(200)).all()
        me = s.get(User, user)
    idle = {"state": "idle", "detail": "", "peer": None}
    return {
        "agents": [{**a.model_dump(mode="json"), **runtime.states.get(a.id, idle),
                    "box_ids": [l.box_id for l in links if l.agent_id == a.id]} for a in agents],
        "boxes": [{**b.model_dump(), "documents": [d.model_dump(mode="json") for d in docs if d.box_id == b.id]} for b in boxes],
        "tasks": [t.model_dump(mode="json") for t in tasks],
        "messages": [m.model_dump(mode="json") for m in reversed(msgs)],
        "settings": _settings_view(me),
        "config": {"langfuse": telemetry.ENABLED, "optional_tools": OPTIONAL, "embedding_model": settings.embedding_model},
    }


@app.post("/api/agents")
def create_agent(data: AgentIn, user: str = Depends(current_user)):
    if not data.name:
        raise HTTPException(422, "name is required")
    agent = Agent(user_id=user, name=data.name)
    _apply(agent, data, user)
    return agent


@app.patch("/api/agents/{agent_id}")
def update_agent(agent_id: str, data: AgentIn, user: str = Depends(current_user)):
    agent = owned(Agent, agent_id, user)
    _apply(agent, data, user)
    return agent


@app.delete("/api/agents/{agent_id}")
def delete_agent(agent_id: str, user: str = Depends(current_user)):
    agent = owned(Agent, agent_id, user)
    if agent.is_orchestrator:
        raise HTTPException(422, "the orchestrator cannot be deleted")
    with session() as s:
        for child in s.exec(select(Agent).where(Agent.parent_id == agent_id)).all():
            child.parent_id = agent.parent_id
            s.add(child)
        s.exec(delete(AgentBox).where(AgentBox.agent_id == agent_id))
        s.delete(s.get(Agent, agent_id))
        s.commit()
    return {"ok": True}


@app.post("/api/boxes")
def create_box(data: BoxIn, user: str = Depends(current_user)):
    box = Box(user_id=user, name=data.name, description=data.description)
    with session() as s:
        s.add(box)
        s.commit()
    return box


@app.delete("/api/boxes/{box_id}")
def delete_box(box_id: str, user: str = Depends(current_user)):
    owned(Box, box_id, user)
    with session() as s:
        for model in (Chunk, Document, AgentBox):
            s.exec(delete(model).where(model.box_id == box_id))
        s.delete(s.get(Box, box_id))
        s.commit()
    return {"ok": True}


@app.post("/api/boxes/{box_id}/files")
async def upload(box_id: str, file: UploadFile, user: str = Depends(current_user)):
    owned(Box, box_id, user)
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "file is larger than 20 MB")
    try:
        key = "" if settings.embedding_model == "local" else llm.creds(user).key
        return await rag.ingest(key, user, box_id, file.filename or "upload.txt", data)
    except (ValueError, llm.MissingKey) as e:
        raise HTTPException(422, str(e))
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Embedding request to OpenRouter failed: {e}")


@app.post("/api/chat")
async def chat(data: ChatIn, user: str = Depends(current_user)):
    agent = owned(Agent, data.agent_id, user) if data.agent_id else ensure_user(user)
    return await runtime.chat(agent, data.content)


@app.post("/api/tasks/{task_id}/retry")
async def retry_task(task_id: str, user: str = Depends(current_user)):
    task = owned(Task, task_id, user)
    with session() as s:
        failed_child = s.exec(select(Task).where(Task.parent_id == task_id, Task.status == "failed")).first()
    if task.status != "failed" and not failed_child:  # a mission can finish "done" while reporting a failed subtask
        raise HTTPException(422, "nothing failed here to retry")
    return {"restarted": runtime.retry(task)}


@app.post("/api/messages/{message_id}/feedback")
def feedback(message_id: str, data: FeedbackIn, user: str = Depends(current_user)):
    msg = owned(Message, message_id, user)
    telemetry.score(msg.trace_id, "user_feedback", float(data.value))
    return {"ok": True, "recorded": bool(telemetry.ENABLED and msg.trace_id)}


@app.post("/api/demo")
async def demo(user: str = Depends(current_user)):
    try:
        key = "" if settings.embedding_model == "local" else llm.creds(user).key
        return {"created": await seed_demo(user, key)}
    except llm.MissingKey as e:
        raise HTTPException(422, str(e))


@app.get("/api/public")
def public_config():
    """Unauthenticated: what the login screen needs. The Clerk publishable key is public by design."""
    return {"clerk_publishable_key": settings.clerk_publishable_key, "dev_auth": settings.dev_auth}


@app.put("/api/settings")
async def save_settings(data: SettingsIn, user: str = Depends(current_user)):
    fields = data.model_dump(exclude_unset=True)
    if fields.get("router_mode") and fields["router_mode"] not in ("heuristic", "learned"):
        raise HTTPException(422, "router_mode must be heuristic or learned")
    if "explore_rate" in fields and not 0 <= fields["explore_rate"] <= 0.5:
        raise HTTPException(422, "explore_rate must be between 0 and 0.5")
    key = fields.pop("openrouter_key", None)
    if key and not await llm.check_key(key.strip()):
        raise HTTPException(422, "OpenRouter did not accept that key.")
    with session() as s:
        me = s.get(User, user)
        if key is not None:
            key = key.strip()
            me.openrouter_key = security.encrypt(key) if key else ""
            me.key_hint = key[-4:] if key else ""
        for k, v in fields.items():
            setattr(me, k, (v or None) if k.startswith("tier_") else v)
        s.add(me)
        s.commit()
        return _settings_view(me)


_models: dict = {"at": 0.0, "data": []}


@app.get("/api/models")
async def models(user: str = Depends(current_user)):
    """Tool-capable OpenRouter models with prices, for the tier pickers. Cached for an hour."""
    if time.time() - _models["at"] > 3600:
        async with httpx.AsyncClient(timeout=20) as client:
            data = (await client.get(f"{llm.OPENROUTER}/models")).json()["data"]
        _models["data"] = [
            {"id": m["id"], "name": m["name"], "input": float(m["pricing"]["prompt"]) * 1e6,
             "output": float(m["pricing"]["completion"]) * 1e6}
            for m in data if "tools" in (m.get("supported_parameters") or [])]
        _models["at"] = time.time()
    return _models["data"]


@app.get("/api/router")
def router_stats(user: str = Depends(current_user)):
    """Spend and judged quality, split by tier and by how each run was routed."""
    def grouped(col):
        with session() as s:
            rows = s.exec(select(col, func.count(), func.count(RunLog.score), func.avg(RunLog.score),
                                 func.avg(RunLog.cost), func.sum(RunLog.cost))
                          .where(RunLog.user_id == user).group_by(col)).all()
        return [{"key": k, "runs": n, "judged": j, "avg_score": sc, "avg_cost": c, "total_cost": t}
                for k, n, j, sc, c, t in rows]
    model = router.load(user)
    return {"by_tier": grouped(RunLog.tier), "by_router": grouped(RunLog.routed_by),
            "trained_tiers": {t: m.get("n", 0) for t, m in model.items()},
            "usable": router.learned(model, "probe") is not None, "min_samples": router.MIN_SAMPLES}


@app.post("/api/router/train")
def router_train(user: str = Depends(current_user)):
    return router.train(user)


@app.websocket("/api/ws")
async def ws(socket: WebSocket, token: str | None = None, user: str | None = None):
    try:  # browsers cannot set headers on a WebSocket, so the session token travels as a query param
        user = await asyncio.to_thread(security.authenticate, token, user)
    except HTTPException:
        await socket.close(code=4401)
        return
    await socket.accept()
    runtime.sockets.setdefault(user, set()).add(socket)
    try:
        while True:
            await socket.receive_text()
    except WebSocketDisconnect:
        runtime.sockets[user].discard(socket)


_dist = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
if os.path.isdir(_dist):  # production: serve the built UI from the same origin
    app.mount("/", StaticFiles(directory=_dist, html=True), name="ui")
