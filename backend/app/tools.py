"""Tool registry.

Three kinds of tool:
  optional      - the user ticks these per agent (calculator, web_fetch, current_time)
  knowledge     - search_knowledge, offered automatically when the agent is assigned a box
  collaboration - message_agent / handoff_task / delegate_task / team_status / read_comms,
                  granted by the runtime from the agent's position in the team, not by the user
"""
import ast
import ipaddress
import operator
import re
import socket
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

import httpx
from sqlmodel import or_, select

from . import rag
from .config import settings
from .db import Agent, Message, Task, new_id, session


@dataclass
class Ctx:
    """Everything a tool may touch during one agent run."""
    rt: Any                       # runtime.Runtime
    creds: Any                    # llm.Creds of the user this run belongs to
    agent: Agent
    text: str                     # what the agent was asked this run
    mode: str
    task: Task | None = None
    mission_id: str | None = None # id to use if this run has to open a mission
    depth: int = 0
    handed_off: bool = False
    spawned: list[str] = field(default_factory=list)  # subtasks to start once this run ends
    usage: dict = field(default_factory=lambda: {"input": 0, "output": 0, "cost": 0.0})  # tokens and USD spent


@dataclass
class Tool:
    name: str
    description: str
    params: dict
    required: list[str]
    fn: Callable[..., Awaitable[str]]

    @property
    def spec(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": self.params, "required": self.required},
        }}


TOOLS: dict[str, Tool] = {}
OPTIONAL = ["calculator", "web_fetch", "current_time"]


def tool(name: str, description: str, params: dict | None = None, required: list[str] | None = None):
    def deco(fn):
        TOOLS[name] = Tool(name, description, params or {}, required or [], fn)
        return fn
    return deco


def _str(desc: str) -> dict:
    return {"type": "string", "description": desc}


def find_agent(user_id: str, ref: str) -> Agent | None:
    with session() as s:
        agents = s.exec(select(Agent).where(Agent.user_id == user_id)).all()
    ref = (ref or "").strip().lower()
    return next((a for a in agents if a.id == ref or a.name.lower() == ref), None)


# ------------------------------ optional tools ------------------------------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.FloorDiv: operator.floordiv}


def _calc(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _calc(node.left), _calc(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 200:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_calc(node.operand))
    raise ValueError("only numbers and + - * / // % ** are allowed")


@tool("calculator", "Evaluate an arithmetic expression exactly.", {"expression": _str("e.g. (1200 * 0.15) / 12")}, ["expression"])
async def calculator(ctx: Ctx, expression: str) -> str:
    return str(_calc(ast.parse(expression, mode="eval").body))


@tool("current_time", "Get the current local date and time.")
async def current_time(ctx: Ctx) -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _is_public(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    return all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


@tool("web_fetch", "Fetch a public web page and return its text.", {"url": _str("http(s) URL")}, ["url"])
async def web_fetch(ctx: Ctx, url: str) -> str:
    parsed = urlparse(url)
    # Agents take URLs from untrusted text, so never let them reach localhost or the private network.
    if parsed.scheme not in ("http", "https") or not parsed.hostname or not _is_public(parsed.hostname):
        return "Error: only public http(s) URLs can be fetched."
    async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
        resp = await client.get(url, headers={"User-Agent": "atrium-agent"})
    if resp.is_redirect:
        return f"Redirects to {resp.headers.get('location')} - fetch that URL if it is what you need."
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", resp.text)
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()
    return f"HTTP {resp.status_code}\n{text[:6000]}"


# ------------------------------ knowledge ------------------------------

@tool("search_knowledge", "Search the knowledge boxes you have been given access to. Returns the most relevant passages.",
      {"query": _str("what to look for")}, ["query"])
async def search_knowledge(ctx: Ctx, query: str) -> str:
    hits = await rag.search(ctx.creds.key, ctx.agent.id, query)
    if not hits:
        return "No matching passages in your knowledge boxes."
    return "\n\n".join(f"[{h['box']} / {h['document']} / score {h['score']}]\n{h['text']}" for h in hits)


# ------------------------------ collaboration ------------------------------

@tool("message_agent", "Ask a teammate a question and wait for their reply. Use for quick consultations; "
      "they answer from their own tools and knowledge.",
      {"agent": _str("teammate name or id"), "message": _str("your question, with all the context they need")},
      ["agent", "message"])
async def message_agent(ctx: Ctx, agent: str, message: str) -> str:
    target = find_agent(ctx.agent.user_id, agent)
    if not target or target.id == ctx.agent.id:
        return f"Error: no teammate called '{agent}'."
    rt, me = ctx.rt, ctx.agent
    task_id = ctx.task.id if ctx.task else None
    rt.say(me.user_id, me.id, target.id, "consult", message, task_id=task_id)
    rt.set_state(me, "talking", f"asking {target.name}", peer=target.id)
    run = await rt.run(target, f"Message from your teammate {me.name}:\n{message}", mode="consult", depth=ctx.depth + 1)
    rt.say(me.user_id, target.id, me.id, "reply", run.text, task_id=task_id)
    for k in ctx.usage:  # the teammate's spend counts toward the run that asked
        ctx.usage[k] += run.ctx.usage[k]
    return run.text


@tool("handoff_task", "Hand your current task over to a teammate who is better placed to finish it. "
      "They take full ownership and your work on it ends.",
      {"agent": _str("teammate name or id"), "notes": _str("what you did, what is left, anything they need to know")},
      ["agent", "notes"])
async def handoff_task(ctx: Ctx, agent: str, notes: str) -> str:
    target = find_agent(ctx.agent.user_id, agent)
    if not target or target.id == ctx.agent.id or target.is_orchestrator:
        return f"Error: cannot hand off to '{agent}'."
    task = ctx.task
    if ctx.spawned:
        return "Error: you already delegated subtasks for this task; finish it yourself."
    if task.handoffs >= settings.max_handoffs:
        return "Error: this task has been handed off too many times. Finish it yourself as best you can."
    task.assignee_id = target.id
    task.handoffs += 1
    task.status = "pending"
    task.brief += f"\n\n[Handoff from {ctx.agent.name}]: {notes}"
    ctx.rt.save_task(task)
    ctx.rt.say(ctx.agent.user_id, ctx.agent.id, target.id, "handoff", notes, task_id=task.id)
    ctx.handed_off = True
    return f"Handed off to {target.name}."


@tool("delegate_task", "Assign a task to one of the agents you can delegate to. It runs in the background; "
      "you are called again with every result once all delegated tasks finish. Call once per task.",
      {"agent": _str("agent name or id"), "title": _str("short task title"),
       "brief": _str("complete instructions: the agent sees only this, not your conversation")},
      ["agent", "title", "brief"])
async def delegate_task(ctx: Ctx, agent: str, title: str, brief: str) -> str:
    me, rt = ctx.agent, ctx.rt
    target = find_agent(me.user_id, agent)
    if not target or target.id == me.id or target.is_orchestrator:
        return f"Error: no agent called '{agent}' to delegate to."
    if not me.is_orchestrator and target.parent_id != me.id:
        return f"Error: {target.name} does not report to you."
    if ctx.task is None:  # first delegation of a chat turn opens a mission to hang the subtasks on
        ctx.task = Task(id=ctx.mission_id or new_id(), user_id=me.user_id, title=ctx.text[:80], brief=ctx.text,
                        status="in_progress", assignee_id=me.id)
        rt.save_task(ctx.task)
    sub = Task(user_id=me.user_id, title=title, brief=brief, assignee_id=target.id,
               created_by=me.id, parent_id=ctx.task.id)
    rt.save_task(sub)
    ctx.spawned.append(sub.id)
    rt.say(me.user_id, me.id, target.id, "delegate", f"{title}\n{brief}", task_id=sub.id)
    return f"Delegated to {target.name} (task {sub.id})."


@tool("team_status", "See what every agent is doing right now, plus open and recently finished tasks.")
async def team_status(ctx: Ctx) -> str:
    uid = ctx.agent.user_id
    with session() as s:
        agents = s.exec(select(Agent).where(Agent.user_id == uid)).all()
        tasks = s.exec(select(Task).where(Task.user_id == uid).order_by(Task.updated_at.desc()).limit(30)).all()
    names = {a.id: a.name for a in agents}
    lines = ["AGENTS"]
    for a in agents:
        st = ctx.rt.states.get(a.id, {})
        lines.append(f"- {a.name}: {st.get('state', 'idle')} {st.get('detail', '')}".rstrip())
    open_ = [t for t in tasks if t.status not in ("done", "failed")]
    lines.append(f"\nOPEN TASKS ({len(open_)})")
    lines += [f"- [{t.status}] {t.title} -> {names.get(t.assignee_id, '?')}" for t in open_]
    lines.append("\nRECENTLY FINISHED")
    lines += [f"- [{t.status}] {t.title} ({names.get(t.assignee_id, '?')}): {t.result[:200]}"
              for t in tasks if t.status in ("done", "failed")][:8]
    return "\n".join(lines)


@tool("read_comms", "Read recent messages between agents (delegations, handoffs, consultations, results).",
      {"agent": _str("optional: only messages involving this agent")})
async def read_comms(ctx: Ctx, agent: str = "") -> str:
    uid = ctx.agent.user_id
    with session() as s:
        q = select(Message).where(Message.user_id == uid, Message.sender != "user", Message.recipient != "user")
        if agent:
            target = find_agent(uid, agent)
            if not target:
                return f"Error: no agent called '{agent}'."
            q = q.where(or_(Message.sender == target.id, Message.recipient == target.id))
        msgs = s.exec(q.order_by(Message.created_at.desc()).limit(20)).all()
        names = {a.id: a.name for a in s.exec(select(Agent).where(Agent.user_id == uid)).all()}
    if not msgs:
        return "No messages between agents yet."
    return "\n".join(f"{names.get(m.sender, '?')} -> {names.get(m.recipient, '?')} [{m.kind}]: {m.content[:300]}"
                     for m in reversed(msgs))


@tool("retry_failed", "Resume the most recent failed work. Only the tasks that failed are run again, by the same "
      "agents; finished tasks keep their results and the final report is rewritten when they are done.")
async def retry_failed(ctx: Ctx) -> str:
    uid = ctx.agent.user_id
    with session() as s:
        failed = s.exec(select(Task).where(Task.user_id == uid, Task.status == "failed")
                        .order_by(Task.updated_at.desc())).all()
    if not failed:
        return "Nothing has failed; there is nothing to retry."
    newest = failed[0]
    with session() as s:  # climb to the mission the newest failure belongs to, and retry that whole mission
        while newest.parent_id:
            newest = s.get(Task, newest.parent_id)
    if newest.status not in ("failed", "done", "waiting"):
        return "That work is already running again."
    n = ctx.rt.retry(newest)
    return f"Restarted {n} failed task(s) from mission '{newest.title}'. The report will follow when they finish."


def tools_for(agent: Agent, mode: str, depth: int, has_boxes: bool, has_reports: bool) -> list[Tool]:
    """The tool set is derived from who the agent is and what it is doing right now."""
    if mode == "synthesize":
        return []
    names = [t for t in agent.tools if t in OPTIONAL]
    if has_boxes:
        names.append("search_knowledge")
    if depth < settings.max_consult_depth:
        names.append("message_agent")
    if mode == "task" and not agent.is_orchestrator:
        names.append("handoff_task")
    if depth == 0 and (agent.is_orchestrator or has_reports):
        names += ["delegate_task", "team_status", "read_comms", "retry_failed"]
    return [TOOLS[n] for n in names]
