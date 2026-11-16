"""The agent runtime: run loop, task execution, delegation fan-out / synthesis fan-in, live events.

Lifecycle of a mission:
  user -> orchestrator.chat()            orchestrator answers directly, or calls delegate_task
       -> execute(subtask) per delegate  each assignee runs its tool loop in the background;
                                         it may consult a teammate, hand the task off, or
                                         (if it is a lead) delegate further down
       -> synthesize(parent)             when every subtask of a parent is finished, the parent's
                                         owner is re-run with all results and writes the answer
       -> report to the user
  Every run is logged with its cost (RunLog); finished tasks are scored by an LLM judge, and those
  scores are what the learned router trains on.
"""
import asyncio
import contextvars
import time
from dataclasses import dataclass

from sqlmodel import or_, select

from . import llm, rag, router, telemetry
from .config import settings
from .db import Agent, Message, RunLog, Task, new_id, now, session
from .judge import judge
from .tools import Ctx, tools_for

ORCHESTRATOR_GUIDE = """How to work:
- You are the user's single point of contact for their agent team.
- Answer directly when the request is a simple question you can answer yourself.
- Otherwise call delegate_task for each piece of work, choosing the agent whose role fits best. Split
  work across several agents when it has independent parts. Each brief must be self-contained.
- After delegating, tell the user in one or two sentences who is doing what. Do not invent results;
  you will be called again with the real results when the work is finished.
- For questions about progress, pending work or what agents said to each other, use team_status and read_comms.
- If the user asks to retry, try again or continue after something failed, call retry_failed. It resumes only
  the failed tasks; do not delegate the work again from scratch.
- Replies are shown as Markdown; keep links and tables from your team's results intact."""

AGENT_GUIDE = """How to work:
- Do the work yourself with your tools. Search your knowledge boxes before answering from memory.
- message_agent: ask a teammate when they hold knowledge or a skill you lack.
- handoff_task: only when the task clearly belongs to a teammate's role rather than yours.
- Finish with the result itself, written for the person who asked. Say plainly what you could not do.
- Replies are shown as Markdown. Use short headings and lists; a table to compare options; and for products
  or sources give the name as a link to its real page (and an image only if you have its real URL). Never invent links."""

SYNTH_GUIDE = """Your team has finished the work you delegated. Using only the results below, write the final
answer to the original request. Be direct, attribute work to the agent who did it where useful, and flag
anything that failed or is missing."""


@dataclass
class Run:
    text: str
    trace_id: str | None
    ctx: Ctx
    log_id: str | None = None  # RunLog row, for attaching the judge score later


class Runtime:
    def __init__(self):
        self.states: dict[str, dict] = {}            # agent_id -> {state, detail, peer}
        self.active: dict[str, int] = {}             # agent_id -> runs in flight
        self.locks: dict[str, asyncio.Lock] = {}     # one task at a time per agent
        self.sockets: dict[str, set] = {}            # user_id -> websockets
        self.bg: set[asyncio.Task] = set()

    # ------------------------------ events ------------------------------

    def publish(self, user_id: str, event: dict) -> None:
        for ws in list(self.sockets.get(user_id, ())):
            self._bg(self._send(user_id, ws, event))

    async def _send(self, user_id, ws, event):
        try:
            await ws.send_json(event)
        except Exception:
            self.sockets.get(user_id, set()).discard(ws)

    def _bg(self, coro, fresh_context: bool = False) -> None:
        # fresh_context: background work must start its own Langfuse trace instead of
        # nesting under whichever trace happened to schedule it.
        ctx = contextvars.Context() if fresh_context else None
        t = asyncio.get_running_loop().create_task(coro, context=ctx)
        self.bg.add(t)
        t.add_done_callback(self.bg.discard)

    async def idle(self) -> None:
        """Wait until all background work has finished (used by tests and evals)."""
        while self.bg:
            # The sleep(0) is required: waiting on already-finished tasks returns without yielding,
            # so their done-callbacks (which remove them from self.bg) would never get to run.
            await asyncio.sleep(0)
            pending = [t for t in self.bg if not t.done()]
            if pending:
                await asyncio.wait(pending)

    def set_state(self, agent: Agent, state: str, detail: str = "", peer: str | None = None) -> None:
        self.states[agent.id] = {"state": state, "detail": detail, "peer": peer}
        self.publish(agent.user_id, {"type": "state", "agent_id": agent.id, **self.states[agent.id]})

    def say(self, user_id, sender, recipient, kind, content, task_id=None, trace_id=None) -> Message:
        msg = Message(user_id=user_id, sender=sender, recipient=recipient, kind=kind, content=content,
                      task_id=task_id, trace_id=trace_id)
        with session() as s:
            s.add(msg)
            s.commit()
        self.publish(user_id, {"type": "message", "message": msg.model_dump(mode="json")})
        return msg

    def save_task(self, task: Task) -> None:
        task.updated_at = now()
        with session() as s:
            s.merge(task)
            s.commit()
        self.publish(task.user_id, {"type": "task", "task": task.model_dump(mode="json")})

    # ------------------------------ the run loop ------------------------------

    def _system_prompt(self, agent: Agent, mode: str, team: list[Agent], boxes: list) -> str:
        if mode == "synthesize":
            return f"You are {agent.name}. {telemetry.prompt('atrium-synthesize', SYNTH_GUIDE)}"
        parts = [f"You are {agent.name}, an AI agent on the user's team.", f"Role: {agent.role}"]
        if agent.instructions:
            parts.append(agent.instructions)
        others = [a for a in team if a.id != agent.id and not a.is_orchestrator]
        delegable = others if agent.is_orchestrator else [a for a in others if a.parent_id == agent.id]
        names = {a.id: a.name for a in team}
        if delegable:
            parts.append("Agents you can delegate to:\n" + "\n".join(
                f"- {a.name} (id: {a.id}) — {a.role}"
                + (f" [reports to {names[a.parent_id]}]" if agent.is_orchestrator and a.parent_id in names else "")
                for a in delegable))
        peers = [a for a in others if a not in delegable]
        if peers:
            parts.append("Other teammates:\n" + "\n".join(f"* {a.name}: {a.role}" for a in peers))
        if boxes:
            parts.append("Your knowledge boxes: " + ", ".join(f"{b.name} ({b.description})" if b.description else b.name for b in boxes))
        if agent.is_orchestrator or delegable:
            parts.append(telemetry.prompt("atrium-orchestrator", ORCHESTRATOR_GUIDE))
        if not agent.is_orchestrator:
            parts.append(telemetry.prompt("atrium-agent", AGENT_GUIDE))
        return "\n\n".join(parts)

    async def run(self, agent: Agent, text: str, *, mode: str, task: Task | None = None, history: list[dict] = (),
                  depth: int = 0, session_id: str | None = None, mission_id: str | None = None) -> Run:
        """One agent run: LLM turn -> tool calls -> LLM turn ... until it answers or hands off."""
        with session() as s:
            team = list(s.exec(select(Agent).where(Agent.user_id == agent.user_id)).all())
        boxes = rag.allowed_boxes(agent.id)
        has_reports = any(a.parent_id == agent.id for a in team)
        tools = {t.name: t for t in tools_for(agent, mode, depth, bool(boxes), has_reports)}
        specs = [t.spec for t in tools.values()]
        creds = llm.creds(agent.user_id)  # raises llm.MissingKey if the user has not added a key
        model, tier, routed_by = router.pick(agent, text, creds)
        started = time.monotonic()
        ctx = Ctx(rt=self, creds=creds, agent=agent, text=text, mode=mode, task=task, mission_id=mission_id, depth=depth)
        usage = ctx.usage
        messages = [{"role": "system", "content": self._system_prompt(agent, mode, team, boxes)},
                    *history, {"role": "user", "content": text}]

        self.active[agent.id] = self.active.get(agent.id, 0) + 1
        try:
            if depth == 0:  # a consult (depth > 0) nests inside the caller's trace instead
                attrs = telemetry.trace_attrs(
                    user_id=agent.user_id, session_id=session_id or f"chat-{agent.id}",
                    tags=[f"agent:{agent.name}", f"tier:{tier}", f"mode:{mode}", f"router:{routed_by}"],
                    trace_name=f"{agent.name} · {mode}", metadata={"model": model, "task_id": task.id if task else ""})
            else:
                attrs = _null()
            with attrs, telemetry.observation(agent.name, as_type="agent", input=text,
                                              metadata={"mode": mode, "tier": tier}) as root:
                final = "I ran out of steps before finishing this."
                for _ in range(settings.max_steps):
                    self.set_state(agent, "thinking", mode)
                    out = await llm.complete(creds.key, model, messages, specs, name=f"{agent.name}-llm")
                    for k in usage:
                        usage[k] += out["usage"][k] or 0
                    messages.append(out["raw"])
                    if not out["tool_calls"]:
                        final = out["content"]
                        break
                    for call in out["tool_calls"]:
                        self.set_state(agent, "tool", call["name"])
                        with telemetry.observation(call["name"], as_type="tool", input=call["arguments"]) as obs:
                            try:
                                result = await tools[call["name"]].fn(ctx, **call["arguments"])
                            except Exception as e:  # the model sees the error and can recover
                                result = f"Error: {type(e).__name__}: {e}"
                            obs.update(output=result)
                        messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
                    if ctx.handed_off:
                        final = "Handed off."
                        break
                root.update(output=final)
                trace_id = telemetry.current_trace_id()
            log_id = None
            if depth == 0:  # a consult (depth > 0) is billed to the run that asked; see tools.message_agent
                log = RunLog(user_id=agent.user_id, agent_id=agent.id, task_id=task.id if task else None,
                             trace_id=trace_id, mode=mode, tier=tier, routed_by=routed_by, model=model,
                             text=text[:2000], input_tokens=usage["input"], output_tokens=usage["output"],
                             cost=usage["cost"], latency=time.monotonic() - started)
                with session() as s:
                    s.add(log)
                    s.commit()
                log_id = log.id
            return Run(final, trace_id, ctx, log_id)
        finally:
            self.active[agent.id] -= 1
            if self.active[agent.id] == 0:
                self.set_state(agent, "idle")
            else:
                self.set_state(agent, "thinking")

    # ------------------------------ user-facing entry point ------------------------------

    def _history(self, user_id: str, agent_id: str, limit: int = 12) -> list[dict]:
        with session() as s:
            msgs = s.exec(select(Message).where(
                Message.user_id == user_id, Message.kind.in_(["chat", "report"]),
                or_((Message.sender == "user") & (Message.recipient == agent_id),
                    (Message.sender == agent_id) & (Message.recipient == "user")),
            ).order_by(Message.created_at.desc()).limit(limit)).all()
        return [{"role": "user" if m.sender == "user" else "assistant", "content": m.content} for m in reversed(msgs)]

    async def chat(self, agent: Agent, content: str) -> Message:
        """The user messages any agent and gets one reply. Work the agent delegates continues in the background."""
        uid = agent.user_id
        history = self._history(uid, agent.id)
        self.say(uid, "user", agent.id, "chat", content)
        mission_id = new_id()
        try:
            run = await self.run(agent, content, mode="chat", history=history,
                                 session_id=mission_id, mission_id=mission_id)
        except Exception as e:
            return self.say(uid, agent.id, "user", "chat", llm.friendly_error(e))
        mission = run.ctx.task
        if mission:
            mission.trace_id = run.trace_id
        reply = self.say(uid, agent.id, "user", "chat", run.text,
                         task_id=mission.id if mission else None, trace_id=run.trace_id)
        if mission:
            self._fan_out(mission, run.ctx.spawned)
        return reply

    # ------------------------------ tasks ------------------------------

    def _fan_out(self, parent: Task, subtask_ids: list[str]) -> None:
        parent.status = "waiting"
        self.save_task(parent)
        for tid in subtask_ids:
            self._bg(self.execute(tid), fresh_context=True)

    def _root_id(self, task: Task) -> str:
        with session() as s:
            while task.parent_id:
                task = s.get(Task, task.parent_id)
        return task.id

    async def execute(self, task_id: str) -> None:
        with session() as s:
            task = s.get(Task, task_id)
            agent = s.get(Agent, task.assignee_id)
        async with self.locks.setdefault(agent.id, asyncio.Lock()):
            task.status = "in_progress"
            self.save_task(task)
            try:
                run = await self.run(agent, f"Task: {task.title}\n\n{task.brief}", mode="task", task=task,
                                     session_id=self._root_id(task))
            except Exception as e:
                await self._finish(task, agent, llm.friendly_error(e), None, status="failed")
                return
        task.trace_id = run.trace_id
        if run.ctx.handed_off:          # the handoff tool already reassigned it; the new owner starts now
            self.save_task(task)
            self._bg(self.execute(task.id), fresh_context=True)
        elif run.ctx.spawned:           # a lead delegated further: wait for its reports
            self._fan_out(task, run.ctx.spawned)
        else:
            await self._finish(task, agent, run.text, run)

    async def _finish(self, task: Task, agent: Agent, text: str, run: Run | None, status: str = "done") -> None:
        trace_id = run.trace_id if run else None
        task.status, task.result = status, text
        self.save_task(task)
        if run and status == "done" and run.ctx.creds.judge_enabled:
            self._bg(self._judge(task, run), fresh_context=True)
        to_user = task.created_by == "user"
        self.say(task.user_id, agent.id, task.created_by, "report" if to_user else "result", text,
                 task_id=task.id, trace_id=trace_id)
        if task.parent_id:
            with session() as s:
                parent = s.get(Task, task.parent_id)
                siblings = s.exec(select(Task).where(Task.parent_id == parent.id)).all()
            if parent.status == "waiting" and all(t.status in ("done", "failed") for t in siblings):
                parent.status = "in_progress"  # claimed synchronously, so only one synthesis starts
                self.save_task(parent)
                self._bg(self.synthesize(parent, siblings), fresh_context=True)

    def retry(self, task: Task) -> int:
        """Resume failed work from where it stopped, without redoing what already succeeded.

        Every task is a saved row with its brief, owner and result, so that row is the checkpoint:
          - a failed task with no subtasks is run again by the same agent from its brief
          - a task whose subtasks failed retries only those subtasks; finished ones keep their results
          - a task whose subtasks all finished but whose own summary failed is only re-summarised
        Its ancestors go back to "waiting", so results flow up and the final report is rewritten.
        Returns how many tasks were restarted.
        """
        with session() as s:
            children = list(s.exec(select(Task).where(Task.parent_id == task.id)).all())
            ancestors, cur = [], task
            while cur.parent_id:
                cur = s.get(Task, cur.parent_id)
                ancestors.append(cur)
        failed = [c for c in children if c.status == "failed"]
        if failed:
            return sum(self.retry(c) for c in failed)
        for a in ancestors:
            a.status = "waiting"
            self.save_task(a)
        if children:  # the work is done; only this task's own synthesis failed
            task.status = "in_progress"
            self.save_task(task)
            self._bg(self.synthesize(task, children), fresh_context=True)
        else:
            task.status, task.result = "pending", ""
            self.save_task(task)
            self._bg(self.execute(task.id), fresh_context=True)
        return 1

    async def synthesize(self, parent: Task, children: list[Task]) -> None:
        with session() as s:
            agent = s.get(Agent, parent.assignee_id)
            names = {a.id: a.name for a in s.exec(select(Agent).where(Agent.user_id == parent.user_id)).all()}
        results = "\n\n".join(f"### {names.get(c.assignee_id, '?')} — {c.title} [{c.status}]\n{c.result}" for c in children)
        text = f"Original request:\n{parent.brief}\n\nResults from your team:\n\n{results}"
        try:
            run = await self.run(agent, text, mode="synthesize", task=parent, session_id=self._root_id(parent))
            await self._finish(parent, agent, run.text, run)
        except Exception as e:
            await self._finish(parent, agent, llm.friendly_error(e), None, status="failed")

    async def _judge(self, task: Task, run: Run) -> None:
        """Online eval: score a finished task. The score goes on the Langfuse trace and on the RunLog
        row, where it becomes a training example for the learned router."""
        creds = run.ctx.creds
        try:
            with telemetry.trace_attrs(user_id=task.user_id, session_id=self._root_id(task), tags=["judge"], trace_name="judge"):
                request = task.title + "\n\n" + task.brief
                value, reason = await judge(creds.key, creds.tiers["balanced"], request, task.result)
        except Exception:
            return  # judging is best-effort; never let it break the mission
        if value is None:
            return
        telemetry.score(run.trace_id, "task_quality" if task.parent_id else "mission_quality", value, comment=reason)
        with session() as s:
            log = s.get(RunLog, run.log_id) if run.log_id else None
            if log:
                log.score = value
                s.add(log)
                s.commit()


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


runtime = Runtime()
