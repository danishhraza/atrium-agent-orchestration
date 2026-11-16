# Atrium: architecture and plan

Atrium is a platform where a user builds their own team of AI agents, gives each one a role, tools,
a model tier and access to specific document stores, and then talks to a single orchestrator
("Jarvis") that routes work, tracks it, and reports back. A live clubhouse view shows who is asleep,
thinking, using a tool, or walking over to talk to whom. Users sign in with Clerk and bring their
own OpenRouter key.

## The shape of it

```
                        ┌──────────────── React UI ────────────────┐
                        │ live floor · chat · tasks · comms · boxes │
                        └───────▲──────────────────────┬───────────┘
                         events │ WebSocket       REST │
┌───────────────────────────────┴──────────────────────▼──────────────────────────────┐
│ FastAPI                                                                              │
│  ┌───────────── Runtime (asyncio) ─────────────┐   ┌────────── Knowledge ─────────┐  │
│  │ run loop: LLM turn → tool calls → LLM turn  │   │ boxes · documents · chunks   │  │
│  │ chat · execute · fan-out · synthesize       │──▶│ search filtered by AgentBox  │  │
│  └──────┬──────────────────────┬───────────────┘   └──────────────────────────────┘  │
│         │ every LLM call       │ every run / tool / retrieval                        │
│  ┌──────▼──────┐        ┌──────▼──────┐        ┌────────────────────────────────┐    │
│  │   LiteLLM   │        │  Langfuse   │        │ SQLite or Postgres (agents,    │    │
│  │ tier router │        │ traces+evals│        │ messages, boxes, chunks, runs) │    │
│  └──────┬──────┘        └─────────────┘        └────────────────────────────────┘    │
└─────────┼────────────────────────────────────────────────────────────────────────────┘
          ▼
  OpenRouter, with the calling user's own key (any model, per tier)
```

## Decisions, and why

**Hierarchy: a tree with the orchestrator at the root, parents optional.**
Every user gets one orchestrator. Agents report to it by default; an agent can instead report to
another agent, which makes that agent a *lead*. Delegation follows the tree (the orchestrator can
delegate to anyone, a lead only to its direct reports), and results flow back up the same path: when
all of a parent's subtasks finish, the parent is re-run to synthesize them. The same two mechanisms
(fan-out, fan-in) work at every level, so "4 sales reps under a sales lead under Jarvis" needs no
special code. Flat teams are just a tree of depth one, so the parent is there when needed and
invisible when not.

**Three ways agents interact, each with a different ownership rule.**

| Primitive | Who owns the work afterwards | Used for |
|---|---|---|
| `delegate_task` | the delegator still owns the outcome; the assignee owns a subtask | splitting work downward |
| `handoff_task` | the receiver owns it fully; the sender is done | "this isn't mine, it's theirs" |
| `message_agent` | nobody changes; it's a question and an answer | borrowing a peer's knowledge |

Handoffs are capped per task and consults are depth-limited, so agents can't ping-pong forever.

**Collaboration tools are granted by position, not configured.** The user ticks optional tools
(calculator, web fetch, clock). `search_knowledge` appears when the agent has a box. Delegation tools
appear when the agent has reports. Users can't build a team that is wired wrong.

**Selective RAG is enforced in the query, not the prompt.** `AgentBox` is a join table; retrieval
does `WHERE box_id IN (boxes assigned to this agent)`. An agent that is told (or tricked by a
document) to "search the other team's list" simply has no rows to find. This is covered by a
deterministic test and an eval gate (`box_leak` must be 0).

**Model per agent, or by difficulty.** Each agent picks `fast`, `balanced`, `smart`, an explicit
OpenRouter model id, or `auto`. Each user maps the three tiers to models of their choice. LiteLLM is
the client library, pointed at OpenRouter.

**The learned router.** For `auto` agents the question is "what is the cheapest tier that will do
this well enough?".
- Every agent run is logged with its cost (`RunLog`), and every finished task is scored 0 to 1 by an
  LLM judge. Those rows are the training set.
- Per tier, a small logistic regression predicts "will this tier pass the judge on this request?"
  from hashed words plus a few shape features (length, hard-task keywords, digits, questions).
- Routing picks the cheapest tier whose predicted pass probability is at least 0.75, else the top tier.
- A configurable share of auto tasks (10% by default) goes to a random tier. Without that
  exploration the system would never learn that a cheap model is good enough for something the
  heuristic always sends upmarket.
- Until a user has 15 judged runs on the fast and balanced tiers, routing falls back to keyword and
  length rules.
- `evals/router_bench.py` produces the cost and quality comparison: it runs a task set on all three
  tiers, then replays each policy against that matrix, scoring the learned router with 5-fold
  cross-validation so it is never graded on tasks it trained on.

**Bring your own key, behind real auth.** Clerk issues the session; the backend verifies the JWT
against the Clerk instance's public keys (derived from the publishable key, so the server holds no
Clerk secret). Each user's OpenRouter key is validated with OpenRouter, encrypted with Fernet under
a server secret, and never returned to the browser; only the last four characters are shown.

**Asynchronous by default.** A chat message returns one reply immediately. Delegated work continues
in background asyncio tasks and the final report arrives over the WebSocket. That is what lets you
ask Jarvis "what's everyone doing?" while a mission is still running.

**Mechanics and quality are tested separately.** The test-suite swaps the model call for a scripted
stand-in, so delegation, handoff, isolation, auth and routing logic are checked deterministically and
for free. The application itself has no mock mode; model quality is answered by the evals.

## Data model

`User` (settings, encrypted key) 1—* `Agent` (self-referencing `parent_id`) · `Agent` *—* `Box` via `AgentBox` ·
`Box` 1—* `Document` 1—* `Chunk(embedding)` · `Task` (self-referencing `parent_id`; a root task is a
mission) · `Message(sender, recipient, kind)` is the full communication log · `RunLog` (cost, tier, judge score per run) ·
`RouterModel` (trained weights per user). Every table carries
`user_id`; every query filters on it.

## Storage

One relational database holds everything, including the vectors. Locally that is a SQLite file
(`backend/data/atrium.db`); in production it is whatever Postgres `ATRIUM_DATABASE_URL` points at
(for Supabase or Neon, use the Postgres **connection string**, not an API or anon key). Tables are
created on first start.

| Table | One row per | Key columns | Scoped by |
|---|---|---|---|
| `user` | signed-in person (id = Clerk user id) | encrypted OpenRouter key, key hint, tier models, judge on/off, router mode, explore rate | is the scope |
| `agent` | agent | name, role, instructions, tier, model override, tools, `parent_id` (reports to), `is_orchestrator` | `user_id` |
| `box` | knowledge box | name, description | `user_id` |
| `agentbox` | agent-to-box grant | `agent_id`, `box_id` | via agent and box |
| `document` | uploaded file | filename, chunk count, `box_id` | `user_id` |
| `chunk` | passage of a document | text, `embedding` (float32 bytes), `box_id`, `document_id` | via box |
| `task` | unit of work | title, brief, status, assignee, creator, `parent_id`, result, handoffs, trace id | `user_id` |
| `message` | anything anyone said | sender, recipient, kind, content, task id, trace id | `user_id` |
| `runlog` | agent run | tier, routed_by, model, tokens, cost, latency, judge score, request text | `user_id` |
| `routermodel` | trained router | weights, sample count, trained at | `user_id` (primary key) |

**Everything is stored per user.** Every route resolves the caller from their Clerk session, and
every query filters on that user id, so one account can never read or change another's agents,
boxes, tasks or messages (covered by a test).

**There is no separate vector database.** Embeddings are created through OpenRouter with the user's
key and stored as bytes in the `chunk` table, in the same database as everything else. A search
loads the chunks of the boxes the agent is allowed to see and ranks them by cosine similarity in
NumPy. So with a Postgres URL set, vectors live in that Postgres automatically; nothing else needs
configuring. This is simple and exact, and fast enough for thousands of chunks per user. Past
roughly 50k chunks, switch `rag.search` to pgvector (Supabase and Neon both support it).

**Uploaded originals** are also kept on the server's disk under `backend/data/uploads/<user>/<box>/`.
Search does not need them (the text is in `chunk`), but they would be lost on a host with an
ephemeral disk; move them to object storage if you need to keep or re-index them.

**Configuration is read once at startup.** After editing `backend/.env`, stop and start the server;
`--reload` only watches code. Changing the database URL starts with empty tables, and switching from
dev auth to Clerk gives you a new user id, so earlier local data will not appear under the new login.

## Request lifecycle

1. `POST /api/chat` → `Runtime.chat(orchestrator, text)`.
2. Orchestrator run: it answers, or calls `delegate_task` one or more times. The first delegation
   opens a mission task; each delegation creates a subtask.
3. Reply returns to the user. Subtasks start (`execute`), one at a time per agent.
4. Each assignee runs its loop. It may search its boxes, consult a peer, hand off, or delegate further.
5. When the last subtask of a parent finishes, `synthesize` re-runs the parent with all results.
6. The root result is sent to the user as the final report. Each finished task is scored by an LLM judge.

Agent replies are rendered as Markdown in the chat (headings, lists, tables, links, images), and the
agent prompt tells agents so: a shopping agent can return a comparison table with product links.
Raw HTML in model output is never rendered.

Every step emits `state`, `message` and `task` events over the WebSocket; the floor view is a pure
function of those events: an agent in the `talking` state walks to its peer's desk.

## Failure and resume

There is no checkpointing framework (no LangGraph or similar). The `task` table is the checkpoint:
each task row stores its brief, its owner, its status and its result, and parent/child links record
how a mission fanned out.

- **Granularity is the task.** When an agent run fails (out of credit, provider error), its task is
  marked `failed` with a readable reason. Finished sibling tasks keep their results.
- **Retry resumes, it does not restart.** `Runtime.retry` re-runs only failed tasks, with the same
  agent and brief; if all subtasks finished and only the summary failed, it only re-summarises. Ancestors
  go back to `waiting`, so the final report is rewritten once the retried work lands.
- **Three ways to trigger it:** the Retry button on a failed task (Tasks tab), `POST /api/tasks/{id}/retry`,
  or telling the orchestrator "try again", which calls its `retry_failed` tool instead of delegating afresh.
- **What is not saved:** progress *inside* one agent run. If an agent fails on its fourth tool call, the
  retry starts that agent's run from its brief, not from the fourth step. Saving each step's message
  history per run would add that; so far a single run is short enough that it has not been worth it.
- **Not covered:** a server restart mid-mission. Tasks left `in_progress` stay that way until retried by
  hand; resuming them automatically on startup is the next step toward durable execution.

## What is deliberately simple today, and the upgrade path

| Today | When it matters | Then |
|---|---|---|
| SQLite by default; Postgres via `ATRIUM_DATABASE_URL` | any deployment | set the URL (no Docker needed with a hosted Postgres) |
| Brute-force cosine in numpy | > ~50k chunks | pgvector; only `rag.search` changes |
| Tables created on start, no migrations | first schema change in production | Alembic |
| In-process asyncio task runner | work must survive a restart | a durable queue (Arq/Celery/Temporal); `Runtime.execute` is already the unit of work |
| Router trains on demand (button) | steady traffic | retrain on a schedule; pool anonymised data across users for cold start |
| Three built-in optional tools | users need their own systems | MCP client: let users attach MCP servers per agent |

## Ideas that would make it more impressive

- **MCP tool marketplace**: users attach any MCP server (CRM, calendar, email) to an agent. Highest
  value per line of code, and a strong talking point.
- **Council mode**: fan the same question out to several agents on different models, have them
  critique each other, and let the orchestrator synthesize. The fan-out/fan-in machinery already exists.
- **Human-in-the-loop approvals**: a task state `needs_approval` for risky tools (send email, spend money).
- **Per-agent memory**: a private, writable knowledge box each agent appends its own notes to.
- **Durable execution**: resume a mission after a crash from the task table.
