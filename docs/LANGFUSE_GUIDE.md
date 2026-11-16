# Using Langfuse to observe, evaluate and optimise Atrium

Langfuse covers four jobs here: **tracing** (what happened), **scores** (was it good),
**experiments** (did my change help), and **prompt management** (change prompts without deploying).
It is enough for this project. The one thing it should not be asked to do is guard hard invariants:
those live in `pytest` (see "What goes where" at the end).

## 1. Setup

1. Create a project at https://cloud.langfuse.com (free tier), or self-host with the Docker Compose
   file from the Langfuse repository.
2. Put the keys in `backend/.env`:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_BASE_URL=https://cloud.langfuse.com
   ```
3. Restart the backend. A "Langfuse on" pill appears in the header.

All Langfuse code is in [backend/app/telemetry.py](../backend/app/telemetry.py). Without keys it is a no-op.

## 2. How Atrium maps onto Langfuse

| Langfuse concept | In Atrium | Why this mapping |
|---|---|---|
| **Session** | one mission: the user request and everything it fans out into | open one session and read the whole story top to bottom |
| **Trace** | one agent run (`Jarvis · chat`, `Eli · task`, `Jarvis · synthesize`) | each run has its own latency, cost and score |
| **Agent observation** | the run itself; a consulted teammate nests inside the caller's trace | consultations show as a tree |
| **Generation** | each LLM call, with model, tokens and cost | cost and latency per model |
| **Tool / Retriever observation** | each tool call; each knowledge-box search with the boxes searched | see what an agent actually looked at |
| **User** | the Atrium user id | per-customer cost and quality |
| **Tags** | `agent:<name>`, `tier:<fast\|balanced\|smart\|custom>`, `mode:<chat\|task\|consult\|synthesize>`, `router:<fixed\|heuristic\|learned\|explore>` | the filters every analysis below uses |
| **Scores** | `task_quality` and `mission_quality` (LLM judge, automatic), `user_feedback` (thumbs in the chat) | quality signal on live traffic |

Cost, tier, routing decision and judge score for every run are also mirrored into the app's own
`RunLog` table. That is what the learned router trains on and what Settings shows, so neither
depends on the Langfuse API being reachable.

## 3. The optimisation loop

Work in this order. Each step tells you whether the next is worth doing.

### a. Find where the money and time go
Dashboard → cost and latency, grouped by tag.
- Group by `tier:*`. If `tier:smart` dominates cost, look at which agents use it.
- Group by `agent:*`. An agent with high cost and many generations per trace is looping: open a
  trace, look at the tool calls, and tighten its instructions or lower `ATRIUM_MAX_STEPS`.
- Check the synthesize traces. Their input is every subtask result; if they are large, have agents
  return shorter results (edit the `atrium-agent` prompt).

### b. Train and check the difficulty router
Agents on the `auto` tier start on keyword and length rules. To move to the learned router:
1. Keep judge scoring on (Settings) and use the team normally. About 10% of auto tasks are sent to a
   random tier on purpose (`router:explore`), which is how the system finds out where cheap models suffice.
2. When Settings shows 15+ judged runs on the fast and balanced tiers, press **Train router**, then
   switch routing to **learned**.
3. Compare in Langfuse: filter `router:learned` against `router:heuristic` and look at average
   `task_quality` and cost per trace. Settings shows the same split from `RunLog`.
4. Audit mistakes: filter `router:learned` + `tier:fast` and sort by lowest `task_quality`. Those are
   tasks it under-served. If there are many, raise `CONFIDENCE` in `router.py` or gather more data.
5. For a controlled before/after number, run the benchmark (below). Live comparisons are confounded
   by which tasks happened to arrive; the benchmark runs every task on every tier.

```
cd backend
OPENROUTER_API_KEY=sk-or-... uv run python -m evals.router_bench
```
It writes `docs/ROUTER_REPORT.md` with mean quality, pass rate and total cost for always-fast,
always-balanced, always-smart, heuristic, learned (cross-validated) and an oracle upper bound.
The headline line is learned versus always-smart.

### c. Cut cost without losing quality: run an experiment
```
cd backend
export OPENROUTER_API_KEY=sk-or-...
uv run python -m evals.run mission            # baseline
ATRIUM_TIER_SMART=anthropic/claude-sonnet-5.5 uv run python -m evals.run mission   # candidate
```
Each run is recorded as a Langfuse experiment. Compare the two runs side by side: if `quality` holds
and cost drops, ship the cheaper tier. Do the same for a single agent by changing its tier.

### d. Improve prompts without deploying
The three behavioural prompts are fetched from Langfuse prompt management, falling back to the
defaults in code: `atrium-orchestrator`, `atrium-agent`, `atrium-synthesize`.
1. Create a text prompt with one of those names in Langfuse and paste the default from
   [runtime.py](../backend/app/runtime.py) as the starting point.
2. Edit, label it `production`, and it is live within 60 seconds (the cache TTL).
3. Re-run `evals.run routing` after changing the orchestrator prompt, `evals.run mission` after
   the others. Keep the version that scores higher; roll back by moving the label.

### e. Turn failures into regression tests
When a trace gets a thumbs-down or a low judge score, add its input (and what the right answer
would have been) to the matching list in [evals/run.py](../backend/evals/run.py). Your eval set
then grows from real failures instead of imagined ones. This is the highest-value habit in the loop.

### f. Watch for drift
Chart `mission_quality` and `user_feedback` over time. A drop after a model, prompt or tool change
points you at the change; filter by `agent:*` to find which agent regressed.

## 4. The evals

| Suite | Question | Metric | Gate | Judge |
|---|---|---|---|---|
| `routing` | does the orchestrator pick the right agent? | `routing_accuracy` | ≥ 0.80 | exact match |
| `rag` | is the right document retrieved? | `retrieval_hit` | ≥ 0.99 | exact match |
| `rag` | does any box leak to an unassigned agent? | `box_leak` | = 0 | exact match |
| `mission` | does the team finish? | `completed` | ≥ 0.99 | status check |
| `mission` | is the final answer good? | `quality` | ≥ 0.70 | LLM judge with a reference answer |

`uv run python -m evals.run` exits non-zero when a gate fails, so it can run in CI.
Prefer exact-match and status checks wherever the answer is checkable; reserve the LLM judge for
open-ended output. Spot-check the judge itself: read ten of its scores and reasons in Langfuse and
confirm you agree before trusting the number.

**Online evals** run on live traffic: every finished task and mission is judged (`task_quality`, `mission_quality`) and users
can rate any reply (`user_feedback`). Offline evals tell you a change is safe to ship; online scores
tell you whether it actually was.

## 5. What goes where

| Kind of check | Where | Why |
|---|---|---|
| Must never break (box isolation, tenant isolation, auth, key encryption, task lifecycle, handoff caps, SSRF guard, router logic) | `pytest`, scripted model | deterministic, free, runs on every commit |
| Model-dependent behaviour (orchestrator routing, answer quality, tier routing savings) | `evals/`, real models, Langfuse experiments | statistical, costs tokens, run before releases |
| Live quality and cost | Langfuse scores and dashboards | the only source of truth about real usage |

## 6. Honest status

The tracing code targets Langfuse Python SDK v5 and was written against its installed signatures,
but it has not yet been exercised against a live Langfuse project in this repo. On first run with
keys, send one message, open the session in Langfuse, and confirm the session → trace → generation
nesting matches the table in section 2.
