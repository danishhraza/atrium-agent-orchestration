# Atrium

Build your own team of AI agents. Give each one a role, tools, a model and its own documents, then
talk to one orchestrator ("Jarvis") that routes the work, tracks it and reports back, while you
watch the team work in a live clubhouse: agents nap at their desks, think, use tools, and walk over
to each other to talk.

- **Build any team**: 4 sales reps each with their own call list, a marketing pod, a research desk.
  Agents optionally report to a lead, who can delegate further.
- **One point of contact**: ask Jarvis for anything, including "what's everyone working on?" and
  "what did Eli tell Wren?". Or message any agent directly for a single reply.
- **Real teamwork**: agents delegate, hand off tasks, and consult each other. Every exchange is logged.
- **Selective RAG**: upload files into knowledge boxes; an agent can only search boxes it is assigned.
- **Bring your own key**: each user signs in (Clerk) and adds their own OpenRouter key, stored
  encrypted. Any tool-capable OpenRouter model can back any tier.
- **Learned model routing**: "auto" agents get the cheapest model tier predicted to pass, from a
  router trained on the user's own judged runs. A benchmark reports cost and quality against always
  using the top model.
- **Resume after failure**: if an agent fails (out of credit, provider error), retry just that task;
  finished work is kept and the report is rewritten.
- **Readable answers**: agent replies render as Markdown, so lists, comparison tables, links and
  images show properly in the chat.
- **Observable and evaluated**: every run traced in Langfuse and costed locally; LLM-judge scores on
  live traffic; an offline eval suite with CI gates.

## Run it locally

Needs Python 3.11+ with [uv](https://docs.astral.sh/uv/) and Node 20+. No Docker.

```bash
cd backend && uv sync
```
```bash
cd frontend && npm install && npm run build
```

Create `backend/.env` (see `backend/.env.example`). The quickest local setup skips login:

```
ATRIUM_DEV_AUTH=1
```

```bash
cd backend && uv run uvicorn app.main:app --port 8000
```

Open http://localhost:8000, click **Settings**, paste an OpenRouter key, then **Load demo team**.

Settings in `backend/.env` are read at startup: restart the server after changing them.

For frontend development, run `npm run dev` in `frontend/` (port 5173, proxies to the backend).

## Ship it

One process serves the API and the built UI. To deploy:

1. **Auth**: create a Clerk application and set `ATRIUM_CLERK_PUBLISHABLE_KEY`. Do not set `ATRIUM_DEV_AUTH`.
2. **Secret**: set `ATRIUM_SECRET_KEY` (command in `.env.example`). It encrypts users' API keys.
3. **Database**: set `ATRIUM_DATABASE_URL` to a Postgres connection string (Neon and Supabase have free
   hosted tiers). It must be the `postgresql://...` URL, not an API key. On Supabase use the direct or
   session-pooler connection (port 5432). Tables are created on first start, and document vectors live
   in the same database; there is no separate vector store to set up.
4. **Host**: any Python host (Render, Railway, Fly). Build `frontend/`, then start
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT` from `backend/`.
5. **Run a single instance.** Live events and background missions are held in process memory; see
   "Known limits" before scaling out.

## Check it

```bash
cd backend && uv run pytest
```

These call real models and need `OPENROUTER_API_KEY` in the environment:

```bash
cd backend && uv run python -m evals.run
```
```bash
cd backend && uv run python -m evals.router_bench --limit 12
```

## Read more

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): design decisions, tables and storage, lifecycle, roadmap
- [docs/LANGFUSE_GUIDE.md](docs/LANGFUSE_GUIDE.md): tracing model, the optimisation loop, evals, the router

## Layout

```
backend/app/runtime.py     run loop, delegation fan-out, synthesis fan-in, live events
backend/app/router.py      heuristic and learned difficulty routers
backend/app/tools.py       tool registry: optional, knowledge, collaboration
backend/app/rag.py         knowledge boxes and access-filtered retrieval
backend/app/llm.py         OpenRouter calls through LiteLLM, per-user credentials
backend/app/security.py    Clerk token verification, API-key encryption
backend/app/telemetry.py   Langfuse wrapper
backend/evals/             eval suites, router benchmark
frontend/src/Room.tsx      the clubhouse
frontend/src/Markdown.tsx  safe Markdown rendering of agent replies
```

## Known limits

- **Not yet run against live services.** The test-suite uses a scripted model. Real OpenRouter calls,
  Clerk sign-in, Postgres and Langfuse are implemented against their documented interfaces but have
  not been exercised end to end here, because no accounts or keys were available. Expect a round of
  fixes on first contact.
- **Single instance.** Missions in flight are lost if the server restarts, and two instances would not
  share live events. A durable queue and a pub/sub layer are the next step for scale.
- **No schema migrations yet.** Tables are created on start; add Alembic before changing models in
  production.
- **Retrieval is a brute-force scan** per search. Fine for thousands of chunks; move to pgvector beyond that.
