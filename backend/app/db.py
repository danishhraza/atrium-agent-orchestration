"""Data model. Every row is scoped to a user_id so the platform is multi-tenant by construction."""
import os
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, LargeBinary
from sqlalchemy.exc import IntegrityError
from sqlmodel import Field, Session, SQLModel, create_engine, select

from .config import settings


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def now() -> datetime:
    return datetime.now(timezone.utc)


class User(SQLModel, table=True):
    id: str = Field(primary_key=True)  # the Clerk user id
    openrouter_key: str = ""           # encrypted at rest (security.encrypt); never sent to the client
    key_hint: str = ""                 # last 4 characters, for display
    tier_fast: str | None = None       # per-user model overrides; None = server default
    tier_balanced: str | None = None
    tier_smart: str | None = None
    judge_enabled: bool = True         # score finished tasks with an LLM judge (spends the user's tokens)
    router_mode: str = "heuristic"     # heuristic | learned
    explore_rate: float = 0.1          # share of auto-routed tasks sent to a random tier, to gather training data


class Agent(SQLModel, table=True):
    id: str = Field(default_factory=new_id, primary_key=True)
    user_id: str = Field(index=True)
    name: str
    role: str = ""                 # one line: what this agent is for (used for routing)
    instructions: str = ""         # free-form system prompt written by the user
    tier: str = "auto"             # fast | balanced | smart | auto (difficulty-routed)
    model: str | None = None       # explicit LiteLLM model, overrides tier
    tools: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    parent_id: str | None = None   # "reports to" - None means reports to the orchestrator
    is_orchestrator: bool = False
    created_at: datetime = Field(default_factory=now)


class Box(SQLModel, table=True):
    """A knowledge box: an isolated document store. Agents only see boxes they are assigned."""
    id: str = Field(default_factory=new_id, primary_key=True)
    user_id: str = Field(index=True)
    name: str
    description: str = ""


class AgentBox(SQLModel, table=True):
    agent_id: str = Field(primary_key=True)
    box_id: str = Field(primary_key=True)


class Document(SQLModel, table=True):
    id: str = Field(default_factory=new_id, primary_key=True)
    user_id: str = Field(index=True)
    box_id: str = Field(index=True)
    filename: str
    n_chunks: int = 0
    created_at: datetime = Field(default_factory=now)


class Chunk(SQLModel, table=True):
    id: str = Field(default_factory=new_id, primary_key=True)
    box_id: str = Field(index=True)
    document_id: str = Field(index=True)
    text: str
    embedding: bytes = Field(sa_column=Column(LargeBinary))


class Task(SQLModel, table=True):
    """A unit of work. A task with no parent assigned to the orchestrator is a 'mission'."""
    id: str = Field(default_factory=new_id, primary_key=True)
    user_id: str = Field(index=True)
    title: str
    brief: str = ""
    status: str = "pending"        # pending | in_progress | waiting | done | failed
    assignee_id: str
    created_by: str = "user"       # agent id or "user"
    parent_id: str | None = Field(default=None, index=True)
    result: str = ""
    handoffs: int = 0
    trace_id: str | None = None
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


class Message(SQLModel, table=True):
    """Everything anyone said to anyone. sender/recipient are agent ids or "user"."""
    id: str = Field(default_factory=new_id, primary_key=True)
    user_id: str = Field(index=True)
    sender: str
    recipient: str
    kind: str                      # chat | delegate | handoff | consult | reply | result | report
    content: str
    task_id: str | None = None
    trace_id: str | None = None
    created_at: datetime = Field(default_factory=now)


class RunLog(SQLModel, table=True):
    """One row per agent run: what it cost and (once judged) how good it was.

    This is the local mirror of a Langfuse trace, and the training set for the learned router.
    """
    id: str = Field(default_factory=new_id, primary_key=True)
    user_id: str = Field(index=True)
    agent_id: str
    task_id: str | None = None
    trace_id: str | None = None
    mode: str                      # chat | task | synthesize
    tier: str                      # fast | balanced | smart | custom
    routed_by: str                 # fixed | heuristic | learned | explore
    model: str
    text: str                      # the request, truncated: the router's input feature
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0              # USD, as reported by OpenRouter
    latency: float = 0.0           # seconds
    score: float | None = None     # judge score 0..1, filled in asynchronously
    created_at: datetime = Field(default_factory=now)


class RouterModel(SQLModel, table=True):
    """The trained difficulty router for one user (see router.py)."""
    user_id: str = Field(primary_key=True)
    weights: dict = Field(default_factory=dict, sa_column=Column(JSON))
    n_samples: int = 0
    trained_at: datetime = Field(default_factory=now)


os.makedirs(settings.data_dir, exist_ok=True)
_sqlite = settings.db_url.startswith("sqlite")
engine = create_engine(settings.db_url, pool_pre_ping=True,
                       # prepare_threshold=None: no server-side prepared statements, so pooled
                       # Postgres connections (Supabase/Neon poolers) work too.
                       connect_args={"check_same_thread": False} if _sqlite else {"prepare_threshold": None})


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def session() -> Session:
    return Session(engine, expire_on_commit=False)


def ensure_user(user_id: str) -> Agent:
    """Create the user and their orchestrator on first sight; return the orchestrator."""
    with session() as s:
        if not s.get(User, user_id):
            s.add(User(id=user_id))
            try:
                s.commit()
            except IntegrityError:  # two first requests raced; the other one created the row
                s.rollback()
        orch = s.exec(select(Agent).where(Agent.user_id == user_id, Agent.is_orchestrator)).first()
        if not orch:
            orch = Agent(
                user_id=user_id, name=settings.orchestrator_name, is_orchestrator=True, tier="balanced",
                role="Orchestrator: routes requests to the right agent and reports back",
            )
            s.add(orch)
            s.commit()
        return orch
