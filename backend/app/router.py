"""Difficulty routing for agents on the "auto" tier: which model tier should handle this request?

Two routers:
  heuristic - keyword/length rules. Zero cost, no data needed. The cold-start default.
  learned   - one small logistic regression per tier that predicts "will this tier pass the judge
              on this request?", trained on the user's own judged runs (RunLog). It picks the
              cheapest tier predicted to pass.

The learned router can only discover that a cheap tier is good enough for something if cheap tiers
sometimes get tried on it, so a small share of auto-routed tasks (explore_rate) goes to a random tier.
`evals/router_bench.py` measures both routers against always-smart on cost and quality.
"""
import hashlib
import random
import re

import numpy as np
from sqlmodel import select

from .config import TIERS, settings
from .db import RouterModel, RunLog, now, session

HARD = re.compile(
    r"\b(analy[sz]e|strategy|plan|design|architect|compare|evaluate|research|forecast|negotiat|"
    r"debug|refactor|prove|trade-?off|root cause|multi-step|step by step)\w*", re.I)

DIM = 256            # hashed bag-of-words buckets
MIN_SAMPLES = 15     # judged runs needed on a tier before we trust a model for it
CONFIDENCE = 0.75    # pick the cheapest tier whose predicted pass probability reaches this


def heuristic(text: str) -> str:
    hard = len(HARD.findall(text))
    words = len(text.split())
    if hard >= 2 or words > 400:
        return "smart"
    if hard == 1 or words > 60:
        return "balanced"
    return "fast"


def featurize(text: str) -> np.ndarray:
    """Hashed word counts (what is being asked) plus a few shape features (how big the ask is)."""
    vec = np.zeros(DIM + 5, dtype=np.float64)
    words = re.findall(r"[a-z]{3,}", text.lower())
    for w in words:
        vec[int.from_bytes(hashlib.md5(w.encode()).digest()[:4], "little") % DIM] += 1.0
    norm = np.linalg.norm(vec[:DIM])
    if norm:
        vec[:DIM] /= norm
    vec[DIM:] = [np.log1p(len(text.split())) / 7, min(len(HARD.findall(text)), 4) / 4,
                 float(bool(re.search(r"\d", text))), min(text.count("?"), 3) / 3, 1.0]
    return vec


def fit(X: np.ndarray, y: np.ndarray, l2: float = 0.5, steps: int = 400, lr: float = 0.5) -> np.ndarray:
    """L2-regularised logistic regression by gradient descent. Small data, so this is plenty."""
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ w)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w / len(y))
    return w


def train_from(samples: list[tuple[str, str, float]]) -> dict:
    """samples: (text, tier, score). Returns {tier: {"w": [...]} | {"const": p}} for tiers with enough data."""
    model: dict = {}
    for tier in TIERS:
        rows = [(t, s) for t, tr, s in samples if tr == tier]
        if len(rows) < MIN_SAMPLES:
            continue
        y = np.array([float(s >= settings.pass_score) for _, s in rows])
        if y.min() == y.max():  # all pass or all fail: nothing to separate, use the smoothed rate
            model[tier] = {"const": float((y.sum() + 1) / (len(y) + 2)), "n": len(rows)}
        else:
            X = np.stack([featurize(t) for t, _ in rows])
            model[tier] = {"w": fit(X, y).tolist(), "n": len(rows)}
    return model


def predict(model: dict, tier: str, text: str) -> float | None:
    m = model.get(tier)
    if not m:
        return None
    if "const" in m:
        return m["const"]
    return float(1 / (1 + np.exp(-(featurize(text) @ np.array(m["w"])))))


def learned(model: dict, text: str) -> str | None:
    """Cheapest tier predicted to pass; None if the model is not usable yet."""
    if "fast" not in model or "balanced" not in model:
        return None
    for tier in TIERS[:-1]:
        if predict(model, tier, text) >= CONFIDENCE:
            return tier
    return "smart"


# ------------------------------ per-user model storage ------------------------------

_cache: dict[str, dict] = {}


def load(user_id: str) -> dict:
    if user_id not in _cache:
        with session() as s:
            row = s.get(RouterModel, user_id)
        _cache[user_id] = row.weights if row else {}
    return _cache[user_id]


def train(user_id: str) -> dict:
    """Retrain this user's router from every judged task run they have."""
    with session() as s:
        rows = s.exec(select(RunLog).where(RunLog.user_id == user_id, RunLog.mode == "task",
                                           RunLog.score != None, RunLog.tier.in_(TIERS))).all()  # noqa: E711
        model = train_from([(r.text, r.tier, r.score) for r in rows])
        s.merge(RouterModel(user_id=user_id, weights=model, n_samples=len(rows), trained_at=now()))
        s.commit()
    _cache[user_id] = model
    return {"samples": len(rows), "tiers_trained": sorted(model), "usable": learned(model, "probe") is not None}


def pick(agent, text: str, creds) -> tuple[str, str, str]:
    """Returns (model, tier, routed_by) for this run."""
    if agent.model:
        return agent.model, "custom", "fixed"
    if agent.tier in TIERS:
        return creds.tiers[agent.tier], agent.tier, "fixed"
    if random.random() < creds.explore_rate:
        tier, how = random.choice(TIERS), "explore"
    else:
        tier = learned(load(creds.user_id), text) if creds.router_mode == "learned" else None
        how = "learned" if tier else "heuristic"
        tier = tier or heuristic(text)
    return creds.tiers[tier], tier, how
