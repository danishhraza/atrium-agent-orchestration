"""Router benchmark: what does difficulty routing save, and what does it cost in quality?

  cd backend
  OPENROUTER_API_KEY=sk-or-... uv run python -m evals.router_bench

Method
  1. Run every benchmark task on all three tiers and judge each answer (the "outcome matrix").
     This is the expensive step, so results are cached in data/router_bench.json and reused.
  2. Replay routing policies against the matrix. Because every task was run on every tier, any
     policy can be scored exactly, with no extra model calls:
       always-fast / always-balanced / always-smart   fixed baselines
       heuristic                                      the keyword/length rules
       learned                                        trained on the other folds (5-fold CV), so it
                                                      is never scored on a task it was trained on
       oracle                                         cheapest tier that passes: the upper bound
  3. Write the table to docs/ROUTER_REPORT.md. The headline is learned vs always-smart.

Spends real tokens: about 3 answers + 3 judge calls per task (45 tasks by default). Use --limit N for a cheap trial.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from app import llm, router
from app.config import TIERS, settings
from app.judge import judge

CACHE = Path(settings.data_dir) / "router_bench.json"
REPORT = Path(__file__).resolve().parent.parent.parent / "docs" / "ROUTER_REPORT.md"

TASKS = [
    # easy: lookups, rewrites, single-step answers
    "What is the capital of Australia?",
    "Rewrite this to sound more polite: 'Send me the report now.'",
    "Convert 72 degrees Fahrenheit to Celsius.",
    "Give me three synonyms for 'reliable'.",
    "Summarise in one sentence: 'The meeting moved from Tuesday to Thursday because the venue was double-booked.'",
    "What does the acronym CRM stand for?",
    "Write a subject line for an email announcing a 20 percent discount.",
    "Is 91 a prime number? Answer yes or no with one line of reasoning.",
    "Translate 'Where is the train station?' into Spanish.",
    "List the days of the week in reverse order.",
    "Fix the grammar: 'Me and him goes to the store yesterday.'",
    "What is 15 percent of 240?",
    "Write a two-line thank-you note to a customer for their first order.",
    "Name two common causes of a 404 error.",
    "Turn this into a bullet list: apples, pears and plums are in season; oranges are not.",
    # medium: some reasoning or structured writing
    "A SaaS plan costs $79 per seat per month with 15 percent off for annual billing. What does a 40-seat team pay per year? Show the working.",
    "Write a cold outreach email (under 120 words) to a VP of Sales at a 40-person company still using spreadsheets for their pipeline.",
    "Explain the difference between a left join and an inner join to a marketer, with a small example.",
    "A prospect says 'your price is too high'. Give three distinct responses and say when to use each.",
    "Draft a 5-question discovery call script for a logistics company evaluating a new CRM.",
    "Compare email and LinkedIn as first-touch channels for B2B outreach in a short table.",
    "Write a Python function that removes duplicates from a list while preserving order, with a docstring.",
    "A campaign had 1,800 sends, 414 opens and 37 replies. Compute open rate and reply rate and say whether reply rate is healthy for cold email.",
    "Rewrite this paragraph for a non-technical executive: 'We migrated the monolith to services, cutting p95 latency from 900ms to 310ms and deploys from weekly to daily.'",
    "Outline a one-week onboarding plan for a new sales development rep.",
    "Given churn of 3 percent a month, what share of customers remains after a year? Show the calculation.",
    "Write a polite but firm reply to a customer asking for a refund outside the 30-day policy.",
    "Suggest a naming convention for CRM deal stages and justify each stage in one line.",
    "Explain what an uptime SLA of 99.9 percent allows in downtime per month.",
    "Turn these notes into meeting minutes with owners and dates: 'Sam to send pricing Friday. Dana wants SSO details. Revisit in two weeks.'",
    # hard: multi-step analysis, trade-offs, planning under constraints
    "Design a 90-day go-to-market strategy for a CRM entering the dental-clinic market with a team of three reps and a $30k budget. Include channels, weekly targets, risks and how you would know by day 30 whether it is working.",
    "Analyze the trade-offs between usage-based and per-seat pricing for a B2B SaaS with 200 customers, 30 percent of whom have fewer than 5 seats. Recommend one, quantify the revenue risk, and plan the migration.",
    "A sales team's win rate fell from 28 percent to 19 percent over two quarters while pipeline volume rose 40 percent. Propose a root cause analysis: hypotheses ranked by likelihood, the data needed to test each, and what you would do for each outcome.",
    "Design a lead-scoring model for an outbound team: features, how to weight them without labelled data, how to validate it after a quarter, and the failure modes to watch for.",
    "Plan the migration of 50,000 customer records from spreadsheets to a CRM with zero downtime for the sales team. Cover deduplication strategy, validation, rollback and a cutover schedule.",
    "Compare three architectures for a multi-tenant document search feature (shared index with filters, index per tenant, hybrid) on isolation, cost and operational risk, and recommend one for a 500-tenant product.",
    "Our two regional sales teams disagree: East wants to discount to close faster, West wants to hold price. Evaluate both positions with a simple model of margin versus cycle time and propose a policy with guardrails.",
    "Write a step by step negotiation plan for renewing a contract where the customer has a credible competing offer 20 percent cheaper, including your walk-away point and three concessions ranked by cost to you.",
    "Forecast next quarter's revenue from: 120 open deals averaging $18k, historical stage conversion of 60/45/30 percent across three stages, and a 75-day cycle. State assumptions, give a range, and explain the biggest source of error.",
    "Design an experiment to test whether personalised first lines improve cold-email reply rates: hypothesis, sample size reasoning, randomisation, what could confound it, and the decision rule.",
    "Debug this plan: a team wants to double outbound volume by buying a 100k-contact list and emailing all of it in a week from their main domain. Explain every way this fails and design a safer alternative with numbers.",
    "Evaluate whether a 12-person startup should build its own CRM integration layer or buy one. Provide a decision framework, a 3-year cost comparison with stated assumptions, and the conditions under which the answer flips.",
    "Research-style brief: explain how an LLM agent team should divide a complex customer proposal among specialist agents, where handoffs lose information, and how to design the briefs to prevent that.",
    "Create a compensation plan for SDRs that rewards qualified meetings without encouraging low-quality bookings. Model the payout for a low, median and top performer and identify how each could game it.",
    "A company wants to enter two new regions but can staff only one. Build a strategy comparing the regions on market size, competition and ramp time with a weighted scoring model, and state what data would change the decision.",
]


# ------------------------------ step 1: the outcome matrix ------------------------------

async def collect(key: str, tasks: list[str]) -> list[dict]:
    tiers = settings.default_tiers
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    sem = asyncio.Semaphore(4)

    async def one(text: str, tier: str) -> None:
        cid = f"{tiers[tier]}::{text}"
        if cid in cache:
            return
        async with sem:
            out = await llm.complete(key, tiers[tier], [
                {"role": "system", "content": "You are a capable assistant on a business team. Answer the request directly and completely."},
                {"role": "user", "content": text}], name=f"bench-{tier}")
            score, _ = await judge(key, tiers["smart"], text, out["content"])  # strongest tier grades all three
        if score is not None:
            cache[cid] = {"score": score, "cost": float(out["usage"]["cost"] or 0)}
            CACHE.write_text(json.dumps(cache, indent=1))
            print(f"  {tier:9} score {score:.2f}  ${cache[cid]['cost']:.5f}  {text[:60]}", flush=True)

    await asyncio.gather(*(one(t, tier) for t in tasks for tier in TIERS))
    rows = [{"text": t, "results": {tier: cache.get(f"{tiers[tier]}::{t}") for tier in TIERS}} for t in tasks]
    return [r for r in rows if all(r["results"].values())]  # keep only tasks judged on every tier


# ------------------------------ step 2: replay policies ------------------------------

def evaluate(matrix: list[dict], folds: int = 5) -> dict[str, dict]:
    """Score each routing policy against a full outcome matrix. Pure function: no model calls."""
    pass_at = settings.pass_score

    def learned_choices() -> list[str]:
        choices = []
        for i, row in enumerate(matrix):
            train = [r for j, r in enumerate(matrix) if j % folds != i % folds]  # held-out fold never trains
            model = router.train_from([(r["text"], t, r["results"][t]["score"]) for r in train for t in TIERS])
            choices.append(router.learned(model, row["text"]) or router.heuristic(row["text"]))
        return choices

    def oracle(row) -> str:
        return next((t for t in TIERS if row["results"][t]["score"] >= pass_at), "smart")

    policies = {f"always-{t}": [t] * len(matrix) for t in TIERS}
    policies["heuristic"] = [router.heuristic(r["text"]) for r in matrix]
    policies["learned"] = learned_choices()
    policies["oracle"] = [oracle(r) for r in matrix]

    out = {}
    for name, choices in policies.items():
        picked = [r["results"][c] for r, c in zip(matrix, choices)]
        out[name] = {
            "quality": sum(p["score"] for p in picked) / len(picked),
            "pass_rate": sum(p["score"] >= pass_at for p in picked) / len(picked),
            "cost": sum(p["cost"] for p in picked),
            "mix": {t: choices.count(t) for t in TIERS},
        }
    base = out["always-smart"]["cost"] or 1e-12
    for stats in out.values():
        stats["cost_vs_smart"] = stats["cost"] / base - 1
    return out


def report(results: dict[str, dict], n: int) -> str:
    l, s = results["learned"], results["always-smart"]
    lines = [
        "# Router benchmark", "",
        f"{n} tasks, each run on all three tiers and judged; policies replayed against that matrix "
        "(learned router scored with 5-fold cross-validation).", "",
        f"**Learned router vs always-smart: {l['cost_vs_smart']:+.0%} cost, "
        f"{l['quality'] - s['quality']:+.2f} mean quality, {l['pass_rate'] - s['pass_rate']:+.0%} pass rate.**", "",
        f"Tiers: {settings.default_tiers}", "",
        "| Policy | Mean quality | Pass rate | Total cost | Cost vs always-smart | fast / balanced / smart |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        mix = " / ".join(str(r["mix"][t]) for t in TIERS)
        lines.append(f"| {name} | {r['quality']:.2f} | {r['pass_rate']:.0%} | ${r['cost']:.4f} | {r['cost_vs_smart']:+.0%} | {mix} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        print("Set OPENROUTER_API_KEY to run the benchmark (it calls real models).")
        return 2
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else len(TASKS)
    step = max(1, len(TASKS) // limit)
    tasks = TASKS[::step][:limit]  # a limited run still spans easy, medium and hard
    print(f"Collecting outcomes for {len(tasks)} tasks x {len(TIERS)} tiers (cached in {CACHE})")
    matrix = asyncio.run(collect(key, tasks))
    if len(matrix) < 10:
        print(f"Only {len(matrix)} fully judged tasks; need at least 10 for a meaningful comparison.")
        return 1
    text = report(evaluate(matrix), len(matrix))
    REPORT.write_text(text, encoding="utf-8")
    print("\n" + text + f"\nWritten to {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
