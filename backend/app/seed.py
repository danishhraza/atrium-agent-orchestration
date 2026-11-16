"""A demo team so a new user has something to look at: a small sales org with per-rep call lists."""
from sqlmodel import select

from . import rag
from .db import Agent, AgentBox, Box, session

PRODUCT = """Atlas CRM pricing (2026)

Starter plan: $29 per seat per month. Up to 5 seats. Email support.

Growth plan: $79 per seat per month. Unlimited seats, workflow automation, phone support.
Annual billing on Growth gets a 15 percent discount.

Enterprise plan: custom pricing, SSO, dedicated success manager, 99.9 percent uptime SLA.

Objection handling: if a prospect says the price is too high, lead with the automation savings:
customers report saving 6 hours per rep per week."""

EAST = """Call list - East region

Northwind Traders - contact Dana Whitfield, VP Sales - 40 reps - currently on spreadsheets - warm lead from webinar.

Harbor Logistics - contact Sam Okafor, RevOps lead - 120 reps - contract with a competitor ends in March.

Brightline Dental Group - contact Priya Raman, COO - 12 reps - price sensitive, asked for Starter pricing."""

WEST = """Call list - West region

Redwood Robotics - contact Lena Fischer, Head of Sales - 65 reps - wants workflow automation.

Pacific Solar Co - contact Marcus Hale, CRO - 200 reps - needs SSO and an uptime SLA.

Tidewater Brewing - contact Jo Alvarez, owner - 4 reps - first CRM, wants something simple."""


async def seed_demo(user_id: str, key: str) -> list[str]:
    with session() as s:
        if s.exec(select(Agent).where(Agent.user_id == user_id, Agent.is_orchestrator == False)).first():  # noqa: E712
            return []
        boxes = {name: Box(user_id=user_id, name=name, description=desc) for name, desc in [
            ("Product Docs", "pricing and objection handling"),
            ("Call List East", "East region prospects"),
            ("Call List West", "West region prospects"),
        ]}
        lead = Agent(user_id=user_id, name="Morgan", tier="smart",
                     role="Sales lead: plans outreach campaigns and coordinates the regional reps")
        s.add_all([*boxes.values(), lead])
        s.flush()
        east = Agent(user_id=user_id, name="Eli", parent_id=lead.id, tier="auto",
                     role="Sales rep for the East region: knows the East call list and drafts outreach to those prospects")
        west = Agent(user_id=user_id, name="Wren", parent_id=lead.id, tier="auto",
                     role="Sales rep for the West region: knows the West call list and drafts outreach to those prospects")
        analyst = Agent(user_id=user_id, name="Quinn", tier="fast", tools=["calculator", "web_fetch", "current_time"],
                        role="Pricing analyst: answers pricing and discount questions and does the math")
        s.add_all([east, west, analyst])
        s.flush()
        access = [(lead, ["Product Docs"]), (east, ["Product Docs", "Call List East"]),
                  (west, ["Product Docs", "Call List West"]), (analyst, ["Product Docs"])]
        for agent, names in access:
            for n in names:
                s.add(AgentBox(agent_id=agent.id, box_id=boxes[n].id))
        s.commit()
        ids = {n: b.id for n, b in boxes.items()}
        created = [lead.name, east.name, west.name, analyst.name]
    for name, text in [("Product Docs", PRODUCT), ("Call List East", EAST), ("Call List West", WEST)]:
        await rag.ingest(key, user_id, ids[name], f"{name.lower().replace(' ', '-')}.txt", text.encode())
    return created
