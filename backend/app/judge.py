"""LLM-as-judge, shared by the online eval (every finished mission) and the offline eval suite."""
import json
import re

from . import llm

RUBRIC = """You are grading the final answer an AI agent team gave to a user request.

Score from 0.0 to 1.0:
- 1.0  fully answers the request, specific, grounded in the work shown, nothing invented
- 0.5  partially answers, or is vague, or misses a part of the request
- 0.0  does not answer, is wrong, or fabricates

Reply with only JSON: {"score": <number>, "reason": "<one sentence>"}"""


async def judge(key: str, model: str, request: str, answer: str, reference: str = "") -> tuple[float | None, str]:
    """Returns (score, reason). Score is None when the judge's reply could not be parsed."""
    prompt = f"<request>\n{request}\n</request>\n\n<answer>\n{answer}\n</answer>"
    if reference:
        prompt += f"\n\n<reference_answer>\n{reference}\n</reference_answer>"
    out = await llm.complete(
        key, model,
        [{"role": "system", "content": RUBRIC}, {"role": "user", "content": prompt}],
        name="judge",
    )
    match = re.search(r"\{.*\}", out["content"], re.S)
    try:
        data = json.loads(match.group(0))
        return max(0.0, min(1.0, float(data["score"]))), str(data.get("reason", ""))
    except (AttributeError, KeyError, ValueError, TypeError):
        return None, f"judge returned unparseable output: {out['content'][:120]}"
