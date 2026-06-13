"""Optional LLM narrator for the backtest's worst misses.

The neural twin is the predictor; this is a *narrator* at the very end of the
pipeline. It takes the worst-miss contexts (event time, predicted vs actual, IOB/COB
proxies, the model's ISF/CR at that hour) and writes a one-sentence likely-cause
attribution per miss. Behind a flag, because it costs tokens.

Token-light by design: one batched call, structured output, low max_tokens, no
extended thinking. Fails soft — if the SDK or API key is missing, returns no
narratives and the gallery still renders.
"""

from __future__ import annotations

import json

DEFAULT_MODEL = "claude-opus-4-8"

_SYSTEM = (
    "You explain why a Type 1 diabetes glucose digital twin mispredicted blood "
    "glucose. You are a narrator, not a predictor or clinician. For each event, write "
    "ONE concise sentence (max 28 words) attributing the most likely cause of the "
    "error, grounded in the provided context and the model's internal state. Name "
    "concrete mechanisms when the data supports them: unannounced or under-counted "
    "meal (a positive error with no logged carbs near a spike), exercise or activity "
    "dip, dawn phenomenon, infusion-site failure, or stacked insulin. Do NOT give "
    "medical advice and do NOT recommend any dose or setting change. If the cause is "
    "genuinely unclear, say so."
)


def narrate_misses(
    contexts: list[dict],
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> list[str]:
    """Return one explanation per context (same order). Empty list on any failure."""
    if not contexts:
        return []
    try:
        import anthropic
    except ImportError:
        return []

    try:
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    except Exception:
        return []

    user = (
        "Here are the worst mispredictions as JSON. Errors are signed (predicted minus "
        "actual mg/dL); positive means the twin ran high vs reality. Return a JSON "
        'object: {"explanations": [one sentence per event, in the same order]}.\n\n'
        + json.dumps(contexts, indent=2)
    )
    schema = {
        "type": "object",
        "properties": {"explanations": {"type": "array", "items": {"type": "string"}}},
        "required": ["explanations"],
        "additionalProperties": False,
    }
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=900,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        if resp.stop_reason == "refusal":
            return []
        text = next((b.text for b in resp.content if b.type == "text"), "")
        explanations = json.loads(text).get("explanations", [])
    except Exception:
        return []

    # Pad/truncate defensively so the gallery indexing always lines up.
    explanations = [str(e) for e in explanations][: len(contexts)]
    explanations += [""] * (len(contexts) - len(explanations))
    return explanations
