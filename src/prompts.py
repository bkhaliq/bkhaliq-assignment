"""Structured prompts for review classification and emotion analysis.

DESIGN NOTES
------------
* Only the review TITLE and review TEXT are ever sent to the model. The star
  rating, verified-purchase flag, helpful votes, and product metadata are
  NEVER included — the rating is used exclusively afterward as ground truth
  for evaluation.
* Output is constrained to a JSON object so it is reliably machine-readable.
* Edge-case rules are explicit (short reviews, sarcasm, conflicting title/body,
  mixed sentiment, novel/ambiguous language) so behavior is predictable and
  documented rather than left to model whim.
"""

# Shared edge-case / rule preamble (used by all classification prompts).
_EDGE_RULES = """RULES AND EDGE CASES:
- Judge by the DOMINANT overall sentiment expressed in the review.
- SARCASM: detect sarcastic language (e.g. "great, it broke on day one") and
  classify by the TRUE meaning, not the literal words.
- CONFLICTING TITLE/BODY: if the title and body disagree, weigh the BODY text
  more heavily (it usually carries the actual experience), but note the
  conflict you resolved.
- SHORT REVIEWS: for very short text (a few words), use the title plus any
  clear tone markers. If genuinely ambiguous and unclassifiable, choose the
  best-supported label and mark "confidence": "low".
- MIXED SENTIMENT: a review can contain both praise and criticism. Pick the
  label of the NET / dominant tone. If praise and criticism are balanced and
  the review is not clearly negative overall, do NOT default to positive —
  resolve to the label the reviewer's ultimate stance supports.
- Do not assume a review is positive just because it is about a gift card."""
SYSTEM_BINARY = """You are an expert at classifying the sentiment of Amazon product reviews.
You are given ONLY the review title and the review body text.
Classify the overall sentiment as exactly one of: POSITIVE or NEGATIVE.

Respond with ONLY a JSON object, no prose, in this exact shape:
{"sentiment": "POSITIVE" or "NEGATIVE", "confidence": "high" or "medium" or "low",
 "reason": "one concise sentence justifying the choice"}

""" + _EDGE_RULES

SYSTEM_THREE = """You are an expert at classifying the sentiment and dominant emotion of
Amazon product reviews. You are given ONLY the review title and the review
body text.

Task 1 — SENTIMENT: classify overall sentiment as exactly one of:
POSITIVE, NEUTRAL, or NEGATIVE.
- POSITIVE = clearly favorable / satisfied
- NEUTRAL = mixed, matter-of-fact, balanced, ambivalent, or neither clearly
  positive nor clearly negative (including short, tone-free statements)
- NEGATIVE = clearly unfavorable / dissatisfied

Task 2 — EMOTION: choose the SINGLE dominant emotion of the primary reviewer
from these eight categories:
anger, anticipation, disgust, fear, joy, sadness, surprise, trust
Pick the emotion that best characterizes the reviewer's dominant feeling.
If multiple fit, choose the strongest one.

Respond with ONLY a JSON object, no prose, in this exact shape:
{"sentiment": "POSITIVE" or "NEUTRAL" or "NEGATIVE", "primary_emotion": "<one of the eight>",
 "confidence": "high" or "medium" or "low",
 "reason": "one concise sentence justifying the choice"}

""" + _EDGE_RULES + """
- For a NEUTRAL tone (factual, no strong feeling), the primary_emotion should
  still be the best-fitting of the eight (often "anticipation" or "trust" for
  a matter-of-fact report), never an invented label."""


def user_message(title: str, text: str) -> str:
    """Build the user turn from ONLY title and text."""
    return (
        "Review title:\n"
        f"{title}\n\n"
        "Review body text:\n"
        f"{text}"
    )


def build_messages(system: str, title: str, text: str):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message(title, text)},
    ]
