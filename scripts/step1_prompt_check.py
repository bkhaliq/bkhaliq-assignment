#!/usr/bin/env python3
"""STEP 1 — validate the structured prompt on hand-picked reviews (no star
ratings are sent to the model; these cases are chosen for their content)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llm_client import LLMClient
from src.prompts import SYSTEM_BINARY, build_messages

CASES = [
    ("obvious positive", "Great gift", "Having Amazon money is always good."),
    ("obvious positive 2", "Perfect! Arrived just when stated!", "Arrived quickly and exactly as described. Very happy."),
    ("obvious negative", "Card did not work", "Card did not work!!!! The code would not redeem. Terrible."),
    ("obvious negative 2", "Mistake", "Hit button by mistake and deduction from credit card on file."),
    ("sarcasm", "Yeah, fantastic", "Great, the card arrived with zero balance. Love losing money."),
    ("short positive", "Five Stars", "Great"),
    ("conflicting title/body", "Unhappy with purchase", "Despite the title, everything was fine and I got what I paid for. Happy overall."),
    ("mixed sentiment", "Mostly good", "The card worked fine but the delivery took a week longer than promised. Would buy again."),
    ("neutral/factual", "Gift card", "It is a gift card. Valid amount."),
]

def main():
    client = LLMClient(temperature=0.0, max_tokens=250)
    print("=" * 70)
    for name, title, text in CASES:
        msgs = build_messages(SYSTEM_BINARY, title, text)
        try:
            out = client.structured(msgs, temperature=0.0)
            print(f"\n--- {name} ---")
            print(f"  title: {title!r}")
            print(f"  text : {text[:70]!r}")
            print(f"  -> {json.dumps(out)}")
        except Exception as e:
            print(f"\n--- {name} --- ERROR: {e}")

if __name__ == "__main__":
    main()
