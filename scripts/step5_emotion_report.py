#!/usr/bin/env python3
"""STEP 5 — emotion comparison (LLM vs NRC) reported from the balanced run.
Reads the balanced results table (which carries both emotion_llm and
emotion_nrc), computes agreement metrics, and writes STEP5_METRICS."""
import sys, json
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.metrics import save_json

def main():
    tbl = pd.read_csv(config.STEP6_TABLE)
    tbl = tbl[tbl["error"].isna()] if "error" in tbl else tbl
    both = tbl[tbl["emotion_nrc"] != "neutral"].copy()
    agree = both[both["emotion_llm"] == both["emotion_nrc"]]
    dis = both[both["emotion_llm"] != both["emotion_nrc"]]

    # representative disagreement examples for the README/dashboard
    examples = []
    for _, r in dis.head(10).iterrows():
        examples.append({
            "rating": int(r["rating"]),
            "title": r["title"],
            "emotion_llm": r["emotion_llm"],
            "emotion_nrc": r["emotion_nrc"],
        })

    metrics = {
        "n_compared": int(len(both)),
        "n_total_scored": int(len(tbl)),
        "agree": int(len(agree)),
        "disagree": int(len(both) - len(agree)),
        "agreement_rate": round(len(agree)/len(both), 4) if len(both) else None,
        "llm_counts": tbl["emotion_llm"].value_counts().to_dict(),
        "nrc_counts": both["emotion_nrc"].value_counts().reindex(config.EMOTIONS, fill_value=0).to_dict(),
        "nrc_neutral_count": int((tbl["emotion_nrc"] == "neutral").sum()),
        "disagreement_examples": examples,
        "tie_rule": ("NRC primary emotion = highest summed score across the 8 categories; "
                     "ties broken by fixed canonical order anger,anticipation,disgust,fear,joy,"
                     "sadness,surprise,trust; zero-score -> 'neutral'."),
    }
    save_json(metrics, config.STEP5_METRICS)

    print("=== STEP 5 EMOTION COMPARISON ===")
    print(f"LLM vs NRC agreement: {metrics['agree']}/{metrics['n_compared']} "
          f"= {metrics['agreement_rate']}")
    print("LLM counts:", metrics["llm_counts"])
    print("NRC counts :", metrics["nrc_counts"])
    print("NRC neutral (no word-list match):", metrics["nrc_neutral_count"])
    print("\nDisagreement examples:")
    for e in examples:
        print(f"  r={e['rating']} LLM={e['emotion_llm']:14} NRC={e['emotion_nrc']:14} | {e['title'][:45]}")

if __name__ == "__main__":
    main()
