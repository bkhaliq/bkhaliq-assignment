#!/usr/bin/env python3
"""STEP 6 — three-class balanced model on a reproducible 50/50/50 sample.

Changes vs STEP 2:
  - sentiment definition becomes 4-5 POSITIVE / 3 NEUTRAL / 1-2 NEGATIVE
  - LLM prompt returns sentiment + primary_emotion (STEP 5 method A)
  - sample is drawn balanced 50/50/50 from the WHOLE dataset (fixed seed)
Star ratings are used ONLY for ground truth; the prompt contains title+text.
"""
import sys, time, json
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data import load_reviews, build_eval_frame
from src.classification import classify_rows
from src.prompts import SYSTEM_THREE
from src.metrics import compute_metrics, save_json
from scripts.nrc_emotions import add_nrc_emotion

def main():
    out = config.OUT
    out.mkdir(parents=True, exist_ok=True)
    seed = config.RANDOM_SEED
    print("Loading full dataset...")
    df3 = build_eval_frame(load_reviews(), rule=config.THREE_RULE)

    # ---- balanced sample (50/50/50) ----
    rng = __import__("numpy").random.default_rng(seed)
    classes = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
    parts = []
    for i, cls in enumerate(classes):
        sub = df3[df3["ground_truth"] == cls]
        take = rng.choice(sub.index, size=config.BALANCED_PER_CLASS, replace=False)
        parts.append(sub.loc[take])
    sample = pd.concat(parts).sample(frac=1.0, random_state=seed)
    sample.to_csv(config.STEP6_SAMPLE, index=False)
    print(f"balanced sample: {len(sample)} rows -> "
          f"{sample.ground_truth.value_counts().to_dict()} (seed {seed})")

    # ---- LLM classification (sentiment + emotion) ----
    rows = [(str(i), r["title"], r["text"]) for i, r in sample.iterrows()]
    print("Running LLM (three-class + emotion)...")
    t0 = time.time()
    preds = classify_rows(rows, SYSTEM_THREE, concurrency=8)
    print(f"done in {time.time()-t0:.1f}s")

    # ---- NRC emotion (method B) ----
    print("Computing NRC word-list emotion...")
    sample = add_nrc_emotion(sample)

    # ---- assemble results ----
    records = []
    for i, r in sample.iterrows():
        p = preds.get(str(i), {})
        nrc_scores = r.get("nrc_scores") or {}
        records.append({
            "_eval_key": str(i),
            "rating": int(r["rating"]),
            "title": r["title"],
            "text": r["text"],
            "actual": r["ground_truth"],
            "predicted": p.get("sentiment"),
            "confidence": p.get("confidence"),
            "reason": p.get("reason"),
            "emotion_llm": p.get("primary_emotion"),
            "emotion_nrc": r.get("nrc_emotion"),
            "nrc_scores": nrc_scores,
            "error": p.get("error"),
        })
    tbl = pd.DataFrame(records)

    raw = [{"_eval_key": k, "model_output": preds[k]} for k in preds]
    save_json(raw, config.STEP6_RAW)
    tbl.to_csv(config.STEP6_TABLE, index=False)
    print(f"wrote {config.STEP6_SAMPLE}, {config.STEP6_RAW}, {config.STEP6_TABLE}")

    scored = tbl[tbl["error"].isna()].copy()
    print(f"scored {len(scored)}/{len(tbl)} (errored {len(tbl)-len(scored)})")

    metrics = compute_metrics(scored, "actual", "predicted", labels=config.CLASS_ORDER_3)
    metrics["seed"] = seed
    metrics["sample_size"] = len(sample)
    metrics["n_attempted"] = len(tbl)
    metrics["n_scored"] = len(scored)
    metrics["n_errored"] = len(tbl) - len(scored)

    # emotion agreement (LLM vs NRC) over scored rows without NRC 'neutral'
    both = scored[scored["emotion_nrc"] != "neutral"].copy()
    agree = both[both["emotion_llm"] == both["emotion_nrc"]]
    metrics["emotion"] = {
        "n_compared": int(len(both)),
        "agree": int(len(agree)),
        "disagree": int(len(both) - len(agree)),
        "agreement_rate": round(len(agree) / len(both), 4) if len(both) else None,
        "llm_counts": scored["emotion_llm"].value_counts().to_dict(),
        "nrc_counts": both["emotion_nrc"].value_counts().reindex(
            config.EMOTIONS, fill_value=0).to_dict(),
        "nrc_neutral_count": int((scored["emotion_nrc"] == "neutral").sum()),
    }
    metrics["3star_note"] = ("Three-class rule: 4-5=POSITIVE, 3=NEUTRAL, 1-2=NEGATIVE. "
                             "Balanced 50/50/50 sample from the full dataset, seed 6418.")
    save_json(metrics, config.STEP6_METRICS)

    # ---- console report ----
    print("\n=== STEP 6 (balanced three-class) RESULTS ===")
    print("sample counts:", metrics["sample_size"], "| scored:", metrics["n_scored"],
          "| accuracy:", metrics["accuracy"])
    print("confusion matrix (rows=actual POSITIVE,NEUTRAL,NEGATIVE; cols=pred):")
    for row in metrics["confusion_matrix"]:
        print("  ", row)
    print("per-class:")
    for lab in config.CLASS_ORDER_3:
        print("  ", lab, metrics["per_class"][lab])
    print("misclassification direction:", metrics["misclassification_direction"])
    print("NEUTRAL actual -> predicted:", json.dumps(
        metrics["misclassification_direction"].get("NEUTRAL", {})))
    e = metrics["emotion"]
    print(f"\nEmotion agreement (LLM vs NRC): {e['agree']}/{e['n_compared']} = {e['agreement_rate']}")
    print("LLM emotion counts:", e["llm_counts"])
    print("NRC emotion counts:", e["nrc_counts"])
    print("NRC neutral (no match):", e["nrc_neutral_count"])

    # disagreement examples
    dis = both[both["emotion_llm"] != both["emotion_nrc"]]
    print(f"\n=== DISAGREEMENT EXAMPLES ({len(dis)}) ===")
    for _, r in dis.head(12).iterrows():
        print(f"  LLM={r['emotion_llm']:14} NRC={r['emotion_nrc']:14} | {str(r['title'])[:40]}")

if __name__ == "__main__":
    main()
