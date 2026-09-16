#!/usr/bin/env python3
"""STEP 2 — run the binary classifier on a reproducible 100-review sample.

Ratings are used ONLY to derive ground truth; the prompt sent to the LLM
contains only title + text.
"""
import sys, json, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data import load_reviews, build_eval_frame
from src.sampling import seeded_sample
from src.llm_client import LLMClient
from src.prompts import SYSTEM_BINARY, build_messages
from src.metrics import compute_metrics, save_json

def classify_batch(rows, system=SYSTEM_BINARY, concurrency=8):
    """rows: iterable of (key, title, text). Returns {key: parsed json}."""
    client = LLMClient(temperature=0.0, max_tokens=250)
    from concurrent.futures import ThreadPoolExecutor

    def one(item):
        key, title, text = item
        try:
            return key, client.structured(build_messages(system, title, text), temperature=0.0)
        except Exception as e:
            return key, {"error": str(e)}

    results = {}
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for key, out in ex.map(one, rows):
            results[key] = out
    return results

def main():
    out = config.OUT
    out.mkdir(parents=True, exist_ok=True)
    print("Loading reviews...")
    df = build_eval_frame(load_reviews(), rule=config.BINARY_RULE)
    print("Sampling", config.BINARY_BATCH_N, "reviews (seed", config.RANDOM_SEED, ")")
    sample = seeded_sample(df, config.BINARY_BATCH_N, seed=config.RANDOM_SEED)

    rows = [(str(i), r["title"], r["text"]) for i, r in sample.iterrows()]
    print("Running LLM classification...")
    t0 = time.time()
    preds = classify_batch(rows)
    print(f"done in {time.time()-t0:.1f}s")

    # Assemble results table (rating kept only for ground truth / reporting)
    table_rows = []
    for i, r in sample.iterrows():
        p = preds.get(str(i), {})
        table_rows.append({
            "_eval_key": str(i),
            "rating": int(r["rating"]),
            "title": r["title"],
            "text": r["text"],
            "actual": r["ground_truth"],
            "predicted": p.get("sentiment"),
            "confidence": p.get("confidence"),
            "reason": p.get("reason"),
            "error": p.get("error"),
        })
    tbl = pd.DataFrame(table_rows)
    # save raw LLM output (keyed)
    raw = [{"_eval_key": k, "model_output": preds[k]} for k in preds]
    save_json(raw, config.STEP2_RAW)

    # rows where model errored are excluded from scoring
    scored = tbl[tbl["error"].isna()].copy()
    print(f"rows scored: {len(scored)} / {len(tbl)} (errored: {len(tbl)-len(scored)})")

    metrics = compute_metrics(scored, "actual", "predicted", labels=config.CLASS_ORDER_2)
    metrics["seed"] = config.RANDOM_SEED
    metrics["n_attempted"] = len(tbl)
    metrics["n_scored"] = len(scored)
    metrics["n_errored"] = len(tbl) - len(scored)
    metrics["note"] = ("Ratings used ONLY for ground truth; LLM prompt contained "
                       "title and text only. Sentiment: rating>=4 POSITIVE else NEGATIVE.")
    save_json(metrics, config.STEP2_METRICS)
    tbl.to_csv(config.STEP2_TABLE, index=False)
    print(f"\nwrote {config.STEP2_TABLE} ({len(tbl)} rows), {config.STEP2_METRICS}")

    # console report
    print("\n=== STEP 2 RESULTS ===")
    print(f"total       : {metrics['total']}")
    print(f"correct     : {metrics['correct']}")
    print(f"incorrect   : {metrics['incorrect']}")
    print(f"accuracy    : {metrics['accuracy']}")
    print("actual counts  :", metrics["actual_counts"])
    print("predicted counts:", metrics["predicted_counts"])
    print("confusion matrix (rows=actual, cols=pred):")
    print("  labels:", metrics["confusion_matrix_labels"])
    for row in metrics["confusion_matrix"]:
        print("  ", row)
    print("per-class:")
    for lab in config.CLASS_ORDER_2:
        print("  ", lab, metrics["per_class"][lab])
    mis = scored[scored.actual != scored.predicted]
    print(f"\n=== MISCLASSIFIED ({len(mis)}) ===")
    for _, r in mis.iterrows():
        print(f"  [{r['actual']}->{r['predicted']}] r={r['rating']} | {str(r['title'])[:45]} | {str(r['text'])[:60]}")

if __name__ == "__main__":
    main()
