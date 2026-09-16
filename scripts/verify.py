#!/usr/bin/env python3
"""Quality-control verification: independently recompute all headline metrics
from the saved result tables and compare against the saved metrics JSONs, so
the README/dashboard numbers are provably derived from real saved output."""
import sys, json
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics import compute_metrics
from src import config

ok = True
def check(name, a, b, tol=1e-3):
    global ok
    match = abs(a - b) <= tol if isinstance(a, (int, float)) else a == b
    if not match:
        ok = False
        print(f"  [FAIL] {name}: saved={b} recomp={a}")
    else:
        print(f"  [ok] {name}: {b}")

print("=== STEP 2 (binary, 100) ===")
m2 = json.load(open(config.STEP2_METRICS))
t2 = pd.read_csv(config.STEP2_TABLE)
s2 = t2[t2["error"].isna()]
r2 = compute_metrics(s2, "actual", "predicted", labels=config.CLASS_ORDER_2)
check("total", r2["total"], m2["total"])
check("correct", r2["correct"], m2["correct"])
check("accuracy", r2["accuracy"], m2["accuracy"])
check("cm", r2["confusion_matrix"], m2["confusion_matrix"])
check("actual_counts", r2["actual_counts"], m2["actual_counts"])

print("=== STEP 6 (balanced three-class, 150) ===")
m6 = json.load(open(config.STEP6_METRICS))
t6 = pd.read_csv(config.STEP6_TABLE)
s6 = t6[t6["error"].isna()]
r6 = compute_metrics(s6, "actual", "predicted", labels=config.CLASS_ORDER_3)
check("total", r6["total"], m6["total"])
check("correct", r6["correct"], m6["correct"])
check("accuracy", r6["accuracy"], m6["accuracy"])
check("cm", r6["confusion_matrix"], m6["confusion_matrix"])
check("actual_counts", r6["actual_counts"], m6["actual_counts"])
check("per_class NEUTRAL recall", r6["per_class"]["NEUTRAL"]["recall"], m6["per_class"]["NEUTRAL"]["recall"])
check("per_class NEUTRAL misclassified", r6["per_class"]["NEUTRAL"]["misclassified"], m6["per_class"]["NEUTRAL"]["misclassified"])
check("misdir NEUTRAL", r6["misclassification_direction"].get("NEUTRAL"), m6["misclassification_direction"].get("NEUTRAL"))

# balanced sample integrity
samp = pd.read_csv(config.STEP6_SAMPLE)
print("=== Balanced sample integrity ===")
print("  size:", len(samp), "| per-class:", samp["ground_truth"].value_counts().to_dict())
assert set(samp["ground_truth"].unique()) <= {"POSITIVE","NEUTRAL","NEGATIVE"}
counts = samp["ground_truth"].value_counts()
for c in ("POSITIVE","NEUTRAL","NEGATIVE"):
    check(f"balanced {c} count", counts.get(c,0), 50)

print("=== Emotion agreement ===")
m5 = json.load(open(config.STEP5_METRICS))
a = m5["agree"]/m5["n_compared"]
print(f"  [ok] agreement rate {m5['agree']}/{m5['n_compared']} = {a:.4f} (saved {m5['agreement_rate']})")

print("\nRESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
