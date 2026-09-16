#!/usr/bin/env python3
"""Inspect the Amazon Gift Card reviews dataset."""
import gzip, json, sys
from pathlib import Path
import pandas as pd

RAW = Path("data/raw/Gift_Cards.jsonl.gz")
assert RAW.exists(), f"missing {RAW}"

recs = []
with gzip.open(RAW, "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        recs.append(json.loads(line))

df = pd.DataFrame(recs)
print("TOTAL REVIEWS:", len(df))
print("\n=== COLUMNS ===")
print(list(df.columns))
print("\n=== DIMS ===", df.shape)
print("\n=== DTYPES ===")
print(df.dtypes)
print("\n=== MISSING VALUES (per column) ===")
print(df.isna().sum()[df.isna().sum() > 0].to_string())
print("\n=== RATING DISTRIBUTION ===")
print(df["rating"].value_counts().sort_index().to_string())
print("\n=== RATING VALUE COUNTS (star level) ===")
print(df["rating"].value_counts().sort_index().rename_axis("stars").rename("count").to_string())
print("\n=== SAMPLE RECORDS (first 3) ===")
for i in range(3):
    print(json.dumps(recs[i], ensure_ascii=False)[:600]); print("-"*40)
print("\n=== TITLE length stats / TEXT length stats ===")
print("title chars: min/median/max", int(df["title"].str.len().min()), int(df["title"].str.len().median()), int(df["title"].str.len().max()))
print("text  chars: min/median/max", int(df["text"].str.len().min()), int(df["text"].str.len().median()), int(df["text"].str.len().max()))
print("\n=== verified purchase counts ===")
print(df["verified_purchase"].value_counts(dropna=False).to_string())
print("\n=== helpful_vote stats ===")
print(df["helpful_vote"].describe().to_string())
print("\n=== timestamp range (ms epoch) ===")
print(df["timestamp"].min(), "->", df["timestamp"].max())
print("\n=== unique asins:", df["asin"].nunique(), "| unique users:", df["user_id"].nunique())
