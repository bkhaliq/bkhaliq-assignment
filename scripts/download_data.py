#!/usr/bin/env python3
"""Download the Amazon Reviews 2023 Gift Cards dataset (McAuley Lab)."""
import sys, gzip
from pathlib import Path
import urllib.request
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

def main():
    config.RAW_JSONL.parent.mkdir(parents=True, exist_ok=True)
    if config.RAW_JSONL.exists():
        print(f"already present: {config.RAW_JSONL} ({config.RAW_JSONL.stat().st_size/1e6:.1f} MB)")
        return
    print(f"downloading {config.DATA_URL}")
    tmp = config.RAW_JSONL.with_suffix(".gz.part")
    urllib.request.urlretrieve(config.DATA_URL, tmp)
    tmp.replace(config.RAW_JSONL)
    # quick sanity: count lines
    n = 0
    with gzip.open(config.RAW_JSONL, "rt", encoding="utf-8") as f:
        for _ in f:
            n += 1
    print(f"saved {config.RAW_JSONL} — {n} reviews")

if __name__ == "__main__":
    main()
