"""Project-wide configuration and constants (no secrets here)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Reproducibility
RANDOM_SEED = 6418          # course number, fixed for reproducible sampling
BINARY_BATCH_N = 100        # STEP 2: initial evaluation batch size
BALANCED_PER_CLASS = 50     # STEP 6: per-class target for balanced sample

# Data
RAW_JSONL = PROJECT_ROOT / "data/raw/Gift_Cards.jsonl.gz"
DATA_URL = ("https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
            "review_categories/Gift_Cards.jsonl.gz")

# Output paths
OUT = PROJECT_ROOT / "outputs"
STEP2_RAW = OUT / "step2_binary_raw.jsonl"      # raw LLM predictions for the 100-row run
STEP2_METRICS = OUT / "step2_binary_metrics.json"
STEP2_TABLE = OUT / "step2_binary_table.csv"    # review-level results (dashboard source)
STEP5_RAW = OUT / "step5_emotion_raw.jsonl"     # the emotion run (both LLM emotion + NRC)
STEP5_METRICS = OUT / "step5_emotion_metrics.json"
STEP6_RAW = OUT / "step6_balanced_raw.jsonl"    # raw LLM predictions for the balanced run
STEP6_SAMPLE = OUT / "step6_balanced_sample.csv"
STEP6_METRICS = OUT / "step6_balanced_metrics.json"
STEP6_TABLE = OUT / "step6_balanced_table.csv"
FINAL_DASHBOARD = PROJECT_ROOT / "dashboard.html"

# Data dirs
SRC = PROJECT_ROOT / "src"
DATA = PROJECT_ROOT / "data"
NRC_LEXICON = DATA / "nrc_emotion_en.tsv"

# Sentiment label mappings
# Binary (Steps 1-5): rating >= 4 -> POSITIVE, rating < 4 -> NEGATIVE
BINARY_LABELS = {"POSITIVE", "NEGATIVE"}
BINARY_RULE = lambda r: "POSITIVE" if r >= 4 else "NEGATIVE"

# Three-class (Step 6+): 4-5 -> POSITIVE, 3 -> NEUTRAL, 1-2 -> NEGATIVE
THREE_LABELS = {"POSITIVE", "NEUTRAL", "NEGATIVE"}
THREE_RULE = lambda r: "POSITIVE" if r >= 4 else ("NEUTRAL" if r == 3 else "NEGATIVE")

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

CLASS_ORDER_3 = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
CLASS_ORDER_2 = ["POSITIVE", "NEGATIVE"]
