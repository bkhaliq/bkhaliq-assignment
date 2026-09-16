"""Load and prepare the Amazon Gift Cards reviews."""
import gzip, json
import pandas as pd
from . import config


def load_reviews(dataframe: bool = True):
    """Read the Gift_Cards.jsonl.gz into a list of dicts or a DataFrame."""
    rows = []
    with gzip.open(config.RAW_JSONL, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if dataframe:
        df = pd.DataFrame(rows)
        return df
    return rows


def build_eval_frame(df=None, rule=None) -> pd.DataFrame:
    """Add ground-truth labels (rating-dependent) to the frame.

    The star rating is used ONLY to derive ground truth here; it is never
    sent to the LLM (the prompt builder only passes title+text).
    """
    if df is None:
        df = load_reviews()
    rule = rule or config.BINARY_RULE
    df = df.copy()
    df["rating"] = df["rating"].astype(int)
    df["ground_truth"] = df["rating"].apply(rule)
    df = df.reset_index(drop=True)
    df["_eval_key"] = df.index.astype(str)
    return df
