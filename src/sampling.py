"""Reproducible sampling helpers (fixed seed)."""
import numpy as np
import pandas as pd
from . import config


def seeded_sample(df, n, seed=None, by_class=None):
    """Draw a reproducible sample.

    by_class=None -> simple random sample of n rows.
    by_class={'POSITIVE': 50, ...} -> stratified random sample keyed on df['class_col'].
    """
    seed = config.RANDOM_SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    if by_class is None:
        idx = rng.choice(df.index, size=min(n, len(df)), replace=False)
        return df.loc[idx].copy()
    # stratified by the value of df.loc[:, 'class_col']
    parts = []
    col = "class_col"
    for cls, k in by_class.items():
        sub = df[df[col] == cls]
        k = min(k, len(sub))
        if k == 0:
            continue
        take = rng.choice(sub.index, size=k, replace=False)
        parts.append(sub.loc[take])
    out = pd.concat(parts)
    return out.sample(frac=1.0, random_state=seed)
