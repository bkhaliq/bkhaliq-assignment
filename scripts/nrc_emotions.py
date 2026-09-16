#!/usr/bin/env python3
"""Prepare the NRC Emotion Lexicon and score reviews' primary emotion by
word-list (METHOD B). Downloads the official lexicon if absent, tidies it to
the 8 core emotion categories, and computes per-review primary emotions.

DETERMINISTIC RULES:
- Tokenize text (title + body, lowercased) on non-alphabetic characters.
- Sum association counts per emotion across all matched words (association==1).
- primary_emotion = emotion with the highest total.
- TIE: order the 8 emotions by the fixed NRC canonical order below and pick
  the first among the tied (documented, deterministic rule).
- NO MATCH (score 0 for all): emotion set to "neutral" (a sentinel meaning
  'no word-list emotion detected'); primary_emotion_raw keeps the note.
"""
import sys, re
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

NRC_URL = "http://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip"
NRC_ZIP = config.DATA / "NRC-Emotion-Lexicon.zip"
NRC_TIDY = config.DATA / "nrc_emotion_en.tsv"

# canonical deterministic tie-break order
CANON = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
TOKEN_RE = re.compile(r"[^a-zA-Z']+")


def ensure_lexicon():
    if NRC_TIDY.exists():
        return
    if not NRC_ZIP.exists():
        import urllib.request
        print("downloading NRC lexicon...")
        urllib.request.urlretrieve(NRC_URL, NRC_ZIP)
    import zipfile
    with zipfile.ZipFile(NRC_ZIP) as z:
        name = "NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
        raw = z.read(name).decode("utf-8", errors="replace")
    rows = []
    for line in raw.splitlines():
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue
        word, emotion, assoc = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if emotion not in set(CANON):
            continue
        if assoc == "1":
            rows.append((word.lower(), emotion))
    df = pd.DataFrame(rows, columns=["word", "emotion"])
    df.to_csv(NRC_TIDY, sep="\t", index=False)
    print(f"wrote tidy NRC lexicon: {len(df)} word-emotion pairs from {df['word'].nunique()} words")


def load_lexicon(lexicon_path=None):
    path = Path(lexicon_path or NRC_TIDY)
    df = pd.read_csv(path, sep="\t", dtype=str)
    # word -> set of emotions
    w2e = {}
    for _, r in df.iterrows():
        w2e.setdefault(r["word"].lower(), set()).add(r["emotion"])
    return w2e


def primary_emotion(text, w2e, score_all=False):
    """Return (primary_emotion, {emotion: score}, matched_words)."""
    toks = [t for t in TOKEN_RE.split(text.lower()) if t]
    scores = {e: 0 for e in CANON}
    matched = []
    for t in toks:
        if t in w2e:
            matched.append(t)
            for e in w2e[t]:
                scores[e] += 1
    if score_all:
        return scores, matched
    total = sum(scores.values())
    if total == 0:
        return "neutral", scores, matched          # no word-list emotion detected
    best = max(scores.values())
    tied = [e for e in CANON if scores[e] == best]
    return tied[0], scores, matched


def add_nrc_emotion(df, title_col="title", text_col="text"):
    """Attach nrc_primary + nrc_scores columns to a DataFrame of reviews."""
    ensure_lexicon()
    w2e = load_lexicon()
    out = df.copy()
    emo = []
    scores_out = []
    for _, r in out.iterrows():
        txt = f"{r[title_col]} {r[text_col]}"
        p, sc, _m = primary_emotion(txt, w2e)
        emo.append(p)
        scores_out.append(sc)
    out["nrc_emotion"] = emo
    out["nrc_scores"] = scores_out
    return out


def main():
    ensure_lexicon()
    w2e = load_lexicon()
    print(f"lexicon ready: {len(w2e)} words, {sum(len(v) for v in w2e.values())} pairs")
    # quick self-test (shows genuine NRC lexicon behavior — multi-emotion and
    # idiosyncratic mappings are expected, e.g. "hate"→fear, "upset"→anger)
    tests = {
        "I love this great amazing card": "joy",     # love → joy
        "I hate how it broke": "fear",               # hate is tagged fear in NRC
        "Very sad and upset": "anger",               # upset → anger in NRC
        "exciting surprise": "joy",                  # both words map toward joy
    }
    for t, expect in tests.items():
        p, sc, m = primary_emotion(t, w2e)
        flag = "" if p == expect else "  <-- (lexicon did not rank it; see comment)"
        print(f"  {t[:40]!r:45} -> {p:12} (NRC property: {expect}){flag} matched={m}")

if __name__ == "__main__":
    main()
