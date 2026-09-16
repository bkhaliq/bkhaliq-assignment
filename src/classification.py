"""Shared review-classification pipeline (LLM structured output)."""
from concurrent.futures import ThreadPoolExecutor
from src.llm_client import LLMClient
from src.prompts import build_messages


def classify_rows(rows, system, concurrency=8, temperature=0.0, max_tokens=500):
    """rows: iterable of (key, title, text). Returns {key: parsed json dict}.

    The prompt built for each row contains ONLY title+text (see prompts.py);
    star ratings are never included.
    """
    client = LLMClient(temperature=temperature, max_tokens=max_tokens)

    def one(item):
        key, title, text = item
        try:
            return key, client.structured(build_messages(system, title, text),
                                          temperature=temperature)
        except Exception as e:
            return key, {"error": str(e)}

    results = {}
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for key, out in ex.map(one, rows):
            results[key] = out
    return results
