"""Shared helpers for the anchoring pilot."""
import json
import re
from concurrent.futures import ThreadPoolExecutor


def pmap(fn, items, workers=6):
    """Parallel map preserving order. Exceptions -> None."""
    out = [None] * len(items)
    def wrap(i_item):
        i, item = i_item
        try:
            return i, fn(item)
        except Exception as e:  # noqa
            print(f"  [warn] item {i} failed: {e}")
            return i, None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in ex.map(wrap, list(enumerate(items))):
            out[i] = r
    return out


def label_fix(orig_results, new_results):
    """Compare per-instruction pass/fail of draft vs a candidate's revised text.

    GOOD: flips >=1 failing instruction to pass AND breaks none.
    BAD:  breaks >=1 currently-passing instruction.
    NEUTRAL: no instruction outcome changed.
    (A change that both fixes and breaks counts BAD: it broke a passing one.)
    """
    fixed, broke = [], []
    for i, (o, n) in enumerate(zip(orig_results, new_results)):
        if (not o) and n:
            fixed.append(i)
        elif o and (not n):
            broke.append(i)
    if broke:
        label = "BAD"
    elif fixed:
        label = "GOOD"
    else:
        label = "NEUTRAL"
    return label, fixed, broke


def extract_json(text):
    """Pull the first JSON object out of a model response."""
    text = text.strip()
    # strip code fences
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])


_DECISION_RE = re.compile(r"(\d+)\s*[\.\):\-]*\s*\**\s*(ACCEPT|REJECT)", re.I)


def parse_decisions(text, n_expected):
    """Parse 'N. ACCEPT/REJECT' lines into a dict {index(0-based): 'ACCEPT'/'REJECT'}."""
    out = {}
    for m in _DECISION_RE.finditer(text):
        num = int(m.group(1)) - 1
        if 0 <= num < n_expected and num not in out:
            out[num] = m.group(2).upper()
    return out
