"""Load the real IFEval dataset from the HuggingFace jsonl (datasets lib in this
env has an fsspec glob incompatibility, so we read the raw jsonl directly)."""
import json
import os
from huggingface_hub import hf_hub_download

LOCAL = os.path.join(os.path.dirname(__file__), "ifeval_input_data.jsonl")


def load_ifeval():
    if not os.path.exists(LOCAL):
        p = hf_hub_download("google/IFEval", "ifeval_input_data.jsonl",
                            repo_type="dataset")
        import shutil
        shutil.copy(p, LOCAL)
    items = []
    with open(LOCAL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


if __name__ == "__main__":
    items = load_ifeval()
    print(f"Loaded {len(items)} IFEval items")
    print("keys:", list(items[0].keys()))
