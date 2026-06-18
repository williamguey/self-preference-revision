"""Sample IFEval prompts whose EVERY instruction is an 'editable' type -- one a
single minimal edit can plausibly flip. Excludes count-heavy constraints
(sentence/word/paragraph counts, large letter-frequency) that resist atomic fixes
and only yield NEUTRALs. Reproducible via fixed seed."""
import json
import random

from data_loader import load_ifeval

SEED = 42
N = 45  # raised from 30: target ~18-20 usable drafts after failure+one-fix filter

# Whitelist of atomically-editable instruction types (+ no_comma per user).
ALLOWED = {
    "keywords:existence",                       # add a required word
    "keywords:frequency",                       # raise/lower a word's count
    "keywords:forbidden_words",                 # remove a banned word
    "change_case:english_capital",              # ALL CAPS
    "change_case:english_lowercase",            # all lowercase
    "detectable_format:title",                  # add <<title>>
    "detectable_content:postscript",            # add P.S.
    "detectable_content:number_placeholders",   # add [ ] placeholders
    "detectable_format:number_highlighted_sections",  # add *italic* sections
    "detectable_format:number_bullet_lists",    # bullets (small counts)
    "startend:end_checker",                     # end with exact phrase
    "startend:quotation",                       # wrap in double quotes
    "punctuation:no_comma",                     # remove commas
}


def editable(item):
    ids = item["instruction_id_list"]
    if not (2 <= len(ids) <= 3):
        return False
    if any(i not in ALLOWED for i in ids):
        return False
    # bullet points: small counts only (<=4)
    for iid, kw in zip(ids, item["kwargs"]):
        if iid == "detectable_format:number_bullet_lists":
            if (kw.get("num_bullets") or 0) > 4:
                return False
    return True


def main():
    items = load_ifeval()
    pool = [it for it in items if editable(it)]
    rng = random.Random(SEED)
    rng.shuffle(pool)
    sample = pool[:N]
    json.dump(sample, open("sampled.json", "w", encoding="utf-8"), indent=2)

    print(f"{len(pool)} IFEval prompts are all-editable & 2-3 instructions; "
          f"sampled {len(sample)} (seed={SEED}).\n")
    for i, it in enumerate(sample):
        n = len(it["instruction_id_list"])
        print(f"--- [{i}] key={it['key']}  ({n} instr) ---")
        print("  instructions:", it["instruction_id_list"])
        kw = [{k: v for k, v in d.items() if v is not None} for d in it["kwargs"]]
        print("  kwargs:", kw)
        oneline = " ".join(it["prompt"].split())
        print("  prompt:", oneline[:240] + ("..." if len(oneline) > 240 else ""))
        print()


if __name__ == "__main__":
    main()
