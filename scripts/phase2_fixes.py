"""Phase 2: for each FAILING draft, generate ONLY targeted minimal edits, each
aimed at one failed constraint's official build_description text. Score every
candidate with the official IFEval checker (the CHECKER assigns GOOD/BAD/NEUTRAL).

No decoy / aggro / stress / manufactured-bad edits. A BAD label can only arise
as a NATURAL side effect (a real fix for constraint A that happens to break a
passing constraint B). Keep drafts that end up with >=2 GOOD fixes."""
import json
import re

import or_client as oc
from ifeval_checker import check_following_list, describe_instructions
from pilot_lib import pmap, label_fix

FIXER = "openai/gpt-4o-mini"
TARGETED_PER_FAIL = 5
MAX_KEEP = 12

TARGETED = """You are an editor revising a draft response to a user's task.

TASK:
{prompt}

DRAFT:
{draft}

Make the SMALLEST possible edit to the draft so that it satisfies this specific \
requirement:
"{desc}"
Keep everything else identical -- change only what is needed to meet that \
requirement. (Variation #{v}: if there are several minimal ways, prefer a \
different one than the most obvious.)

Reply in EXACTLY this format and nothing else:
DESCRIPTION: <one concrete sentence describing the single edit you made>
---REVISED---
<the FULL response text after applying the edit>"""


def parse_fix(text):
    text = text.strip()
    text = re.sub(r"^```[a-z]*", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    if "---REVISED---" not in text:
        raise ValueError("no delimiter")
    head, revised = text.split("---REVISED---", 1)
    desc = head.replace("DESCRIPTION:", "").strip().splitlines()[0].strip()
    revised = revised.strip()
    if not desc or not revised:
        raise ValueError("empty parts")
    return desc, revised


def gen_one(job):
    prompt = TARGETED.format(prompt=job["prompt"], draft=job["draft"],
                             desc=job["desc"], v=job["v"])
    text, _ = oc.chat(FIXER, [{"role": "user", "content": prompt}],
                      temperature=0.85, max_tokens=1600)
    desc, revised = parse_fix(text)
    return {"draft_idx": job["draft_idx"], "target_idx": job["target_idx"],
            "description": desc, "revised_response": revised}


def main():
    failing = json.load(open("drafts_failing.json", encoding="utf-8"))

    jobs = []
    for rec in failing:
        descs = describe_instructions(rec["prompt"], rec["instruction_id_list"],
                                      rec["kwargs"])
        rec["_descs"] = descs
        for i, ok in enumerate(rec["per_instruction"]):
            if not ok:
                for v in range(TARGETED_PER_FAIL):
                    jobs.append({"draft_idx": rec["draft_idx"], "target_idx": i,
                                 "prompt": rec["prompt"], "draft": rec["draft"],
                                 "desc": descs[i], "v": v + 1})

    print(f"Generating {len(jobs)} targeted edits for {len(failing)} failing "
          f"drafts (fixer={FIXER})...")
    results = pmap(gen_one, jobs, workers=8)

    by_draft = {rec["draft_idx"]: [] for rec in failing}
    for r in results:
        if r:
            by_draft[r["draft_idx"]].append(r)

    all_labeled, kept = [], []
    for rec in failing:
        uniq, seen = [], {rec["draft"].strip()}
        for c in by_draft[rec["draft_idx"]]:
            key = c["revised_response"].strip()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
        labeled = []
        for c in uniq:
            new_res = check_following_list(rec["prompt"], c["revised_response"],
                                           rec["instruction_id_list"], rec["kwargs"])
            label, fixed, broke = label_fix(rec["per_instruction"], new_res)
            labeled.append({"kind": "targeted", "target_idx": c["target_idx"],
                            "description": c["description"],
                            "revised_response": c["revised_response"],
                            "per_instruction_new": new_res, "label": label,
                            "fixed_idx": fixed, "broke_idx": broke})
        n_good = sum(c["label"] == "GOOD" for c in labeled)
        n_bad = sum(c["label"] == "BAD" for c in labeled)
        n_neu = sum(c["label"] == "NEUTRAL" for c in labeled)
        gen = f"{rec['generator'].split('/')[-1]}/{rec['mode']}"
        flag = "KEEP" if n_good >= 2 else "drop"
        print(f"  draft {rec['draft_idx']:>3} key={rec['key']:>4} {gen:<24} "
              f"cands={len(labeled):>2} GOOD={n_good} BAD={n_bad} NEU={n_neu} -> {flag}")
        rec_out = {k: v for k, v in rec.items() if k != "_descs"}
        rec_out.update({"instruction_descs": rec["_descs"], "candidates": labeled,
                        "n_good": n_good, "n_bad": n_bad, "n_neutral": n_neu})
        all_labeled.append(rec_out)
        if n_good >= 2:
            kept.append(rec_out)

    # balance kept across generator/mode when capping
    if len(kept) > MAX_KEEP:
        buckets = {}
        for k in kept:
            buckets.setdefault(f"{k['generator']}/{k['mode']}", []).append(k)
        balanced, keys = [], list(buckets)
        i = 0
        while len(balanced) < MAX_KEEP and any(i < len(buckets[kk]) for kk in keys):
            for kk in keys:
                if i < len(buckets[kk]) and len(balanced) < MAX_KEEP:
                    balanced.append(buckets[kk][i])
            i += 1
        print(f"\nCapping {len(kept)} -> {MAX_KEEP} kept drafts (balanced).")
        kept = balanced

    json.dump(all_labeled, open("candidates_all.json", "w", encoding="utf-8"), indent=2)
    json.dump(kept, open("fixes.json", "w", encoding="utf-8"), indent=2)
    tg = sum(k["n_good"] for k in kept)
    tb = sum(k["n_bad"] for k in kept)
    by = {}
    for k in kept:
        kk = f"{k['generator'].split('/')[-1]}/{k['mode']}"
        by[kk] = by.get(kk, 0) + 1
    print(f"\nKept {len(kept)} usable drafts (>=2 GOOD). By generator/mode: {by}")
    print(f"Pooled GOOD={tg}, pooled BAD={tb} (BAD = natural side effects only).")
    print(f"Est cost so far: ${oc.usage()['est_usd']:.4f}  (calls={oc.usage()['calls']})")


if __name__ == "__main__":
    main()
