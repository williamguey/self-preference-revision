"""Pilot v2 PREP: generate drafts (both generators, normal+rushed), keep drafts
that fail >=1 instruction, and attach EXACTLY ONE checker-verified GOOD fix per
draft (Change B). Writes usable_v2.json + a checkpoint (console + checkpoint_v2.txt)
showing (a) failed/usable counts and (b) author/fresh model assignment per draft."""
import json
import re

import or_client as oc
from ifeval_checker import check_following_list, describe_instructions
from pilot_lib import pmap, label_fix

GENERATORS = ["openai/gpt-4o-mini", "anthropic/claude-3.5-haiku"]
RUSHED_SYS = "Answer quickly and concisely; do not over-think formatting."
FIXER = "openai/gpt-4o-mini"
TARGETED_PER_FAIL = 5

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


def gen_draft(job):
    item, model, mode = job
    msgs = []
    if mode == "rushed":
        msgs.append({"role": "system", "content": RUSHED_SYS})
    msgs.append({"role": "user", "content": item["prompt"]})
    text, _ = oc.chat(model, msgs, temperature=0.7, max_tokens=1500)
    return text


def gen_fix(job):
    prompt = TARGETED.format(prompt=job["prompt"], draft=job["draft"],
                             desc=job["desc"], v=job["v"])
    text, _ = oc.chat(FIXER, [{"role": "user", "content": prompt}],
                      temperature=0.85, max_tokens=1600)
    desc, revised = parse_fix(text)
    return {"draft_idx": job["draft_idx"], "target_idx": job["target_idx"],
            "order": job["order"], "description": desc, "revised_response": revised}


def main():
    sample = json.load(open("sampled.json", encoding="utf-8"))
    djobs = [(item, m, mode)
             for m in GENERATORS for mode in ("normal", "rushed") for item in sample]
    print(f"[prep] generating {len(djobs)} drafts...")
    texts = pmap(gen_draft, djobs, workers=8)

    drafts, idx = [], 0
    for (item, model, mode), draft in zip(djobs, texts):
        if draft is None:
            continue
        res = check_following_list(item["prompt"], draft,
                                   item["instruction_id_list"], item["kwargs"])
        if sum(1 for x in res if not x) >= 1:  # keep failing only
            drafts.append({
                "draft_idx": idx, "key": item["key"], "generator": model,
                "mode": mode, "prompt": item["prompt"],
                "instruction_id_list": item["instruction_id_list"],
                "kwargs": item["kwargs"], "draft": draft, "per_instruction": res,
                "instruction_descs": describe_instructions(
                    item["prompt"], item["instruction_id_list"], item["kwargs"]),
            })
            idx += 1
    n_total = sum(1 for t in texts if t is not None)
    print(f"[prep] {len(drafts)}/{n_total} drafts FAIL >=1 instruction.")

    # targeted fix jobs (5 per failed instruction), ordered so we can pick first GOOD
    fjobs = []
    for rec in drafts:
        order = 0
        for i, ok in enumerate(rec["per_instruction"]):
            if not ok:
                for v in range(TARGETED_PER_FAIL):
                    fjobs.append({"draft_idx": rec["draft_idx"], "target_idx": i,
                                  "order": order, "prompt": rec["prompt"],
                                  "draft": rec["draft"],
                                  "desc": rec["instruction_descs"][i], "v": v + 1})
                    order += 1
    print(f"[prep] generating {len(fjobs)} targeted fixes...")
    fixes = pmap(gen_fix, fjobs, workers=8)

    by_draft = {rec["draft_idx"]: [] for rec in drafts}
    for f in fixes:
        if f:
            by_draft[f["draft_idx"]].append(f)

    usable = []
    for rec in drafts:
        cands = sorted(by_draft[rec["draft_idx"]], key=lambda c: c["order"])
        chosen = None
        seen = {rec["draft"].strip()}
        for c in cands:
            key = c["revised_response"].strip()
            if key in seen:
                continue
            seen.add(key)
            new_res = check_following_list(rec["prompt"], c["revised_response"],
                                           rec["instruction_id_list"], rec["kwargs"])
            label, fixed, broke = label_fix(rec["per_instruction"], new_res)
            if label == "GOOD":
                chosen = {"description": c["description"],
                          "revised_response": c["revised_response"],
                          "per_instruction_new": new_res, "label": "GOOD",
                          "target_idx": c["target_idx"], "fixed_idx": fixed,
                          "broke_idx": broke}
                break
        if chosen:
            author = rec["generator"]
            fresh = [m for m in GENERATORS if m != author][0]
            usable.append({**rec, "fix": chosen,
                           "author_model": author, "fresh_model": fresh})

    json.dump(usable, open("usable_v2.json", "w", encoding="utf-8"), indent=2)

    # ---------- checkpoint ----------
    lines = []
    lines.append("=" * 68)
    lines.append("PILOT v2 CHECKPOINT (before decider)")
    lines.append("=" * 68)
    by_author = {}
    for u in usable:
        by_author[u["author_model"]] = by_author.get(u["author_model"], 0) + 1
    lines.append(f"(a) drafts failing >=1 instruction: {len(drafts)}")
    lines.append(f"    usable drafts (failing AND have exactly one verified GOOD "
                 f"fix): {len(usable)}")
    lines.append(f"    usable by AUTHOR model: {by_author}")
    lines.append("")
    lines.append("(b) author/fresh assignment per usable draft:")
    lines.append(f"    {'idx':>4} {'key':>5} {'mode':<7} "
                 f"{'AUTHOR (decides X)':<26} {'FRESH (decides Z)':<26} target_constraint")
    for u in usable:
        tgt = u["instruction_id_list"][u["fix"]["target_idx"]]
        lines.append(f"    {u['draft_idx']:>4} {u['key']:>5} {u['mode']:<7} "
                     f"{u['author_model']:<26} {u['fresh_model']:<26} {tgt}")
    report = "\n".join(lines)
    print("\n" + report)
    open("checkpoint_v2.txt", "w", encoding="utf-8").write(report + "\n")
    print(f"\n[prep] est cost: ${oc.usage()['est_usd']:.4f} (calls={oc.usage()['calls']})")
    print("[prep] wrote usable_v2.json + checkpoint_v2.txt")


if __name__ == "__main__":
    main()
