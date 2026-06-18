"""Pilot v3 PREP (powered, 4 families). Generate drafts across all working
families x normal/rushed on the v2 editable substrate, keep failing drafts, attach
exactly ONE checker-verified GOOD fix each (Change-B, no redundancy). Assign
author=true generator, fresh=rotated different family. Writes usable_v3.json +
checkpoint_v3.txt. Cost-guarded."""
import json
import re

import or_client as oc
from ifeval_checker import check_following_list, describe_instructions
from data_loader import load_ifeval
from sample_prompts import editable
from pilot_lib import pmap, label_fix

FAMILIES = ["openai/gpt-4o-mini", "anthropic/claude-3.5-haiku",
            "google/gemini-2.5-flash-lite", "meta-llama/llama-3.3-70b-instruct"]
RUSHED_SYS = "Answer quickly and concisely; do not over-think formatting."
FIXER = "openai/gpt-4o-mini"
TARGETED_PER_FAIL = 5
COST_STOP = 11.0
WORKERS = 12

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
    oc.set_pricing(FAMILIES + [FIXER])
    items = load_ifeval()
    prompts = [it for it in items if editable(it)]
    print(f"[v3 prep] {len(prompts)} editable (2-3) prompts; families={len(FAMILIES)}")

    djobs = [(it, m, mode) for m in FAMILIES for mode in ("normal", "rushed")
             for it in prompts]
    print(f"[v3 prep] generating {len(djobs)} drafts...")
    texts = pmap(gen_draft, djobs, workers=WORKERS)

    drafts, idx = [], 0
    fail_by = {}
    for (it, model, mode), draft in zip(djobs, texts):
        if draft is None:
            continue
        res = check_following_list(it["prompt"], draft,
                                   it["instruction_id_list"], it["kwargs"])
        if sum(1 for x in res if not x) >= 1:
            drafts.append({"draft_idx": idx, "key": it["key"], "generator": model,
                           "mode": mode, "prompt": it["prompt"],
                           "instruction_id_list": it["instruction_id_list"],
                           "kwargs": it["kwargs"], "draft": draft,
                           "per_instruction": res,
                           "instruction_descs": describe_instructions(
                               it["prompt"], it["instruction_id_list"], it["kwargs"])})
            idx += 1
            fail_by[model] = fail_by.get(model, 0) + 1
    print(f"[v3 prep] {len(drafts)} failing drafts. by family: "
          f"{ {k.split('/')[-1]: v for k, v in fail_by.items()} }")
    print(f"[v3 prep] cost so far ${oc.usage()['est_usd']:.3f}")

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
    print(f"[v3 prep] generating {len(fjobs)} targeted fixes...")
    fixes = pmap(gen_fix, fjobs, workers=WORKERS)

    by_draft = {rec["draft_idx"]: [] for rec in drafts}
    for f in fixes:
        if f:
            by_draft[f["draft_idx"]].append(f)

    usable = []
    for rec in drafts:
        cands = sorted(by_draft[rec["draft_idx"]], key=lambda c: c["order"])
        seen, chosen = {rec["draft"].strip()}, None
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
            cands_fresh = [m for m in FAMILIES if m != author]
            fresh = cands_fresh[len(usable) % len(cands_fresh)]  # rotate
            usable.append({**rec, "fix": chosen, "author_model": author,
                           "fresh_model": fresh})

    json.dump(usable, open("usable_v3.json", "w", encoding="utf-8"), indent=2)

    by_author = {}
    by_fresh = {}
    for u in usable:
        by_author[u["author_model"]] = by_author.get(u["author_model"], 0) + 1
        by_fresh[u["fresh_model"]] = by_fresh.get(u["fresh_model"], 0) + 1

    lines = ["=" * 70, "PILOT v3 CHECKPOINT (before decider)", "=" * 70]
    lines.append(f"working families ({len(FAMILIES)}): " + ", ".join(FAMILIES))
    lines.append(f"failing drafts: {len(drafts)}    USABLE (one verified GOOD fix): "
                 f"{len(usable)}")
    lines.append("usable by AUTHOR family: "
                 + str({k.split('/')[-1]: v for k, v in by_author.items()}))
    lines.append("usable by FRESH  family: "
                 + str({k.split('/')[-1]: v for k, v in by_fresh.items()}))
    lines.append("")
    lines.append(f"{'idx':>4} {'key':>5} {'mode':<7} {'AUTHOR':<22} {'FRESH':<22} target")
    for u in usable:
        tgt = u["instruction_id_list"][u["fix"]["target_idx"]]
        lines.append(f"{u['draft_idx']:>4} {u['key']:>5} {u['mode']:<7} "
                     f"{u['author_model'].split('/')[-1]:<22} "
                     f"{u['fresh_model'].split('/')[-1]:<22} {tgt}")
    report = "\n".join(lines)
    open("checkpoint_v3.txt", "w", encoding="utf-8").write(report + "\n")
    print("\n" + "\n".join(lines[:9]))
    print(f"\n[v3 prep] est cost ${oc.usage()['est_usd']:.3f} (calls={oc.usage()['calls']})")
    print(f"[v3 prep] wrote usable_v3.json ({len(usable)} drafts) + checkpoint_v3.txt")
    if oc.usage()["est_usd"] > COST_STOP:
        print("[v3 prep] WARNING: near cost cap.")


if __name__ == "__main__":
    main()
