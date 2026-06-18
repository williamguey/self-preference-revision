"""v3 PREP top-up: expand to len-1-3 editable prompts (unused ones) to reach the
~100 usable-draft target. Same pipeline as run_v3_prep; appends to usable_v3.json
and rewrites checkpoint_v3.txt. Cost-guarded."""
import json

import or_client as oc
from ifeval_checker import check_following_list, describe_instructions
from data_loader import load_ifeval
from sample_prompts import ALLOWED
from pilot_lib import pmap, label_fix
from run_v3_prep import (FAMILIES, FIXER, TARGETED_PER_FAIL, COST_STOP, WORKERS,
                         gen_draft, gen_fix)

TARGET_TOTAL = 100
CAP_TOTAL = 120
ADD_PROMPTS = 90  # extra prompts to draw drafts from


def editable_1_3(it):
    ids = it["instruction_id_list"]
    if not (1 <= len(ids) <= 3):
        return False
    if any(i not in ALLOWED for i in ids):
        return False
    for iid, kw in zip(ids, it["kwargs"]):
        if iid == "detectable_format:number_bullet_lists" and (kw.get("num_bullets") or 0) > 4:
            return False
    return True


def main():
    oc.set_pricing(FAMILIES + [FIXER])
    usable = json.load(open("usable_v3.json", encoding="utf-8"))
    if len(usable) >= TARGET_TOTAL:
        print(f"[topup] already {len(usable)} usable; nothing to do.")
        return
    used_keys = {u["key"] for u in usable}
    next_idx = max(u["draft_idx"] for u in usable) + 1

    items = load_ifeval()
    pool = [it for it in items if editable_1_3(it) and it["key"] not in used_keys]
    import random
    random.Random(43).shuffle(pool)
    pool = pool[:ADD_PROMPTS]
    print(f"[topup] {len(pool)} new editable(1-3) prompts; need "
          f"{TARGET_TOTAL - len(usable)} more usable.")

    djobs = [(it, m, mode) for m in FAMILIES for mode in ("normal", "rushed")
             for it in pool]
    print(f"[topup] generating {len(djobs)} drafts...")
    texts = pmap(gen_draft, djobs, workers=WORKERS)

    drafts = []
    for (it, model, mode), draft in zip(djobs, texts):
        if draft is None:
            continue
        res = check_following_list(it["prompt"], draft,
                                   it["instruction_id_list"], it["kwargs"])
        if sum(1 for x in res if not x) >= 1:
            drafts.append({"draft_idx": next_idx, "key": it["key"], "generator": model,
                           "mode": mode, "prompt": it["prompt"],
                           "instruction_id_list": it["instruction_id_list"],
                           "kwargs": it["kwargs"], "draft": draft,
                           "per_instruction": res,
                           "instruction_descs": describe_instructions(
                               it["prompt"], it["instruction_id_list"], it["kwargs"])})
            next_idx += 1
    print(f"[topup] {len(drafts)} new failing drafts; cost ${oc.usage()['est_usd']:.3f}")

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
    print(f"[topup] generating {len(fjobs)} targeted fixes...")
    fixes = pmap(gen_fix, fjobs, workers=WORKERS)
    by_draft = {rec["draft_idx"]: [] for rec in drafts}
    for f in fixes:
        if f:
            by_draft[f["draft_idx"]].append(f)

    for rec in drafts:
        if len(usable) >= CAP_TOTAL:
            break
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
            fresh = cands_fresh[len(usable) % len(cands_fresh)]
            usable.append({**rec, "fix": chosen, "author_model": author,
                           "fresh_model": fresh})

    json.dump(usable, open("usable_v3.json", "w", encoding="utf-8"), indent=2)

    by_author, by_fresh = {}, {}
    for u in usable:
        by_author[u["author_model"]] = by_author.get(u["author_model"], 0) + 1
        by_fresh[u["fresh_model"]] = by_fresh.get(u["fresh_model"], 0) + 1
    lines = ["=" * 70, "PILOT v3 CHECKPOINT (before decider) [after top-up]", "=" * 70,
             f"working families ({len(FAMILIES)}): " + ", ".join(FAMILIES),
             f"USABLE (one verified GOOD fix): {len(usable)}",
             "usable by AUTHOR family: "
             + str({k.split('/')[-1]: v for k, v in by_author.items()}),
             "usable by FRESH  family: "
             + str({k.split('/')[-1]: v for k, v in by_fresh.items()}), ""]
    lines.append(f"{'idx':>4} {'key':>5} {'mode':<7} {'AUTHOR':<22} {'FRESH':<22} target")
    for u in usable:
        tgt = u["instruction_id_list"][u["fix"]["target_idx"]]
        lines.append(f"{u['draft_idx']:>4} {u['key']:>5} {u['mode']:<7} "
                     f"{u['author_model'].split('/')[-1]:<22} "
                     f"{u['fresh_model'].split('/')[-1]:<22} {tgt}")
    open("checkpoint_v3.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[:7]))
    print(f"\n[topup] total usable {len(usable)}; est cost ${oc.usage()['est_usd']:.3f} "
          f"(calls={oc.usage()['calls']})")


if __name__ == "__main__":
    main()
