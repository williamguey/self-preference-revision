"""Phase 1: generate drafts for each sampled (all-editable) prompt from BOTH
mid-tier models, in NORMAL and RUSHED modes (rushed = a system message that
discourages careful formatting, to push the natural failure rate up). Run the
official IFEval checker; KEEP drafts that fail >=1 instruction. The true
generator/mode are recorded; OWN/OTHER in phase 3 is a told-label manipulation
applied identically to every kept draft."""
import json

import or_client as oc
from ifeval_checker import check_following_list
from pilot_lib import pmap

GENERATORS = ["openai/gpt-4o-mini", "anthropic/claude-3.5-haiku"]
RUSHED_SYS = "Answer quickly and concisely; do not over-think formatting."


def gen_one(job):
    item, model, mode = job
    msgs = []
    if mode == "rushed":
        msgs.append({"role": "system", "content": RUSHED_SYS})
    msgs.append({"role": "user", "content": item["prompt"]})
    text, _ = oc.chat(model, msgs, temperature=0.7, max_tokens=1500)
    return text


def main():
    sample = json.load(open("sampled.json", encoding="utf-8"))
    jobs = [(item, m, mode)
            for m in GENERATORS for mode in ("normal", "rushed") for item in sample]
    print(f"Generating {len(jobs)} drafts ({len(sample)} prompts x "
          f"{len(GENERATORS)} models x 2 modes, temp 0.7)...")
    texts = pmap(gen_one, jobs, workers=8)

    out, idx = [], 0
    nfail_by = {}
    for (item, model, mode), draft in zip(jobs, texts):
        if draft is None:
            continue
        res = check_following_list(item["prompt"], draft,
                                   item["instruction_id_list"], item["kwargs"])
        n_fail = sum(1 for x in res if not x)
        rec = {
            "draft_idx": idx, "key": item["key"], "generator": model, "mode": mode,
            "prompt": item["prompt"],
            "instruction_id_list": item["instruction_id_list"],
            "kwargs": item["kwargs"], "draft": draft,
            "per_instruction": res, "n_failed": n_fail,
        }
        out.append(rec)
        idx += 1
        if n_fail:
            k = f"{model.split('/')[-1]}/{mode}"
            nfail_by[k] = nfail_by.get(k, 0) + 1

    failing = [r for r in out if r["n_failed"] >= 1]
    json.dump(out, open("drafts_all.json", "w", encoding="utf-8"), indent=2)
    json.dump(failing, open("drafts_failing.json", "w", encoding="utf-8"), indent=2)
    print(f"\nGenerated {len(out)} drafts; {len(failing)} FAIL >=1 instruction.")
    print("Failing drafts by generator/mode:", nfail_by)
    print(f"Est cost so far: ${oc.usage()['est_usd']:.4f}  (calls={oc.usage()['calls']})")


if __name__ == "__main__":
    main()
