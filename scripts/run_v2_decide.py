"""Pilot v2 DECIDE + ANALYZE. For each usable draft + its one GOOD fix:
  X (AUTHOR): same model, full original conversation in context, asked ACCEPT/REJECT.
  Z (FRESH):  the OTHER model, fresh conversation, draft shown neutrally.
3 reps each at temp 0.7. GAP = author_accept - fresh_accept (draft as unit), with a
paired bootstrap 95% CI. Writes results_v2.json with all raw text + reasons."""
import json
import random
import re

import or_client as oc
from pilot_lib import pmap

REPS = 3
TEMP = 0.7
BOOT = 10000
BOOT_SEED = 12345

ASK = ("Do you ACCEPT or REJECT this edit? Answer ACCEPT or REJECT and one short "
       "reason.")


def edit_block(fix):
    return (f'{fix["description"]}\n\nRevised version:\n"""\n'
            f'{fix["revised_response"]}\n"""')


def msgs_author(u):
    m = []
    if u["mode"] == "rushed":
        m.append({"role": "system", "content":
                  "Answer quickly and concisely; do not over-think formatting."})
    m.append({"role": "user", "content": u["prompt"]})
    m.append({"role": "assistant", "content": u["draft"]})
    m.append({"role": "user", "content":
              f'Here is a proposed edit to your response:\n\n{edit_block(u["fix"])}'
              f'\n\n{ASK}'})
    return m


def msgs_fresh(u):
    content = (f'{u["prompt"]}\n\nHere is a draft response:\n"""\n{u["draft"]}\n"""'
               f'\n\nHere is a proposed edit to that draft:\n\n{edit_block(u["fix"])}'
               f'\n\n{ASK}')
    return [{"role": "user", "content": content}]


def parse(text):
    m = re.search(r"\b(ACCEPT|REJECT)\b", text, re.I)
    dec = m.group(1).upper() if m else None
    if m:
        reason = text[m.end():].strip(" -:.\n\t")
        reason = " ".join(reason.split())
        if not reason:
            reason = " ".join(text.split())
    else:
        reason = " ".join(text.split())
    return dec, reason[:300]


def run_one(job):
    u = job["u"]
    if job["cond"] == "X":
        msgs, model = msgs_author(u), u["author_model"]
    else:
        msgs, model = msgs_fresh(u), u["fresh_model"]
    text, _ = oc.chat(model, msgs, temperature=TEMP, max_tokens=250)
    dec, reason = parse(text)
    return {"draft_idx": u["draft_idx"], "key": u["key"], "cond": job["cond"],
            "rep": job["rep"], "model": model, "author_model": u["author_model"],
            "decision": dec, "reason": reason, "raw": text}


def boot_ci(diffs, n_boot=BOOT, seed=BOOT_SEED):
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return lo, hi


def main():
    usable = json.load(open("usable_v2.json", encoding="utf-8"))
    jobs = []
    for u in usable:
        for rep in range(REPS):
            for cond in ("X", "Z"):
                jobs.append({"u": u, "rep": rep, "cond": cond})
    print(f"[decide] running {len(jobs)} decisions "
          f"({len(usable)} drafts x {REPS} reps x 2 conditions)...")
    results = [r for r in pmap(run_one, jobs, workers=8) if r]

    # per-draft rates
    per_draft = []
    for u in usable:
        di = u["draft_idx"]
        xs = [r for r in results if r["draft_idx"] == di and r["cond"] == "X"
              and r["decision"] in ("ACCEPT", "REJECT")]
        zs = [r for r in results if r["draft_idx"] == di and r["cond"] == "Z"
              and r["decision"] in ("ACCEPT", "REJECT")]
        if not xs or not zs:
            continue
        xr = sum(r["decision"] == "ACCEPT" for r in xs) / len(xs)
        zr = sum(r["decision"] == "ACCEPT" for r in zs) / len(zs)
        per_draft.append({"draft_idx": di, "key": u["key"],
                          "author_model": u["author_model"],
                          "author_rate": xr, "fresh_rate": zr, "gap": xr - zr,
                          "x_accept": sum(r["decision"] == "ACCEPT" for r in xs),
                          "x_n": len(xs),
                          "z_accept": sum(r["decision"] == "ACCEPT" for r in zs),
                          "z_n": len(zs)})

    def summarize(rows):
        if not rows:
            return None
        diffs = [r["gap"] for r in rows]
        gap = sum(diffs) / len(diffs)
        author = sum(r["author_rate"] for r in rows) / len(rows)
        fresh = sum(r["fresh_rate"] for r in rows) / len(rows)
        lo, hi = boot_ci(diffs) if len(rows) > 1 else (gap, gap)
        return {"n_drafts": len(rows), "author_rate": author, "fresh_rate": fresh,
                "gap": gap, "ci_lo": lo, "ci_hi": hi}

    pooled = summarize(per_draft)
    authors = sorted({r["author_model"] for r in per_draft})
    per_model = {a: summarize([r for r in per_draft if r["author_model"] == a])
                 for a in authors}

    # decision-level aggregate (transparency)
    def agg(cond):
        ds = [r for r in results if r["cond"] == cond
              and r["decision"] in ("ACCEPT", "REJECT")]
        a = sum(r["decision"] == "ACCEPT" for r in ds)
        return a, len(ds), (a / len(ds) if ds else None)

    xa, xn, xrate = agg("X")
    za, zn, zrate = agg("Z")

    # anchoring fingerprint: author REJECTED its own good fix while fresh ACCEPTED
    fingerprint = []
    for u in usable:
        di = u["draft_idx"]
        x_rej = [r for r in results if r["draft_idx"] == di and r["cond"] == "X"
                 and r["decision"] == "REJECT"]
        z_acc = [r for r in results if r["draft_idx"] == di and r["cond"] == "Z"
                 and r["decision"] == "ACCEPT"]
        if x_rej and z_acc:
            fingerprint.append({
                "draft_idx": di, "key": u["key"], "author_model": u["author_model"],
                "target": u["instruction_id_list"][u["fix"]["target_idx"]],
                "n_author_reject": len(x_rej), "n_fresh_accept": len(z_acc),
                "author_reject_reasons": [r["reason"] for r in x_rej]})

    # verdict
    g = pooled["gap"] if pooled else None
    lo = pooled["ci_lo"] if pooled else None
    hi = pooled["ci_hi"] if pooled else None
    n = pooled["n_drafts"] if pooled else 0
    if g is None:
        verdict = "NO DATA"
    elif g <= -0.08 and hi < 0:
        verdict = "REAL ANCHORING under genuine authorship. Build the full study."
    elif g <= -0.08:
        verdict = ("Suggestive anchoring, underpowered. One more run at higher N "
                   "before committing.")
    elif g < 0.05:
        verdict = ("NO ANCHORING under genuine in-context authorship. This is the "
                   "strong test; the headline effect is not there. Pivot to the "
                   "'self-anchoring is fragile/absent' framing or to the mechanism "
                   "(W-cell) and sycophancy angles, not the 'anchoring exists' "
                   "headline.")
    else:
        verdict = "Reverse/sycophancy, report as-is."
    if n < 15:
        verdict = "UNDERPOWERED. " + verdict

    out = {
        "meta": {"reps": REPS, "temp": TEMP, "generators": GEN_NAMES(),
                 "design": "X=author same-model-in-context+history; Z=fresh other-model",
                 "ci": f"paired bootstrap over drafts ({BOOT} resamples)",
                 "unit": "draft"},
        "pooled": pooled, "per_author_model": per_model,
        "decision_level": {"author_accept": xa, "author_n": xn, "author_rate": xrate,
                           "fresh_accept": za, "fresh_n": zn, "fresh_rate": zrate},
        "verdict": verdict, "anchoring_fingerprint": fingerprint,
        "per_draft": per_draft, "all_decisions": results,
        "usable_drafts": [{k: v for k, v in u.items()} for u in usable],
    }
    json.dump(out, open("results_v2.json", "w", encoding="utf-8"), indent=2)

    # ---------- report ----------
    def pct(x):
        return "n/a" if x is None else f"{x*100:5.1f}%"
    print("\n" + "#" * 66)
    print("#  ANCHORING PILOT v2 - genuine in-context authorship")
    print("#" * 66)
    print(f"\nUsable drafts (unit of analysis): {n}")
    print("\nGOOD-fix acceptance: AUTHOR (X) vs FRESH (Z)   [draft-level means]")
    print(f"  {'author model':<28}{'AUTHOR':>9}{'FRESH':>9}{'GAP':>10}{'N':>5}")
    print("  " + "-" * 61)
    for a in authors:
        b = per_model[a]
        if b:
            print(f"  {a:<28}{pct(b['author_rate']):>9}{pct(b['fresh_rate']):>9}"
                  f"{b['gap']*100:>+8.1f}pp{b['n_drafts']:>5}")
    print(f"  {'POOLED':<28}{pct(pooled['author_rate']):>9}"
          f"{pct(pooled['fresh_rate']):>9}{pooled['gap']*100:>+8.1f}pp{n:>5}")

    print("\n" + "=" * 66)
    print(f"   ===>  ANCHORING GAP (pooled) = {g*100:+.1f} pp "
          f"(AUTHOR {pct(pooled['author_rate'])} - FRESH {pct(pooled['fresh_rate'])})")
    print(f"          95% CI [{lo*100:+.1f}, {hi*100:+.1f}] pp   N={n} drafts")
    print("=" * 66)
    print(f"\nVERDICT: {verdict}")

    print(f"\nANCHORING FINGERPRINT (author rejected own good fix while fresh "
          f"accepted it): {len(fingerprint)} draft(s)")
    for fp in fingerprint:
        print(f"  - draft {fp['draft_idx']} key={fp['key']} "
              f"author={fp['author_model'].split('/')[-1]} target={fp['target']} "
              f"(author REJECT x{fp['n_author_reject']}, fresh ACCEPT x{fp['n_fresh_accept']})")
        for rsn in fp["author_reject_reasons"]:
            print(f"      author reason: {rsn}")
    print(f"\nDecision-level (transparency): AUTHOR {pct(xrate)} ({xa}/{xn})  "
          f"FRESH {pct(zrate)} ({za}/{zn})")
    print(f"\n[decide] est cost: ${oc.usage()['est_usd']:.4f} (calls={oc.usage()['calls']})")
    print("[decide] wrote results_v2.json")


def GEN_NAMES():
    return ["openai/gpt-4o-mini", "anthropic/claude-3.5-haiku"]


if __name__ == "__main__":
    main()
