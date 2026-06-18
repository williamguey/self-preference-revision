"""v3 DECIDE + ANALYZE (powered, 4 families). For each usable draft + its one GOOD
fix: author (X, same model in its own conversation) and fresh (Z, a different
family, neutral) decide ACCEPT/REJECT, 3 reps each at temp 0.7.

SKEPTICISM GAP = author_reject_rate - fresh_reject_rate (POSITIVE = author harsher
on its OWN work = the self-skepticism reversal). Paired bootstrap 95% CI over
drafts; per-author-family breakdown; skeptic vs mirror paired counts; and a
classification of every author-rejection reason into STRICTER-than-checker vs
PREFERENCE. Writes results_v3.json + SUMMARY_v3.txt. Cost-guarded."""
import json
import random
import re

import or_client as oc
from pilot_lib import pmap

FAMILIES = ["openai/gpt-4o-mini", "anthropic/claude-3.5-haiku",
            "google/gemini-2.5-flash-lite", "meta-llama/llama-3.3-70b-instruct"]
CLASSIFIER = "openai/gpt-4o-mini"
REPS = 3
TEMP = 0.7
BOOT = 10000
BOOT_SEED = 12345
COST_STOP = 11.0
WORKERS = 12

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
    if text is None:
        return None, ""
    m = re.search(r"\b(ACCEPT|REJECT)\b", text, re.I)
    dec = m.group(1).upper() if m else None
    if m:
        reason = " ".join(text[m.end():].strip(" -:.\n\t").split()) or " ".join(text.split())
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
            "fresh_model": u["fresh_model"], "decision": dec, "reason": reason,
            "raw": text}


CLS = """An AI assistant rejected a proposed edit to a draft it had written. Its \
stated reason was:

"{reason}"

Classify the reason as exactly one word:
STRICTER - it objects that the edit fails to fully satisfy a requirement, or \
breaks/violates a format, rule, or constraint (i.e. it is catching a flaw).
PREFERENCE - it merely prefers the original wording or style, with no rule-based \
objection.
OTHER - neither.
Answer with ONE word: STRICTER, PREFERENCE, or OTHER."""


def classify_one(reason):
    text, _ = oc.chat(CLASSIFIER, [{"role": "user", "content": CLS.format(reason=reason)}],
                      temperature=0, max_tokens=5)
    t = text.upper()
    for lab in ("STRICTER", "PREFERENCE", "OTHER"):
        if lab in t:
            return lab
    return "OTHER"


def boot_ci(diffs, seed=BOOT_SEED, n_boot=BOOT):
    if len(diffs) < 2:
        return (diffs[0], diffs[0]) if diffs else (0, 0)
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def main():
    oc.set_pricing(FAMILIES)
    usable = json.load(open("usable_v3.json", encoding="utf-8"))
    jobs = [{"u": u, "rep": rep, "cond": cond}
            for u in usable for rep in range(REPS) for cond in ("X", "Z")]
    print(f"[v3 decide] {len(jobs)} decisions ({len(usable)} drafts x{REPS}x2)...")

    results, stopped = [], False
    for s in range(0, len(jobs), 120):
        chunk = jobs[s:s + 120]
        results += [r for r in pmap(run_one, chunk, workers=WORKERS) if r]
        c = oc.usage()["est_usd"]
        print(f"  ...{len(results)}/{len(jobs)} done, cost ${c:.3f}")
        if c > COST_STOP:
            stopped = True
            print("  COST CAP reached - stopping, saving partial.")
            break

    # per-draft reject rates (skepticism framing)
    per_draft = []
    for u in usable:
        di = u["draft_idx"]
        xs = [r for r in results if r["draft_idx"] == di and r["cond"] == "X"
              and r["decision"] in ("ACCEPT", "REJECT")]
        zs = [r for r in results if r["draft_idx"] == di and r["cond"] == "Z"
              and r["decision"] in ("ACCEPT", "REJECT")]
        if not xs or not zs:
            continue
        xr = sum(r["decision"] == "REJECT" for r in xs) / len(xs)
        zr = sum(r["decision"] == "REJECT" for r in zs) / len(zs)
        per_draft.append({"draft_idx": di, "key": u["key"],
                          "author_model": u["author_model"], "fresh_model": u["fresh_model"],
                          "author_reject": xr, "fresh_reject": zr, "gap": xr - zr,
                          "x_n": len(xs), "z_n": len(zs)})

    def summarize(rows):
        if not rows:
            return None
        diffs = [r["gap"] for r in rows]
        lo, hi = boot_ci(diffs)
        return {"n": len(rows),
                "author_reject": sum(r["author_reject"] for r in rows) / len(rows),
                "fresh_reject": sum(r["fresh_reject"] for r in rows) / len(rows),
                "gap": sum(diffs) / len(diffs), "ci_lo": lo, "ci_hi": hi}

    pooled = summarize(per_draft)
    per_model = {a: summarize([r for r in per_draft if r["author_model"] == a])
                 for a in FAMILIES}
    per_model = {a: v for a, v in per_model.items() if v}

    # paired skeptic vs mirror (any-rep and majority)
    def counts(defn):
        sk = mr = 0
        for u in usable:
            di = u["draft_idx"]
            xs = [r for r in results if r["draft_idx"] == di and r["cond"] == "X"]
            zs = [r for r in results if r["draft_idx"] == di and r["cond"] == "Z"]
            xrej = [r for r in xs if r["decision"] == "REJECT"]
            xacc = [r for r in xs if r["decision"] == "ACCEPT"]
            zrej = [r for r in zs if r["decision"] == "REJECT"]
            zacc = [r for r in zs if r["decision"] == "ACCEPT"]
            if defn == "any":
                if xrej and zacc:
                    sk += 1
                if zrej and xacc:
                    mr += 1
            else:  # majority
                xr = len(xrej) / max(1, len(xrej) + len(xacc))
                zr = len(zrej) / max(1, len(zrej) + len(zacc))
                if xr > 0.5 and zr < 0.5:
                    sk += 1
                if zr > 0.5 and xr < 0.5:
                    mr += 1
        return sk, mr

    sk_any, mr_any = counts("any")
    sk_maj, mr_maj = counts("maj")

    # classify author-rejection reasons (stricter-than-checker vs preference)
    author_rejects = [r for r in results if r["cond"] == "X" and r["decision"] == "REJECT"]
    cls_labels = pmap(classify_one, [r["reason"] for r in author_rejects],
                      workers=WORKERS) if author_rejects else []
    for r, lab in zip(author_rejects, cls_labels):
        r["reason_class"] = lab or "OTHER"
    n_strict = sum(1 for r in author_rejects if r.get("reason_class") == "STRICTER")
    n_pref = sum(1 for r in author_rejects if r.get("reason_class") == "PREFERENCE")
    n_other = sum(1 for r in author_rejects if r.get("reason_class") == "OTHER")
    strict_frac = n_strict / len(author_rejects) if author_rejects else None

    # verdict (binding)
    g = pooled["gap"]; lo = pooled["ci_lo"]; hi = pooled["ci_hi"]; n = pooled["n"]
    ci_excl0 = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
    same_sign = sum(1 for v in per_model.values()
                    if (v["gap"] > 0) == (g > 0) and abs(v["gap"]) > 1e-9)
    majority = same_sign >= (len(per_model) + 1) // 2 + (1 if len(per_model) % 2 == 0 else 0)
    majority = same_sign > len(per_model) / 2
    if g >= 0.10 and ci_excl0 and majority:
        verdict = ("REAL, cross-model self-skepticism reversal. Novel, against-"
                   "consensus. Write the paper around this.")
    elif g >= 0.05 and ci_excl0:
        verdict = ("Model-dependent self-skepticism. Real but partial - frame as "
                   "heterogeneous.")
    elif g >= 0.05:
        verdict = "Directional, still underpowered. Bump N to ~300 before claiming."
    elif g > -0.05:
        verdict = ("No reversal. Both self-anchoring AND self-skepticism absent in "
                   "this regime. Clean double-null; write up and STOP experimenting.")
    elif ci_excl0:
        verdict = "Self-anchoring at higher N - reverses v2; report carefully."
    else:
        verdict = ("Negative-leaning but CI includes 0 - inconclusive/underpowered "
                   "self-anchoring direction.")
    if n < 15:
        verdict = "UNDERPOWERED. " + verdict
    if stopped:
        verdict = "[PARTIAL - cost cap hit] " + verdict

    out = {
        "meta": {"families": FAMILIES, "n_usable": len(usable), "n_analyzed": n,
                 "reps": REPS, "temp": TEMP, "unit": "draft",
                 "ci": f"paired bootstrap over drafts ({BOOT})",
                 "measure": "skepticism gap = author_reject_rate - fresh_reject_rate",
                 "stopped_on_cost": stopped},
        "pooled": pooled, "per_author_model": per_model,
        "skeptic_vs_mirror": {"any_rep": {"skeptic": sk_any, "mirror": mr_any},
                              "majority": {"skeptic": sk_maj, "mirror": mr_maj}},
        "author_reject_reason_classes": {"STRICTER": n_strict, "PREFERENCE": n_pref,
                                         "OTHER": n_other, "stricter_fraction": strict_frac},
        "verdict": verdict, "per_draft": per_draft,
        "author_rejections": [{"draft_idx": r["draft_idx"], "key": r["key"],
                               "author_model": r["author_model"], "rep": r["rep"],
                               "reason": r["reason"], "class": r.get("reason_class")}
                              for r in author_rejects],
        "all_decisions": results,
        "usable_drafts": usable,
    }
    json.dump(out, open("results_v3.json", "w", encoding="utf-8"), indent=2)

    # ---------- SUMMARY_v3.txt (one screen) ----------
    def pct(x):
        return "n/a" if x is None else f"{x*100:.1f}%"
    L = []
    L.append("=" * 70)
    L.append("ANCHORING PILOT v3 - SELF-SKEPTICISM REVERSAL (powered, 4 families)")
    L.append("=" * 70)
    L.append(f"Usable drafts analyzed: {n}   Families: "
             + ", ".join(f.split('/')[-1] for f in FAMILIES))
    L.append("")
    L.append("SKEPTICISM GAP = author_reject_rate - fresh_reject_rate "
             "(positive = author harsher on own work)")
    L.append(f"  POOLED: author_reject {pct(pooled['author_reject'])}  "
             f"fresh_reject {pct(pooled['fresh_reject'])}  "
             f"GAP {g*100:+.1f}pp  95% CI [{lo*100:+.1f}, {hi*100:+.1f}]pp")
    L.append("")
    L.append("  per AUTHOR family:")
    L.append(f"    {'family':<24}{'auth_rej':>9}{'fresh_rej':>10}{'GAP':>9}{'N':>5}")
    for a, v in per_model.items():
        L.append(f"    {a.split('/')[-1]:<24}{pct(v['author_reject']):>9}"
                 f"{pct(v['fresh_reject']):>10}{v['gap']*100:>+7.1f}pp{v['n']:>5}")
    L.append(f"  same-sign-as-pooled families: {same_sign}/{len(per_model)} "
             f"(majority={'yes' if majority else 'no'})")
    L.append("")
    L.append(f"PAIRED (any-rep):  skeptic {sk_any}  vs  mirror {mr_any}    "
             f"(majority-vote: skeptic {sk_maj} vs mirror {mr_maj})")
    L.append("")
    L.append(f"AUTHOR-REJECTION REASONS (n={len(author_rejects)}): "
             f"STRICTER-than-checker {n_strict} ({pct(strict_frac)}), "
             f"PREFERENCE {n_pref}, OTHER {n_other}")
    examples = [r for r in author_rejects if r.get("reason_class") == "STRICTER"][:4]
    if len(examples) < 4:
        examples += [r for r in author_rejects if r not in examples][:4 - len(examples)]
    for r in examples[:4]:
        L.append(f"  [{r.get('reason_class','?')}] {r['author_model'].split('/')[-1]} "
                 f"k{r['key']}: {r['reason'][:130]}")
    L.append("")
    L.append("VERDICT: " + verdict)
    L.append("")
    L.append(f"(decision-level reject: AUTHOR "
             f"{pct(sum(rr['decision']=='REJECT' for rr in results if rr['cond']=='X' and rr['decision'])/max(1,sum(1 for rr in results if rr['cond']=='X' and rr['decision'] in ('ACCEPT','REJECT'))))} | "
             f"FRESH {pct(sum(rr['decision']=='REJECT' for rr in results if rr['cond']=='Z' and rr['decision'])/max(1,sum(1 for rr in results if rr['cond']=='Z' and rr['decision'] in ('ACCEPT','REJECT'))))})")
    L.append(f"est cost ${oc.usage()['est_usd']:.3f}  |  results_v3.json saved"
             + ("  |  PARTIAL (cost cap)" if stopped else ""))
    summary = "\n".join(L)
    open("SUMMARY_v3.txt", "w", encoding="utf-8").write(summary + "\n")
    print("\n" + summary)


if __name__ == "__main__":
    main()
