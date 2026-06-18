# LLM Self-Anchoring on IFEval, Pilot Findings (v1 + v2)

**Question.** Does an LLM reject *valid* fixes to its OWN writing more than identical
fixes to ANOTHER model's writing? "Anchoring gap" = acceptance(own) − acceptance(other).
Meaningfully negative ⇒ self-anchoring.

**Substrate / tooling (constant across both pilots).**
- Data: real `google/IFEval` (541 items), loaded from the HF jsonl (the env's
  `datasets` 2.12.0 is broken against `fsspec` 2026.2.0).
- Verifier: the **official** google-research `instruction_following_eval`
  registry (`instructions_registry.INSTRUCTION_DICT`), called via the strict-eval
  loop. Known-answer self-tested (pass/fail/single-constraint-flip all correct).
- Models (OpenRouter): `openai/gpt-4o-mini`, `anthropic/claude-3.5-haiku` (mid-tier
  on purpose, frontier models ceiling out on IFEval).
- Fix labels (GOOD/BAD/NEUTRAL) are assigned by the CHECKER, never a model:
  GOOD = flips a failed constraint to pass, breaks none; BAD = breaks a passing
  constraint; NEUTRAL = no change in outcomes.

---

## Pilot v1, TOLD label ("a draft YOU wrote" vs "ANOTHER AI model")

Same decider judges the same draft+fixes under two byte-identical framings differing
in one sentence. 11 kept drafts, full-revised-text shown, 3 reps × 2 deciders.

| | OWN | OTHER | GAP |
|---|---|---|---|
| pooled (GOOD fixes) | 44.3% | 43.5% | **+0.7pp** (95% CI ±9.5pp) |

- **Flat / inconclusive.** N_good = 35; N_bad = **0** natural ⇒ specificity check
  skipped (not fabricated). IFEval's editable constraints are too *independent* to
  produce natural cross-constraint BAD fixes.
- **Confound discovered:** ~half of all GOOD-fix rejections (both framings) were
  "redundant / duplicate / no change", an artifact of showing 4 near-duplicate
  fixes per draft. This deflated the baseline (~44%) and widened the CI.
- **Qualitative:** OWN and OTHER rejected at near-identical rates *and for the same
  reasons*. No ownership-specific rationalization.

---

## Pilot v2, REAL in-context authorship (the strong test)

Condition **X (author):** the same model decides ACCEPT/REJECT inside its own
conversation (system+prompt+its draft as the assistant turn+the edit question).
Condition **Z (fresh):** the OTHER model, fresh chat, draft shown neutrally.
**Exactly one** checker-verified GOOD fix per draft (kills the redundancy confound).
27 usable drafts (15 gpt-authored, 12 haiku-authored), 3 reps each, draft = unit of
analysis, paired bootstrap CI.

| author model | AUTHOR (X) | FRESH (Z) | gap | N |
|---|---|---|---|---|
| claude-3.5-haiku | 80.6% | 80.6% | +0.0pp | 12 |
| gpt-4o-mini | 86.7% | 86.7% | +0.0pp | 15 |
| **pooled** | **84.0%** | **84.0%** | **+0.0pp** | 27 |

**GAP = +0.0pp, 95% CI [−13.6, +13.6]pp, N=27.**

- **Decisive null.** Per-draft gap signs are symmetric: 4 anchoring-direction,
  4 opposite, 19 tied. Not even a directional lean under the mean.
- Acceptance jumped 44% → **84%** vs v1, confirming the redundancy confound (not
  authorship) drove v1's low/noisy baseline.
- **Fingerprint** (author rejects own good fix while fresh accepts): 6 drafts;
  mirror (fresh rejects, author accepts): 4 drafts, balanced. The author's reasons
  are mostly *stricter-than-checker* objections (e.g. "the edit uses lowercase but
  the task needs all caps"; "nested triple quotes break the double-quote rule"),
  i.e. **self-skepticism, not self-defense.** Only 2 mild self-preference reasons,
  each 1/3 reps, offset by the mirror cases.

---

## Verdict

**No self-anchoring at a detectable magnitude on this substrate.** Both the cheap
told-label proxy (+0.7 ± 9.5pp) and the genuine-authorship strong test
(+0.0, CI [−13.6, +13.6]) point the same way. An effect smaller than ~14pp can't be
excluded at this N, but the point estimate is exactly 0 with a symmetric draft-level
distribution.

Secondary, paper-worthy observation: under real authorship these mid-tier models
sometimes critique their *own* output **more** harshly than a fresh model does
(catching verifier blind spots), closer to self-skepticism than self-anchoring.

## Recommended pivots (from the verdict text)
1. **Reframe** the headline from "self-anchoring exists" to "self-anchoring is
   fragile/absent in mid-tier instruction-following", a clean null with a strong
   test behind it.
2. **Mechanism / sycophancy angle** instead of the anchoring headline (the W-cell
   and sycophancy directions).
3. If still pursuing a magnitude estimate: higher N and **frontier deciders judging
   mid-tier drafts**, and a substrate with **coupled constraints** (IFBench, or
   prompts where fixing A breaks B) so a real BAD/specificity bucket exists.

---

## Pilot v3, powered self-skepticism reversal, 4 families (the test of the v2 hint)

The v2 leftover hint (author harsher on its own work: 6 vs 4 on 27 drafts) was
underpowered and used 2 models. v3 scales to **85 usable drafts** across **4 mid-tier
families** (gpt-4o-mini, claude-3.5-haiku, gemini-2.5-flash-lite, llama-3.3-70b;
`gemini-flash-1.5` 404'd → flash-lite). Same genuine-authorship rig. Measure =
**skepticism gap = author_reject_rate − fresh_reject_rate** (positive = author
harsher on own work). Fresh decider = a *different* family, rotated.

**Pooled gap = −5.1pp** (author_reject 15.3% vs fresh_reject 20.4%), **95% CI
[−12.9, +2.7]pp**, includes 0. The hint **did not replicate**; the point estimate
even flipped slightly negative (authors a touch *more* accepting of fixes to their
own drafts than fresh models).

Per author family (heterogeneous, no majority):

| author family | author_rej | fresh_rej | gap | N |
|---|---|---|---|---|
| gpt-4o-mini | 13.7% | 8.8% | **+4.9pp** | 34 |
| gemini-2.5-flash-lite | 17.5% | 15.8% | **+1.8pp** | 19 |
| claude-3.5-haiku | 15.8% | 29.8% | **−14.0pp** | 19 |
| llama-3.3-70b | 15.4% | 43.6% | **−28.2pp** | 13 |

Paired (any-rep): skeptic 9 vs mirror 10 (majority-vote: 4 vs 10). Balanced→mirror-leaning.

**Key confound:** condition Z's decider is a *different family* with its own baseline
strictness, so per-family gaps conflate authorship with "which fresh comparator this
author happened to face." The big negatives (haiku, llama) largely reflect strict
fresh comparators, not a clean authorship effect. The design (spec-mandated) can't
fully deconfound this.

**The one robust, clean finding (qualitative):** of 39 author self-rejections,
**97.4% (38/39) were STRICTER-than-checker** (catching a real flaw the verifier/fresh
missed, e.g. "the edit includes Sarah but breaks the limerick's AABBA scheme"), only
1 mere preference. So *when* an author rejects a fix to its own work, it is almost
never "I like mine better", it is flaw-catching. The mechanism is real; the net
*rate* just isn't higher than a fresh model's.

**Verdict (v3):** point estimate −5.1pp sits right at the −0.05 boundary with a CI
that spans 0 and no cross-model majority → operationally the pre-committed
**double-null / no-reversal** outcome. The self-skepticism reversal does **not** hold
at scale. Durable takeaway = the *flaw-catching character* of self-rejections (97%
stricter-than-checker), not a higher self-rejection rate.

## Files
`results_v3.json`, `SUMMARY_v3.txt`, `checkpoint_v3.txt`, `decisions_log_v3.txt`,
`usable_v3.json` (v3);
`results_v2.json` (all raw author/fresh text + reasons + labels + analysis),
`results.json` (v1), `checkpoint_v2.txt`, `decisions_log.txt`, `usable_v2.json`,
and the pipeline scripts (`sample_prompts.py`, `run_v2_prep.py`, `run_v2_decide.py`,
`ifeval_checker.py`, `or_client.py`).

