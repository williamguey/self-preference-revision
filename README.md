# Self-Preference in Verifiable Instruction-Following Revision

Code, data, and paper for the preprint:

> **Self-Preference Is Absent in Verifiable Instruction-Following Revision:
> A Four-Model Test Under Genuine Authorship**
> William Guey, Pierrick Bougault (Dept. of Industrial Engineering, Tsinghua University)
> Preprint: [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) *(TODO: set after posting)*

## What this is

Do LLMs resist **machine-verified-correct** fixes to their **own** writing more than to
another model's? We test this on **IFEval** instruction-following revision, using IFEval's
**official deterministic checker** to decide ground truth (so no LLM-judge circularity), and
**genuine in-context authorship** (the model that actually wrote the draft decides on a fix,
inside its own conversation) rather than a told "you wrote this" label.

**Headline result (null):** across 4 mid-tier model families and 85 author-vs-fresh
comparisons, authors reject verified-good fixes to their own drafts at essentially the same
rate as fresh models — self-preference gap **−5.1 pp, 95% CI [−12.9, +2.7]** (includes 0).
A self-skepticism hint from a smaller pilot did **not** replicate. The one robust observation
is qualitative: when authors *do* reject a verified-good fix, **97.4% (38/39)** of reasons are
flaw-catching, not preference. Effects smaller than ~13 pp cannot be excluded at this N.

See [`FINDINGS.md`](FINDINGS.md) for the full v1/v2/v3 write-up and
[`paper/main.pdf`](paper/main.pdf) for the manuscript.

## Repository layout

```
paper/        main.tex, refs.bib, main.pdf      (the preprint)
scripts/      the full pipeline + vendored official IFEval checker
results/      committed outputs of our run (all raw decisions, reasons, checker labels)
data/         note on IFEval loading (data is fetched at runtime, not committed)
FINDINGS.md   consolidated v1+v2+v3 findings
decisions_log*.txt, BUILD_NOTES.txt   autonomous-run logs / build notes
```

## Setup

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
export OPENROUTER_API_KEY=your-key-here     # PowerShell: $env:OPENROUTER_API_KEY="your-key-here"
```

The OpenRouter API key is read **only** from the `OPENROUTER_API_KEY` environment variable
(`scripts/or_client.py`). No key is committed anywhere in this repo.

First, sanity-check the official checker (known-answer self-test, no API needed):

```bash
cd scripts
python selftest_checker.py
```

## Reproduce

Run from the `scripts/` directory (intermediate JSONs are written to the current directory;
the canonical outputs from our run are committed under `results/`). Fixed seeds: prompt
sampling `SEED=42`, bootstrap `seed=12345`.

```bash
cd scripts

# --- P3: the powered four-model headline test (~US$0.70) ---
python run_v3_prep.py          # drafts across 4 families x {normal,rushed}; one verified-GOOD fix each
python run_v3_prep_topup.py    # expand prompt pool to reach the N target
python run_v3_decide.py        # author (in-context) vs fresh decisions; gap, CI, per-family, verdict
#   -> SUMMARY_v3.txt, results_v3.json

# --- P2: genuine-authorship two-model test ---
python run_v2_prep.py
python run_v2_decide.py        # -> results_v2.json

# --- P1: told-label proxy (development path) ---
python sample_prompts.py
python phase1_drafts.py
python phase2_fixes.py
python phase3_decider.py
python phase4_analyze.py       # -> results.json
```

Total API cost for all three pilots was **~US$1.2** (P3 alone ~US$0.70), on mid-tier models.
Mid-tier is deliberate: frontier models nearly ceiling out IFEval and leave few fixable errors.

## Method in one paragraph

For each prompt a model writes a draft; the official IFEval checker confirms the draft fails
≥1 constraint; we generate targeted minimal edits against the failed constraint's official
description and keep **exactly one** edit the checker verifies as GOOD (fixes the failed
constraint, breaks no passing one). The draft+fix is then judged ACCEPT/REJECT by (X) the
**author** model inside its own generation conversation and (Z) a **fresh** different family
seeing the draft neutrally — 3 reps each at temperature 0.7, with one-line reasons captured.
The self-preference gap is the difference in rejection rates of these verified-good fixes.

## Licensing

Project code: **MIT** (see [`LICENSE`](LICENSE)). The vendored checker under
`scripts/instruction_following_eval/` is the google-research IFEval code, redistributed under
its original **Apache-2.0** license (headers retained).

## Citation

See [`CITATION.cff`](CITATION.cff). Underlying benchmark: IFEval (Zhou et al., 2023,
arXiv:2311.07911).
