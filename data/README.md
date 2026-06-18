# Data

This project uses the **real IFEval** dataset (`google/IFEval` on HuggingFace,
541 prompts with verifiable instructions) and IFEval's **official deterministic
checker**.

## How the dataset is loaded

`scripts/data_loader.py` downloads `ifeval_input_data.jsonl` directly from the
HuggingFace Hub (`huggingface_hub.hf_hub_download`) and reads it line by line.

We do **not** use the `datasets` library: in our environment `datasets==2.12.0`
is incompatible with the installed `fsspec==2026.2.0` and raises
`ValueError: Invalid pattern: '**' can only be an entire path component` when
resolving the dataset. Reading the published jsonl directly avoids this and uses
the identical underlying data.

The downloaded `ifeval_input_data.jsonl` is **git-ignored** (re-fetched on first
run). No dataset files are committed.

## The checker

`scripts/instruction_following_eval/` is the official `instruction_following_eval`
module from google-research (Apache-2.0, original headers retained). We call its
`instructions_registry.INSTRUCTION_DICT` through the strict-evaluation loop. All
pass/fail and GOOD/BAD/NEUTRAL fix labels in the results come from this checker,
never from a model. Run `python scripts/selftest_checker.py` for a known-answer
self-test.
