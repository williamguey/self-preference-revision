"""Thin wrapper around the OFFICIAL Google IFEval checker.

Replicates the per-instruction loop from google-research
instruction_following_eval/evaluation_lib.py (strict eval), exposing a
function that, given a response + an item's instruction_id_list + kwargs,
returns a per-instruction list of True/False (pass/fail).

The verification logic itself is NOT reimplemented here: we call into the
downloaded instruction_following_eval.instructions_registry classes.
"""
from typing import List, Dict, Any
from instruction_following_eval import instructions_registry


def check_following_list(prompt: str,
                         response: str,
                         instruction_id_list: List[str],
                         kwargs_list: List[Dict[str, Any]]) -> List[bool]:
    """Return per-instruction pass/fail using the official IFEval registry.

    This mirrors evaluation_lib.test_instruction_following_strict.
    """
    assert len(instruction_id_list) == len(kwargs_list), \
        "instruction_id_list and kwargs_list length mismatch"
    is_following = []
    for index, instruction_id in enumerate(instruction_id_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Drop None-valued kwargs (HF dataset pads unused params with None).
        kwargs = {k: v for k, v in kwargs_list[index].items() if v is not None}
        instruction.build_description(**kwargs)

        # Some instructions need the original prompt (e.g. repeat-the-prompt).
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)

        if response.strip() and instruction.check_following(response):
            is_following.append(True)
        else:
            is_following.append(False)
    return is_following


def describe_instructions(prompt, instruction_id_list, kwargs_list):
    """Return the official natural-language description of each instruction."""
    descs = []
    for index, instruction_id in enumerate(instruction_id_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        kwargs = {k: v for k, v in kwargs_list[index].items() if v is not None}
        desc = instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            desc = instruction.build_description(prompt=prompt)
        descs.append(desc)
    return descs


if __name__ == "__main__":
    # Quick smoke test on one real IFEval example + a known-answer construction.
    from datasets import load_dataset
    print("Loading google/IFEval ...")
    ds = load_dataset("google/IFEval", split="train")
    print(f"Loaded {len(ds)} items. Columns: {ds.column_names}\n")

    ex = ds[0]
    print("=== EXAMPLE 0 ===")
    print("prompt:", ex["prompt"][:300], "...\n")
    print("instruction_id_list:", ex["instruction_id_list"])
    print("kwargs:", ex["kwargs"], "\n")

    # 1) Check an empty/garbage response (should mostly fail).
    bad = check_following_list(ex["prompt"], "no.", ex["instruction_id_list"], ex["kwargs"])
    print("garbage response 'no.' -> per-instruction pass:", bad)

    # 2) Check the prompt echoed back (often passes some constraints).
    echo = check_following_list(ex["prompt"], ex["prompt"], ex["instruction_id_list"], ex["kwargs"])
    print("echo-prompt response   -> per-instruction pass:", echo)
