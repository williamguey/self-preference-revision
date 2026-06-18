"""Self-test the OFFICIAL IFEval checker with KNOWN-ANSWER constructions.

We build responses where we already know which constraints should pass/fail,
and confirm the official registry agrees. This proves the checker discriminates,
not just that it runs.
"""
from ifeval_checker import check_following_list
from data_loader import load_ifeval


def banner(s):
    print("\n" + "=" * 70 + f"\n{s}\n" + "=" * 70)


items = load_ifeval()
ex = items[0]
prompt = ex["prompt"]
ids = ex["instruction_id_list"]
kwargs = ex["kwargs"]

banner("REAL IFEVAL EXAMPLE (item 0)")
print("prompt:\n", prompt)
print("\ninstruction_id_list:", ids)
print("kwargs:", kwargs)
# Constraints here: no commas; >=3 markdown-highlighted sections; >=300 words.

# ---- Known-answer PASS case -------------------------------------------------
# A comma-free body of 300+ words with three *highlighted* sections.
sentence = "The count of Tripoli ruled a crusader state in the twelfth century "
body = (sentence * 60)  # ~ 60*11 = 660 words, no commas
pass_response = (
    "*First section* " + body + " *Second section* " + body +
    " *Third section* " + body
)
res_pass = check_following_list(prompt, pass_response, ids, kwargs)
print("\n[PASS construction] expect [True, True, True]")
print("   no_comma / >=3 highlights / >=300 words ->", res_pass)

# ---- Known-answer FAIL case -------------------------------------------------
# Has commas, only one highlight, far too short -> should fail ALL three.
fail_response = "*Only one highlight*, and this draft is short, with commas."
res_fail = check_following_list(prompt, fail_response, ids, kwargs)
print("\n[FAIL construction] expect [False, False, False]")
print("   has commas / 1 highlight / short ->", res_fail)

# ---- Targeted single-constraint flip ----------------------------------------
# Same as PASS but reintroduce a comma -> only no_comma should flip to False.
comma_response = pass_response.replace("twelfth century", "twelfth, century")
res_comma = check_following_list(prompt, comma_response, ids, kwargs)
print("\n[COMMA flip] expect [False, True, True] (only no_comma fails)")
print("   ->", res_comma)

banner("SELF-TEST VERDICT")
ok = (res_pass == [True, True, True]
      and res_fail == [False, False, False]
      and res_comma == [False, True, True])
print("All known-answer checks correct:", ok)
if not ok:
    raise SystemExit("CHECKER SELF-TEST FAILED")
print("Official IFEval checker is wired up and discriminates correctly.")
