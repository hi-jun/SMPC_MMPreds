#!/usr/bin/env python3
"""Check that the 2026-11-12 readability pass of docs/agents/open_items.md kept every sentence fragment.

The old file (git revision OLD_REV, default 4c3c108, the version before the pass) is cut into fragments
at ". ", "; " and line breaks. Fragments and the new file are normalised alike: a list marker ("- ",
"* ") or heading mark ("#") at the start of a line is removed, "**" and "|" are removed everywhere, and
whitespace runs (line breaks included) become one space. The new file is normalised as one string, so a
sentence now spread over bullets reads as it did. A fragment also loses its trailing punctuation
(.,;:!?). Every fragment of MIN_WORDS words or more must occur in the normalised new file, at least as
often as in the normalised old file (so a repeated sentence cannot be dropped unnoticed).

Fragments changed on purpose go in EXPECTED_CHANGES with the reason. A fragment laid out as a table
("table" in its reason) must also have all its numbers in the new file, in the same order and with no
other number in between. Any other missing fragment, or a failed number check, makes the script exit 1.

The new file is `git show NEW_REV:<path>` (default 38a1b46, the commit of the readability pass), not the
working tree: pinned on 2026-11-12 so that later content fixes, marked in the file with dated notes, are not
read as reformat changes and the check stays a record of the pass.

Usage, from the repository root: python3 docs/agents/_reformat/check_open_items.py [OLD_REV] [NEW_REV]
"""
import pathlib
import re
import subprocess
import sys

OLD_REV = sys.argv[1] if len(sys.argv) > 1 else "4c3c108"
NEW_REV = sys.argv[2] if len(sys.argv) > 2 else "38a1b46"
ROOT = pathlib.Path(__file__).resolve().parents[3]
PATH = "docs/agents/open_items.md"
MIN_WORDS = 4
TABLE = "laid out as a table (item 'Ghosts that enter beside the ego', recount by controller)"
EXPECTED_CHANGES = {  # normalised old fragment -> reason
    "contacts would not be, since the ego never reacted), delta_max % and floor breaches, critical / mild: "
    "Proposed 9.2 -> 10.9 / 5.9 -> 7.4, 9 -> 13 / 3 -> 7": TABLE,
    "ISA-MPC 24.1 -> 27.9 / 15.1 -> 19.2, 14 -> 22 / 17 -> 29": TABLE,
    "PC-ACC 28.3 -> 30.3 / 26.9 -> 30.4, 24 -> 27 / 36 -> 46": TABLE,
    "PC-ACC as published 39.7 -> 43.0 / 41.3 -> 43.9, 29 -> 36 / 58 -> 66": TABLE,
}

LINE_MARK = re.compile(r"^\s*(?:[-*]\s+|#+\s*)")
TRAILING_PUNCT = ".,;:!? "
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def normalise(text):
    lines = [LINE_MARK.sub("", line, count=1) for line in text.split("\n")]
    text = " ".join(lines).replace("**", "").replace("|", "")
    return " ".join(text.split())


def numbers_in_order(frag, numbers):
    want = NUMBER.findall(frag)
    return any(numbers[i:i + len(want)] == want for i in range(len(numbers) - len(want) + 1))


old = subprocess.run(["git", "-C", str(ROOT), "show", f"{OLD_REV}:{PATH}"],
                     check=True, capture_output=True, text=True).stdout
new = subprocess.run(["git", "-C", str(ROOT), "show", f"{NEW_REV}:{PATH}"],
                     check=True, capture_output=True, text=True).stdout
old_norm, new_norm = normalise(old), normalise(new)
new_numbers = NUMBER.findall(new_norm)

fragments = []
for line in old.split("\n"):
    for piece in re.split(r"\. |; ", line):
        frag = normalise(piece).rstrip(TRAILING_PUNCT)
        if len(frag.split()) >= MIN_WORDS:
            fragments.append(frag)

missing = [frag for frag in fragments if new_norm.count(frag) < old_norm.count(frag)]
bad = []
for name, text in ((f"old {PATH} at {OLD_REV}", old), (f"new {PATH} at {NEW_REV}", new)):
    longest = max(len(line) for line in text.split("\n"))
    print(f"{name}: {len(text.encode())} bytes, {len(text.splitlines())} lines, longest line {longest} chars")
print(f"fragments of >= {MIN_WORDS} words: {len(fragments)}; found {len(fragments) - len(missing)}, "
      f"missing {len(missing)}")
for frag in missing:
    reason = EXPECTED_CHANGES.get(frag)
    if reason and "table" in reason:
        ok = numbers_in_order(frag, new_numbers)
        reason += f"; numbers in order in the new file: {'yes' if ok else 'NO'}"
        if not ok:
            bad.append(frag)
    elif not reason:
        reason = "UNEXPECTED (not in EXPECTED_CHANGES)"
        bad.append(frag)
    print(f"  missing: {frag} -- {reason}")
for frag in EXPECTED_CHANGES:
    if frag not in missing:
        print(f"  note: expected change not missing (stale entry?): {frag[:100]}")
print("OK" if not bad else f"FAIL: {len(bad)} fragment(s) missing or misplaced")
sys.exit(1 if bad else 0)
