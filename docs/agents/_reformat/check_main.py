#!/usr/bin/env python3
"""Check that the 2026-11-12 readability pass of AGENTS.md kept every sentence fragment.

The old AGENTS.md (git revision OLD_REV, default 4c3c108, the split version before the pass) is cut
into fragments at ". ", "; " and line breaks. Fragments and the new AGENTS.md are normalised alike:
a list marker ("- ", "* ") or heading mark ("#") at the start of a line is removed, "**" and "|" are
removed everywhere, and whitespace runs (line breaks included) become one space. The new text is
normalised as one string, so a sentence now spread over bullets reads as it did. A fragment also
loses its trailing punctuation (.,;:!?), so it matches whatever now follows it (a semicolon that
became a bullet break). Every fragment of MIN_WORDS words or more must occur in the normalised new
text. Fragments changed on purpose go in EXPECTED_CHANGES with the reason; any other missing
fragment makes the script exit 1.

Usage, from the repository root: python3 docs/agents/_reformat/check_main.py [OLD_REV]
"""
import pathlib
import re
import subprocess
import sys

OLD_REV = sys.argv[1] if len(sys.argv) > 1 else "4c3c108"
ROOT = pathlib.Path(__file__).resolve().parents[3]
MIN_WORDS = 4
EXPECTED_CHANGES = {}  # normalised old fragment -> reason

LINE_MARK = re.compile(r"^\s*(?:[-*]\s+|#+\s*)")
TRAILING_PUNCT = ".,;:!? "


def normalise(text):
    lines = [LINE_MARK.sub("", line, count=1) for line in text.split("\n")]
    text = " ".join(lines).replace("**", "").replace("|", "")
    return " ".join(text.split())


old = subprocess.run(["git", "-C", str(ROOT), "show", f"{OLD_REV}:AGENTS.md"],
                     check=True, capture_output=True, text=True).stdout
new = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
new_norm = normalise(new)

fragments = []
for line in old.split("\n"):
    for piece in re.split(r"\. |; ", line):
        frag = normalise(piece).rstrip(TRAILING_PUNCT)
        if len(frag.split()) >= MIN_WORDS:
            fragments.append(frag)

missing = [frag for frag in fragments if frag not in new_norm]
unexpected = [frag for frag in missing if frag not in EXPECTED_CHANGES]

for name, text in ((f"old AGENTS.md at {OLD_REV}", old), ("new AGENTS.md", new)):
    longest = max(len(line) for line in text.split("\n"))
    print(f"{name}: {len(text.encode())} bytes, longest line {longest} chars")
print(f"fragments of >= {MIN_WORDS} words: {len(fragments)}; found {len(fragments) - len(missing)}, "
      f"missing {len(missing)}")
for frag in missing:
    reason = EXPECTED_CHANGES.get(frag, "UNEXPECTED (not in EXPECTED_CHANGES)")
    print(f"  missing: {frag[:120]} -- {reason}")
print("OK" if not unexpected else f"FAIL: {len(unexpected)} fragment(s) missing")
sys.exit(1 if unexpected else 0)
