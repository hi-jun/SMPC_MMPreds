#!/usr/bin/env python3
"""Check that the reformatted part 1 of tokens.md keeps every sentence of part 1 of tokens.md.

The new text is docs/agents/tokens.md at NEW_REV (default 38a1b46, the commit of the readability pass; the
reformatted part 1 was first written as docs/agents/_reformat/tokens_part1.md and merged into tokens.md the same
day), read with `git show`, not from the working tree: pinned on 2026-11-12 so that later content fixes, marked in
the file with dated notes, are not read as reformat changes and the check stays a record of the pass.

Part 1 = docs/agents/tokens.md at OLD_REV (default 4c3c108), from the start of the file up to, not
including, the heading "### `safebranch`".

Check 1 (as specified): split the old text into fragments at ". ", "; " and line breaks; normalise
whitespace, markdown markers (leading "- ", "* ", "#"; "**"; "|") and trailing punctuation; every
fragment of 4 words or more must occur in the new text, normalised the same way with its lines
joined by spaces. A missing fragment is explained when it is a table row's first fragment, which
glues the row's first cell (the token name, now only the section heading) to the first sentence.

Check 2 (stricter): the old text without its table scaffolding (header row, separator row, first
cell) and the new text, as word sequences under the same normalisation with a trailing "." or ";"
dropped from each word, must differ only by inserted bold labels of at most 5 words.

Usage, from the repository root: python3 docs/agents/_reformat/check_part1.py [OLD_REV] [NEW_REV]
"""
import difflib
import pathlib
import re
import subprocess
import sys

OLD_REV = sys.argv[1] if len(sys.argv) > 1 else "4c3c108"
NEW_REV = sys.argv[2] if len(sys.argv) > 2 else "38a1b46"
ROOT = pathlib.Path(__file__).resolve().parents[3]
NEW_PATH = "docs/agents/tokens.md"   # the joined file (parts merged 2026-11-12), read at NEW_REV
CUT = "### `safebranch`"
ROW = re.compile(r"^\| (.+?) \| (.*) \|$")
SCAFFOLD = {"| Token | Effect |", "|---|---|"}


def norm_line(line):
    s = re.sub(r"^#+\s*", "", line.strip())
    s = re.sub(r"^[-*]\s+", "", s)
    return s.replace("**", "").replace("|", " ")


def norm(text):
    return re.sub(r"\s+", " ", " ".join(norm_line(x) for x in text.split("\n"))).strip()


def words(text):
    return [w.rstrip(".;") for w in norm(text).split() if w.rstrip(".;")]


full = subprocess.run(["git", "-C", str(ROOT), "show", f"{OLD_REV}:docs/agents/tokens.md"],
                      check=True, capture_output=True, text=True).stdout
old = full[:full.index(CUT)]
_t = subprocess.run(["git", "-C", str(ROOT), "show", f"{NEW_REV}:{NEW_PATH}"],
                    check=True, capture_output=True, text=True).stdout
new = _t[:_t.index(CUT)]
# the preamble sentence added when the halves were joined (metadata, not moved text)
new = new.replace(" Reformatted for readability the same day: the words are unchanged and long paragraphs became bullets"
                  " (checks: `docs/agents/_reformat/check_*.py`).", "")
new_norm = norm(new)
for name, text in (("old part 1", old), ("new part 1", new)):
    longest = max(len(x) for x in text.split("\n"))
    print(f"{name}: {len(text.encode())} bytes, {len(text.splitlines())} lines, longest line {longest} chars")

# Check 1
found, missing, heading = 0, [], None
for line in old.split("\n"):
    if line.startswith("### "):
        heading = line[4:]
    if line in SCAFFOLD:
        continue
    row = ROW.match(line)
    first_cell = norm(row.group(1)) if row else None
    if row and row.group(1) != heading:
        missing.append((line[:80], "first cell differs from its section heading"))
    for frag in re.split(r"\. |; ", line):
        f = norm(frag).rstrip(".;:, ")
        if len(f.split()) < 4:
            continue
        if f in new_norm:
            found += 1
        elif first_cell and f.startswith(first_cell + " ") and f[len(first_cell) + 1:] in new_norm:
            missing.append((f[:110], "row's first cell is now only the heading; the rest of the fragment is found"))
        else:
            missing.append((f[:110], "NOT FOUND"))
total = found + len(missing)
print(f"\nCheck 1: {found} / {total} fragments of >= 4 words found verbatim")
for frag, why in missing:
    print(f"  missing: {frag}... -> {why}")

# Check 2
old_clean = []
for line in old.split("\n"):
    if line in SCAFFOLD:
        continue
    row = ROW.match(line)
    old_clean.append(row.group(2) if row else line)
ow, nw = words("\n".join(old_clean)), words(new)
labels, problems = [], []
for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ow, nw, autojunk=False).get_opcodes():
    if tag == "equal":
        continue
    ins = " ".join(nw[j1:j2])
    if tag == "insert" and j2 - j1 <= 5 and f"**{ins}**" in new:
        labels.append(ins)
    else:
        problems.append(f"{tag}: old {' '.join(ow[i1:i2])!r} -> new {ins!r}")
print(f"\nCheck 2: {len(ow)} old words, {len(nw)} new words; {len(labels)} inserted labels, "
      f"{len(problems)} other differences")
counts = {}
for lab in labels:
    counts[lab] = counts.get(lab, 0) + 1
print("  labels: " + ", ".join(f"{k} x{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
for p in problems:
    print(f"  DIFF {p}")
bad = [m for m in missing if m[1] == "NOT FOUND" or "differs" in m[1]]
print("\nOK" if not bad and not problems else f"\nFAIL: {len(bad)} fragment(s), {len(problems)} word diff(s)")
sys.exit(1 if bad or problems else 0)
