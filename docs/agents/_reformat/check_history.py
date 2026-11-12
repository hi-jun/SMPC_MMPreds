#!/usr/bin/env python3
"""Check that the 2026-11-12 readability pass of three docs/agents files kept every sentence fragment.

Files: design_history.md, ngsim_replay_defaults.md, controller.md. Per file, the old text is
`git show OLD_REV:<path>` (default 4c3c108, the split version before the pass). It is cut into
fragments at ". ", "; " and line breaks. Fragments and the new text are normalised alike: a list
marker ("- ", "* ") or heading mark ("#") at the start of a line is removed, "**" and "|" are removed
everywhere, and whitespace runs (line breaks included) become one space. The new text is normalised
as one string, so a sentence now spread over bullets reads as it did. A fragment also loses its
trailing punctuation (.,;:!?), so it matches whatever now follows it (a semicolon that became a
bullet break). Every fragment of MIN_WORDS words or more must occur in the normalised new text.
Fragments changed on purpose go in EXPECTED_CHANGES with the reason; any other missing fragment
makes the script exit 1.

Three stricter checks follow, also exit 1 on failure:
  - counts: every such fragment occurs at least as often in the new text as in the old one;
  - tokens: every number and every `code span` of the old text occurs at least as often in the new;
  - words: a word-level diff of the two normalised texts (trailing .,;:!? stripped from each word)
    may only insert words; every inserted run is printed, so it can be read as an added heading or
    label (at most 5 words each), and every deleted or replaced run is a failure.

Usage, from the repository root: python3 docs/agents/_reformat/check_history.py [OLD_REV]
"""
import collections
import difflib
import pathlib
import re
import subprocess
import sys

OLD_REV = sys.argv[1] if len(sys.argv) > 1 else "4c3c108"
ROOT = pathlib.Path(__file__).resolve().parents[3]
FILES = ["docs/agents/design_history.md", "docs/agents/ngsim_replay_defaults.md",
         "docs/agents/controller.md"]
MIN_WORDS = 4
EXPECTED_CHANGES = {}  # (path, normalised old fragment) -> reason

LINE_MARK = re.compile(r"^\s*(?:[-*]\s+|#+\s*)")
TRAILING_PUNCT = ".,;:!? "
NUMBER = re.compile(r"[−+-]?\d+(?:\.\d+)?")
CODE = re.compile(r"`[^`]+`")


def normalise(text):
    lines = [LINE_MARK.sub("", line, count=1) for line in text.split("\n")]
    text = " ".join(lines).replace("**", "").replace("|", "")
    return " ".join(text.split())


def words(norm):
    return [w for w in (w.rstrip(TRAILING_PUNCT) for w in norm.split()) if w]


failed = False
for path in FILES:
    old = subprocess.run(["git", "-C", str(ROOT), "show", f"{OLD_REV}:{path}"],
                         check=True, capture_output=True, text=True).stdout
    new = (ROOT / path).read_text(encoding="utf-8")
    old_norm, new_norm = normalise(old), normalise(new)
    print(f"== {path}")
    for name, text in ((f"old ({OLD_REV})", old), ("new", new)):
        longest = max(len(line) for line in text.split("\n"))
        print(f"  {name}: {len(text.encode())} bytes, {len(text.splitlines())} lines, "
              f"longest line {longest} chars")

    # Required check: every fragment of >= MIN_WORDS words occurs in the new text.
    fragments = []
    for line in old.split("\n"):
        for piece in re.split(r"\. |; ", line):
            frag = normalise(piece).rstrip(TRAILING_PUNCT)
            if len(frag.split()) >= MIN_WORDS:
                fragments.append(frag)
    missing = [frag for frag in fragments if frag not in new_norm]
    unexpected = [frag for frag in missing if (path, frag) not in EXPECTED_CHANGES]
    print(f"  fragments of >= {MIN_WORDS} words: {len(fragments)}; found "
          f"{len(fragments) - len(missing)}, missing {len(missing)}")
    for frag in missing:
        reason = EXPECTED_CHANGES.get((path, frag), "UNEXPECTED (not in EXPECTED_CHANGES)")
        print(f"    missing: {frag[:120]} -- {reason}")

    # Stricter check 1: no fragment occurs fewer times than before.
    fewer = [(frag, old_norm.count(frag), new_norm.count(frag)) for frag in sorted(set(fragments))
             if new_norm.count(frag) < old_norm.count(frag)]
    print(f"  counts: {len(fewer)} distinct fragment(s) occur fewer times than before")
    for frag, n_old, n_new in fewer:
        print(f"    {n_old} -> {n_new}: {frag[:120]}")

    # Stricter check 2: numbers and code spans.
    lost = []
    for kind, pattern in (("number", NUMBER), ("code span", CODE)):
        short = collections.Counter(pattern.findall(old_norm)) - collections.Counter(pattern.findall(new_norm))
        lost += [(kind, token, n) for token, n in short.items()]
    print(f"  tokens: {len(lost)} number(s) or code span(s) occur fewer times than before")
    for kind, token, n in lost:
        print(f"    {kind} {token[:100]} short by {n}")

    # Stricter check 3: word-level diff, only insertions allowed.
    old_words, new_words = words(old_norm), words(new_norm)
    matcher = difflib.SequenceMatcher(None, old_words, new_words, autojunk=False)
    inserted, changed = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            inserted.append(" ".join(new_words[j1:j2]))
        elif tag != "equal":
            changed.append((tag, " ".join(old_words[i1:i2]), " ".join(new_words[j1:j2])))
    print(f"  words: {len(old_words)} old, {len(new_words)} new; {len(inserted)} inserted run(s), "
          f"{len(changed)} deleted or replaced run(s)")
    for run in inserted:
        print(f"    inserted ({len(run.split())} words): {run}")
    for tag, before, after in changed:
        print(f"    {tag.upper()}: old '{before[:100]}' -> new '{after[:100]}'")

    failed |= bool(unexpected or fewer or lost or changed)

print("OK" if not failed else "FAIL")
sys.exit(1 if failed else 0)
