#!/usr/bin/env python3
"""Check the readability pass on part 2 of docs/agents/tokens.md against the old text.

Part 2 is the text of docs/agents/tokens.md at git revision OLD_REV (default 4c3c108) from the heading
line "### `safebranch`" to the end of the file; the new text is docs/agents/_reformat/tokens_part2.md.

Main check (the one asked for):
  1. split the old part 2 into fragments at ". ", "; ", line breaks and table-cell boundaries (" | ";
     a table cell ends a fragment the way a line break does);
  2. normalise old fragments and the whole new file alike: `\\|` unescaped, leading list markers ("- ",
     "* ") and leading "#"s removed, table delimiter rows dropped, "**" and "|" removed, whitespace
     collapsed; fragments also lose trailing punctuation;
  3. every old fragment of 4 or more words must occur in the normalised new text; it is also checked
     to occur in the old order (each found after the end of the previous one).
The old one-row tables are not carried over: their header cells ("Token", "Effect", "Notes") are
dropped, "Label" and "Policy string" become labels, and a token cell must repeat its heading exactly.
Extra checks: fragments of 1-3 words; the headings in order; the multisets of numbers, of `code spans`
and of words, whose only allowed differences are the dropped cells and the added labels.

Usage, from the repository root: python3 docs/agents/_reformat/check_part2.py [OLD_REV]
Exit 1 on any unexplained difference.
"""
import collections
import pathlib
import re
import subprocess
import sys

OLD_REV = sys.argv[1] if len(sys.argv) > 1 else "4c3c108"
ROOT = pathlib.Path(__file__).resolve().parents[3]
NEW_PATH = ROOT / "docs" / "agents" / "tokens.md"   # the joined file (parts merged 2026-11-12)
START = "### `safebranch`"
# Fragments of >= 4 words that are allowed to be missing, with the reason (none at present).
EXPECTED_MISSING = {}
DROPPED_HEADER_CELLS = {"Token", "Effect", "Notes"}

DELIM_ROW = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
LIST_MARK = re.compile(r"^\s*[-*]\s+")
HEAD_MARK = re.compile(r"^\s*#+\s*")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
CODE = re.compile(r"`[^`]+`")
WORD = re.compile(r"\w+")


def norm(text):
    lines = []
    for line in text.replace("\\|", "|").split("\n"):
        if DELIM_ROW.match(line):
            continue
        lines.append(LIST_MARK.sub("", HEAD_MARK.sub("", line)))
    text = " ".join(lines).replace("**", "").replace("|", " ")
    return re.sub(r"\s+", " ", text).strip()


def cells_of(line):
    """A table row's cells, split at the cell boundaries " | " (escaped or in-text pipes never have spaces on both sides here)."""
    return [cell.strip() for cell in line.strip().strip("|").split(" | ")]


def parse_old(text):
    """Old fragments with their source: 'text', 'header' (a table header cell) or 'token' (a token cell)."""
    out, heading, header = [], None, None
    for line in text.split("\n"):
        if not line.strip() or DELIM_ROW.match(line):
            continue
        if line.startswith("#"):
            heading = HEAD_MARK.sub("", line).strip()
        if line.lstrip().startswith("|"):
            cells = cells_of(line)
            if header is None:
                header = cells
                out += [(cell, "header") for cell in cells]
                continue
            pieces = list(zip(cells, ["token" if header == ["Token", "Effect"] else "text"]
                              + ["text"] * (len(cells) - 1)))
            if pieces[0][1] == "token" and pieces[0][0] != heading:
                pieces[0] = (pieces[0][0], "token-mismatch")
            header = None
        else:
            pieces = [(line, "text")]
        for piece, source in pieces:
            if source != "text":
                out.append((piece, source))
                continue
            for frag in re.split(r"\. |; ", piece):
                frag = norm(frag).rstrip(" .;:,")
                if frag:
                    out.append((frag, "text"))
    return out


old_full = subprocess.run(["git", "-C", str(ROOT), "show", f"{OLD_REV}:docs/agents/tokens.md"],
                          check=True, capture_output=True, text=True).stdout
old = old_full[old_full.index("\n" + START + "\n") + 1:]
_t = NEW_PATH.read_text(encoding="utf-8")
new = _t[_t.index("\n" + START + "\n") + 1:]
new_norm = norm(new)
failures = 0

for name, text in (("old part 2", old), ("new part 2", new)):
    lines = text.split("\n")
    longest = max(range(len(lines)), key=lambda i: len(lines[i]))
    print(f"{name}: {len(text.encode())} bytes, {len(text)} chars, {len(lines)} lines, "
          f"longest line {len(lines[longest])} chars (line {longest + 1})")

parsed = parse_old(old)
text_frags = [f for f, s in parsed if s == "text"]
long_frags = [f for f in text_frags if len(f.split()) >= 4]
short_frags = [f for f in text_frags if len(f.split()) < 4]

# Main check: fragments of >= 4 words, found and in the old order.
found, missing, out_of_order, pos = 0, [], [], 0
for frag in long_frags:
    at = new_norm.find(frag, pos)
    if at >= 0:
        found += 1
        pos = at + len(frag)
    elif frag in new_norm:
        found += 1
        out_of_order.append(frag)
    else:
        missing.append(frag)
print(f"\nfragments of >= 4 words found: {found} / {len(long_frags)} "
      f"(in the old order: {found - len(out_of_order)})")
for frag in missing:
    reason = EXPECTED_MISSING.get(frag)
    print(f"  {'expected' if reason else 'MISSING'}: {frag[:150]}" + (f" ({reason})" if reason else ""))
    failures += reason is None
for frag in out_of_order:
    print(f"  OUT OF ORDER: {frag[:150]}")
    failures += 1

# Extra checks.
short_missing = [f for f in short_frags if f not in new_norm]
print(f"fragments of 1-3 words found: {len(short_frags) - len(short_missing)} / {len(short_frags)}")
for frag in short_missing:
    print(f"  MISSING: {frag}")
    failures += 1

headers = collections.Counter(f for f, s in parsed if s == "header")
tokens = [f for f, s in parsed if s in ("token", "token-mismatch")]
bad_tokens = [f for f, s in parsed if s == "token-mismatch"]
print(f"table header cells: {dict(headers)}; dropped: "
      f"{ {k: v for k, v in headers.items() if k in DROPPED_HEADER_CELLS} }, kept as labels: "
      f"{ {k: v for k, v in headers.items() if k not in DROPPED_HEADER_CELLS} }")
kept_labels_ok = all(f"**{k}:**" in new for k in headers if k not in DROPPED_HEADER_CELLS)
print(f"  kept header cells present as bold labels: {kept_labels_ok}")
failures += not kept_labels_ok
print(f"token cells dropped: {len(tokens)}, each identical to its heading: {not bad_tokens}")
for f in bad_tokens:
    print(f"  TOKEN CELL DIFFERS FROM HEADING: {f}")
    failures += 1


def headings(text):
    return [line.strip() for line in text.split("\n") if line.startswith("#")]


same_headings = headings(old) == headings(new)
print(f"headings identical and in the same order: {same_headings} ({len(headings(old))})")
failures += not same_headings

old_numbers = collections.Counter(NUMBER.findall(norm(old)))
new_numbers = collections.Counter(NUMBER.findall(new_norm))
print(f"numbers: {sum(old_numbers.values())} old, {sum(new_numbers.values())} new, "
      f"identical multiset: {old_numbers == new_numbers}")
if old_numbers != new_numbers:
    print(f"  only old: {dict(old_numbers - new_numbers)}; only new: {dict(new_numbers - old_numbers)}")
    failures += 1

dropped_text = " ".join(tokens + [h for h in headers.elements() if h in DROPPED_HEADER_CELLS])
old_code = collections.Counter(CODE.findall(old.replace("\\|", "|")))
new_code = collections.Counter(CODE.findall(new.replace("\\|", "|")))
code_ok = (old_code - new_code == collections.Counter(CODE.findall(dropped_text))
           and not new_code - old_code)
print(f"code spans: {sum(old_code.values())} old, {sum(new_code.values())} new; "
      f"only difference the dropped token cells: {code_ok}")
failures += not code_ok

old_words = collections.Counter(WORD.findall(norm(old)))
new_words = collections.Counter(WORD.findall(new_norm))
words_ok = old_words - new_words == collections.Counter(WORD.findall(dropped_text))
added = new_words - old_words
print(f"words: {sum(old_words.values())} old, {sum(new_words.values())} new; "
      f"words lost are exactly the dropped header and token cells: {words_ok}")
print(f"  words added (the labels): {sum(added.values())}: {dict(sorted(added.items()))}")
failures += not words_ok

print("OK" if not failures else f"FAIL: {failures} problem(s)")
sys.exit(1 if failures else 0)
