#!/usr/bin/env python3
"""Check that the 2026-11-12 split of AGENTS.md kept every line.

Every non-empty line of the old AGENTS.md (git revision OLD_REV, default 3d9bcf8, the last commit
before the split), stripped of leading and trailing whitespace, must appear as a stripped line of
AGENTS.md or of a file docs/agents/*.md (a set check). Lines changed on purpose are listed in
EXPECTED_CHANGES with the reason; the script exits 1 on any other missing line.

The new files are read at NEW_REV (default 4c3c108, the split commit), because the readability pass
after it re-flowed the lines; that pass is checked by docs/agents/_reformat/check_*.py.

Usage, from the repository root: python3 docs/agents/check_split.py [OLD_REV] [NEW_REV]
"""
import pathlib
import subprocess
import sys

OLD_REV = sys.argv[1] if len(sys.argv) > 1 else "3d9bcf8"
NEW_REV = sys.argv[2] if len(sys.argv) > 2 else "4c3c108"
ROOT = pathlib.Path(__file__).resolve().parents[2]
EXPECTED_CHANGES = {
    "It describes the state on 2026-11-04, branch `feat/probability-gated-cutin`. Where this file and the code "
    "disagree, the code is right.":
        "stale date; AGENTS.md line 4 now gives 2026-11-12 and points to docs/agents/",
}


def stripped_lines(text):
    return [line.strip() for line in text.split("\n") if line.strip()]


old = subprocess.run(["git", "-C", str(ROOT), "show", f"{OLD_REV}:AGENTS.md"],
                     check=True, capture_output=True, text=True).stdout
listed = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "--name-only", NEW_REV, "docs/agents/"],
                        check=True, capture_output=True, text=True).stdout.split()
new_files = ["AGENTS.md"] + sorted(p for p in listed if p.endswith(".md"))


def at_rev(path):
    return subprocess.run(["git", "-C", str(ROOT), "show", f"{NEW_REV}:{path}"],
                          check=True, capture_output=True, text=True).stdout

new = set()
for path in new_files:
    new.update(stripped_lines(at_rev(path)))

old_lines = stripped_lines(old)
missing = [line for line in old_lines if line not in new]
unexpected = [line for line in missing if line not in EXPECTED_CHANGES]

print(f"old AGENTS.md at {OLD_REV}: {len(old.encode())} bytes, {len(old_lines)} non-empty lines")
for path in new_files:
    text = at_rev(path)
    longest = max(len(line) for line in text.split("\n"))
    print(f"  {path} at {NEW_REV}: {len(text.encode())} bytes, longest line {longest} chars")
print(f"old lines found verbatim: {len(old_lines) - len(missing)} / {len(old_lines)}")
for line in missing:
    if line in EXPECTED_CHANGES:
        print(f"  expected change: {line[:90]}... ({EXPECTED_CHANGES[line]})")
    else:
        print(f"  MISSING: {line[:120]}")
print("OK" if not unexpected else f"FAIL: {len(unexpected)} line(s) missing")
sys.exit(1 if unexpected else 0)
