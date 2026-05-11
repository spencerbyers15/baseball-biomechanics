"""Extract vtable field offsets for FieldVision event classes from the JS bundle.

Reads samples/mlb_bundles/gd.@bvg_poser.min.js and prints, for each class of
interest, an ordered list of (vtable_offset, fieldName, addExpression).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = REPO_ROOT / "samples" / "mlb_bundles" / "gd.@bvg_poser.min.js"

CLASSES_OF_INTEREST = [
    "GameEventWire",
    "TrackedEventWire",
    "PlayEventDataWire",
    "CountEventDataWire",
    "AtBatEventDataWire",
    "HandedEventDataWire",
    "BallPitchDataWire",
    "BatImpactEventDataWire",
    "InningEventDataWire",
]


def find_class_block(src: str, name: str) -> str | None:
    """Return the substring covering the class definition, or None.

    The minified bundle uses minified class names (e.g. xM, not GameEventWire),
    so we cannot search for 'class <Name>'. Instead, we locate the
    getRootAs<Name> call which always lives inside the class body, then
    walk backward through the source to find the enclosing class opening brace,
    then forward to the matching closing brace.
    """
    pat = re.compile(rf"\bgetRootAs{re.escape(name)}\s*\(")
    m = pat.search(src)
    if not m:
        return None

    # Walk backward from getRootAs<Name> to find the enclosing class's opening
    # brace (the one that started the class body we're inside).
    pos = m.start()
    depth = 0
    class_start = -1
    i = pos
    while i >= 0:
        c = src[i]
        if c == "}":
            depth += 1
        elif c == "{":
            if depth == 0:
                class_start = i
                break
            depth -= 1
        i -= 1

    if class_start < 0:
        return None

    # Walk forward from the opening brace to the matching closing brace.
    depth = 1
    i = class_start + 1
    while i < len(src) and depth > 0:
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    class_end = i
    return src[class_start:class_end]


def extract_add_methods(class_body: str) -> list[tuple[int, str, str]]:
    """Find static add<FieldName>(b, ...) methods in source order. Returns
    [(vtable_offset, field_name, raw_expression), ...].

    The minified bundle uses patterns like:
      static addSpeed(t,e){t.addFieldFloat32(0,e,0)}
    where the first argument to addFieldXxx is the field index.
    """
    # Match: static add<Name>(t, ...) { ... addField<Type>(<idx>, ...) ... }
    # The class body is one long line, so [^}]* can work but may be greedy.
    # We use a non-greedy match inside the brace.
    method_pat = re.compile(
        r"static\s+add([A-Z][A-Za-z0-9_]*)\s*\([^)]*\)\s*\{([^}]*)\}",
        re.DOTALL,
    )
    idx_pat = re.compile(r"addField\w+\s*\(\s*(\d+)\s*,")
    out: list[tuple[int, str, str]] = []
    for mm in method_pat.finditer(class_body):
        name = mm.group(1)
        body = mm.group(2)
        idx_match = idx_pat.search(body)
        if not idx_match:
            continue
        field_idx = int(idx_match.group(1))
        vtable_offset = (field_idx + 1) * 2 + 2  # field 0 -> 4, field 1 -> 6, ...
        out.append((vtable_offset, name, body.strip()))
    out.sort(key=lambda r: r[0])
    return out


def main() -> int:
    if not BUNDLE.exists():
        print(f"ERROR: bundle missing at {BUNDLE}. Run Task 1 first.", file=sys.stderr)
        return 1
    src = BUNDLE.read_text(errors="replace")
    print(f"# Field offsets extracted from {BUNDLE.name} ({len(src):,} chars)")
    print()
    for cls in CLASSES_OF_INTEREST:
        body = find_class_block(src, cls)
        if body is None:
            print(f"## {cls}\n\n  WARNING: NOT FOUND in bundle. Search for alternate name.\n")
            continue
        methods = extract_add_methods(body)
        print(f"## {cls}\n")
        if not methods:
            print("  (no static add* methods found -- class may be a non-table.)")
            print()
            continue
        print("| vtable_offset | field |")
        print("|---|---|")
        for off, name, _expr in methods:
            print(f"| {off} | {name} |")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
