"""Validate sequence-review keep/demo/reject CSV files.

This is a read-only helper. It checks that marked ranges exist in the review
manifest, stay inside each sequence's frame span, and that every demo range is
also mirrored into the keep CSV.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _range_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("robot", ""),
        row.get("sequence", ""),
        row.get("start_frame", ""),
        row.get("end_frame", ""),
    )


def check(args: argparse.Namespace) -> int:
    manifest_rows = _read_csv(args.manifest)
    meta = {(row.get("robot", ""), row.get("sequence", "")): row for row in manifest_rows}
    tables = {
        "KEEP": _read_csv(args.keep_csv),
        "DEMO": _read_csv(args.demo_csv),
        "REJECT": _read_csv(args.reject_csv),
    }

    print("===== files =====")
    for name, path in [
        ("manifest", args.manifest),
        ("keep", args.keep_csv),
        ("demo", args.demo_csv),
        ("reject", args.reject_csv),
    ]:
        print(f"{name:8s}", "OK" if path and path.is_file() else "MISSING", path)

    print("\n===== counts =====")
    for name, rows in tables.items():
        sequences = {(row.get("robot", ""), row.get("sequence", "")) for row in rows}
        print(f"{name:8s} rows={len(rows):5d} sequences={len(sequences):5d}")

    print("\n===== by robot =====")
    for name, rows in tables.items():
        counts = Counter(row.get("robot", "") for row in rows)
        print(f"\n{name}")
        for robot, count in sorted(counts.items()):
            print(f"  {robot}: {count}")

    print("\n===== latest rows =====")
    for name, rows in tables.items():
        print(f"\n{name} latest {args.latest}")
        for row in rows[-args.latest :]:
            print(
                f"  {row.get('robot')},{row.get('sequence')},"
                f"{row.get('start_frame')}-{row.get('end_frame')}, "
                f"reason={row.get('reason', '')}"
            )

    print("\n===== validation =====")
    bad: list[tuple[str, int, tuple[str, str], str]] = []
    full = defaultdict(int)
    partial = defaultdict(int)

    for name, rows in tables.items():
        for line_number, row in enumerate(rows, start=2):
            key = (row.get("robot", ""), row.get("sequence", ""))
            manifest_row = meta.get(key)
            if not manifest_row:
                bad.append((name, line_number, key, "sequence not in review manifest"))
                continue
            try:
                start = int(row.get("start_frame", ""))
                end = int(row.get("end_frame", ""))
                first = int(manifest_row.get("first_frame", "0"))
                last = int(manifest_row.get("last_frame", "0"))
            except ValueError:
                bad.append((name, line_number, key, "bad frame integer"))
                continue
            if start > end:
                bad.append((name, line_number, key, f"start>end {start}>{end}"))
            if end < first or start > last:
                bad.append((name, line_number, key, f"outside span {start}-{end} vs {first}-{last}"))
            if start <= first and end >= last:
                full[name] += 1
            else:
                partial[name] += 1

    for name in tables:
        print(f"{name:8s} full_sequence={full[name]} partial_ranges={partial[name]}")

    keep_keys = {_range_key(row) for row in tables["KEEP"]}
    demo_keys = {_range_key(row) for row in tables["DEMO"]}
    demo_not_in_keep = sorted(demo_keys - keep_keys)
    print(f"DEMO exact ranges not mirrored in KEEP: {len(demo_not_in_keep)}")
    for item in demo_not_in_keep[: args.latest]:
        print("  ", item)

    print("\nerrors:", len(bad))
    for item in bad[: args.latest * 5]:
        print("  ", item)
    return 1 if bad or demo_not_in_keep else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--keep-csv", type=Path, required=True)
    parser.add_argument("--demo-csv", type=Path, default=None)
    parser.add_argument("--reject-csv", type=Path, default=None)
    parser.add_argument("--latest", type=int, default=10)
    return parser


def main() -> None:
    raise SystemExit(check(build_parser().parse_args()))


if __name__ == "__main__":
    main()
