"""Create a non-destructive filtered ARCTIC tree from review frame ranges.

The original server output is left untouched. Accepted frames are copied,
hardlinked, or symlinked into a new root with the same directory conventions:

    originals/<sequence>/frame_XXXXXX.jpg
    hand_masks/<sequence>/frame_XXXXXX.png
    object_masks/<sequence>/frame_XXXXXX.png
    bg/<sequence>/frame_XXXXXX.jpg
    pseudo_gt/<robot>/<sequence>/frame_XXXXXX.jpg
    robot_masks/<robot>/<sequence>/frame_XXXXXX.png
    robot_overlay_rgba/<robot>/<sequence>/frame_XXXXXX.png
    qpos/<robot>/<sequence>/retargeted_pose_bimanual.npz
    qpos/<robot>/<sequence>/retargeting_log_bimanual.txt

By default qpos files are copied at sequence level. Pass
``--slice-qpos-to-kept-frames`` to write qpos arrays sliced to exactly the
accepted frame ids for each robot/sequence.

Use ``keep_ranges.csv`` for good frame intervals. ``demo_ranges.csv`` is also
treated as good data and additionally mirrored under ``demo/`` for quick demos.
``reject_ranges.csv`` can subtract local bad spans from keep/demo ranges.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_SERVER_ROOT = Path("/path/to/server_outputs")
DEFAULT_FILTERED_ROOT = Path("/path/to/server_outputs_filtered")
FRAME_RE = re.compile(r"(\d+)$")
QPOS_FILE = "retargeted_pose_bimanual.npz"
QPOS_LOG = "retargeting_log_bimanual.txt"


@dataclass(frozen=True)
class RangeRow:
    robot: str
    sequence: str
    start: int
    end: int
    purpose: str = ""
    reason: str = ""


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_ranges(path: Path | None, default_purpose: str) -> list[RangeRow]:
    ranges: list[RangeRow] = []
    for row in _read_csv(path):
        robot = str(row.get("robot", "")).strip()
        sequence = str(row.get("sequence", "")).strip()
        start_raw = row.get("start_frame") or row.get("start") or row.get("begin") or ""
        end_raw = row.get("end_frame") or row.get("end") or start_raw
        if not robot or not sequence or not str(start_raw).strip():
            continue
        start = int(start_raw)
        end = int(end_raw)
        if end < start:
            start, end = end, start
        ranges.append(
            RangeRow(
                robot=robot,
                sequence=sequence,
                start=start,
                end=end,
                purpose=str(row.get("purpose", "") or default_purpose).strip(),
                reason=str(row.get("reason", "") or row.get("note", "")).strip(),
            )
        )
    return ranges


def _frame_id(path: Path) -> int | None:
    match = FRAME_RE.search(path.stem)
    return int(match.group(1)) if match else None


def _frame_name(frame_id: int) -> str:
    return f"frame_{int(frame_id):06d}"


def _copy_or_link(src: Path, dst: Path, mode: str, *, dry_run: bool) -> bool:
    if not src.is_file():
        return False
    if dry_run:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError("--copy-mode must be copy, hardlink, or symlink")
    return True


def _in_ranges(frame_id: int, ranges: Iterable[RangeRow]) -> bool:
    return any(item.start <= int(frame_id) <= item.end for item in ranges)


def _ranges_by_key(ranges: Iterable[RangeRow]) -> dict[tuple[str, str], list[RangeRow]]:
    result: dict[tuple[str, str], list[RangeRow]] = {}
    for item in ranges:
        result.setdefault((item.robot, item.sequence), []).append(item)
    return result


def _copy_shared_assets(
    *,
    server_root: Path,
    output_root: Path,
    sequence: str,
    frame_id: int,
    mode: str,
    dry_run: bool,
    prefix: Path = Path(),
) -> list[str]:
    frame = _frame_name(frame_id)
    missing: list[str] = []
    specs = [
        ("originals", ".jpg"),
        ("hand_masks", ".png"),
        ("object_masks", ".png"),
        ("bg", ".jpg"),
    ]
    for folder, suffix in specs:
        src = server_root / folder / sequence / f"{frame}{suffix}"
        dst = output_root / prefix / folder / sequence / f"{frame}{suffix}"
        if not _copy_or_link(src, dst, mode, dry_run=dry_run):
            missing.append(str(src))
    return missing


def _copy_robot_assets(
    *,
    server_root: Path,
    output_root: Path,
    robot: str,
    sequence: str,
    frame_id: int,
    mode: str,
    dry_run: bool,
    prefix: Path = Path(),
) -> list[str]:
    frame = _frame_name(frame_id)
    specs = [
        ("pseudo_gt", ".jpg"),
        ("robot_masks", ".png"),
        ("robot_overlay_rgba", ".png"),
    ]
    missing: list[str] = []
    for folder, suffix in specs:
        src = server_root / folder / robot / sequence / f"{frame}{suffix}"
        dst = output_root / prefix / folder / robot / sequence / f"{frame}{suffix}"
        if not _copy_or_link(src, dst, mode, dry_run=dry_run):
            missing.append(str(src))
    return missing


def _copy_qpos(
    *,
    server_root: Path,
    output_root: Path,
    robot: str,
    sequence: str,
    mode: str,
    dry_run: bool,
    prefix: Path = Path(),
) -> list[str]:
    missing: list[str] = []
    for name in (QPOS_FILE, QPOS_LOG):
        src = server_root / "qpos" / robot / sequence / name
        dst = output_root / prefix / "qpos" / robot / sequence / name
        if not _copy_or_link(src, dst, mode, dry_run=dry_run):
            missing.append(str(src))
    return missing


def _read_qpos_log(path: Path) -> dict[str, str]:
    info: dict[str, str] = {}
    if not path.is_file():
        return info
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            info[key.strip()] = value.strip()
    return info


def _qpos_frame_ids_from_log(path: Path) -> list[int]:
    raw = _read_qpos_log(path).get("frame_ids", "")
    return [int(part) for part in raw.split(",") if part.strip()]


def _write_sliced_qpos(
    *,
    server_root: Path,
    output_root: Path,
    robot: str,
    sequence: str,
    frame_ids: list[int],
    dry_run: bool,
    prefix: Path = Path(),
) -> list[str]:
    src_npz = server_root / "qpos" / robot / sequence / QPOS_FILE
    src_log = server_root / "qpos" / robot / sequence / QPOS_LOG
    dst_dir = output_root / prefix / "qpos" / robot / sequence
    dst_npz = dst_dir / QPOS_FILE
    dst_log = dst_dir / QPOS_LOG
    missing: list[str] = []
    if not src_npz.is_file():
        missing.append(str(src_npz))
    if not src_log.is_file():
        missing.append(str(src_log))
    source_frame_ids = _qpos_frame_ids_from_log(src_log)
    if not source_frame_ids:
        missing.append(f"{src_log}: missing frame_ids")
    if missing:
        return missing

    frame_to_index = {int(frame_id): index for index, frame_id in enumerate(source_frame_ids)}
    exported = [int(frame_id) for frame_id in frame_ids if int(frame_id) in frame_to_index]
    skipped = [int(frame_id) for frame_id in frame_ids if int(frame_id) not in frame_to_index]
    if skipped:
        missing.append(f"{src_log}: missing selected frame_ids={skipped[:20]} total={len(skipped)}")
    if not exported:
        missing.append(f"{src_log}: no selected frame_ids match qpos")
        return missing
    if dry_run:
        return missing

    indices = np.asarray([frame_to_index[frame_id] for frame_id in exported], dtype=np.int64)
    payload: dict[str, np.ndarray] = {}
    with np.load(str(src_npz), allow_pickle=True) as src:
        for key in src.files:
            arr = np.asarray(src[key])
            if arr.ndim > 0 and arr.shape[0] == len(source_frame_ids):
                payload[str(key)] = arr[indices].copy()
            else:
                payload[str(key)] = arr.copy()
    dst_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(dst_npz), **payload)
    log_lines = [
        "QPOS summary",
        "=" * 60,
        "dataset=arctic_filtered",
        f"combo={robot}",
        f"sequence={sequence}",
        f"source_qpos={src_npz}",
        f"source_qpos_log={src_log}",
        f"source_frame_count={len(source_frame_ids)}",
        f"exported_start={exported[0]}",
        f"exported_end={exported[-1]}",
        f"exported_frame_count={len(exported)}",
        "frame_ids=" + ",".join(str(frame_id) for frame_id in exported),
        "qpos_keys=" + ";".join(sorted(payload)),
    ]
    for key, arr in sorted(payload.items()):
        log_lines.append(f"{key}: shape={tuple(int(dim) for dim in arr.shape)}")
    dst_log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return missing


def _copy_or_slice_qpos(
    *,
    server_root: Path,
    output_root: Path,
    robot: str,
    sequence: str,
    frame_ids: list[int],
    mode: str,
    dry_run: bool,
    slice_to_frames: bool,
    prefix: Path = Path(),
) -> list[str]:
    if slice_to_frames:
        return _write_sliced_qpos(
            server_root=server_root,
            output_root=output_root,
            robot=robot,
            sequence=sequence,
            frame_ids=frame_ids,
            dry_run=dry_run,
            prefix=prefix,
        )
    return _copy_qpos(
        server_root=server_root,
        output_root=output_root,
        robot=robot,
        sequence=sequence,
        mode=mode,
        dry_run=dry_run,
        prefix=prefix,
    )


def _candidate_frames(server_root: Path, robot: str, sequence: str) -> list[int]:
    path = server_root / "pseudo_gt" / robot / sequence
    frames: list[int] = []
    if path.is_dir():
        for item in path.glob("frame_*.jpg"):
            frame_id = _frame_id(item)
            if frame_id is not None:
                frames.append(frame_id)
    return sorted(set(frames))


def filter_tree(args: argparse.Namespace) -> None:
    server_root = args.server_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    keep_ranges = _parse_ranges(args.keep_ranges_csv, "keep")
    demo_ranges = _parse_ranges(args.demo_ranges_csv, "demo")
    reject_ranges = _parse_ranges(args.reject_ranges_csv, "reject")
    if not keep_ranges and not demo_ranges:
        raise ValueError("No keep/demo ranges found. Mark good ranges before exporting.")

    keep_by_key = _ranges_by_key([*keep_ranges, *demo_ranges])
    demo_by_key = _ranges_by_key(demo_ranges)
    reject_by_key = _ranges_by_key(reject_ranges)

    if args.clean_output and output_root.exists() and not args.dry_run:
        shutil.rmtree(output_root)

    filtered_rows: list[dict[str, object]] = []
    demo_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    qpos_keys_copied: set[tuple[str, str]] = set()
    demo_qpos_keys_copied: set[tuple[str, str]] = set()

    for key, ranges in sorted(keep_by_key.items()):
        robot, sequence = key
        frames = _candidate_frames(server_root, robot, sequence)
        accepted_frames = [
            frame_id
            for frame_id in frames
            if _in_ranges(frame_id, ranges) and not _in_ranges(frame_id, reject_by_key.get(key, []))
        ]
        demo_accepted_frames = [
            frame_id for frame_id in accepted_frames if _in_ranges(frame_id, demo_by_key.get(key, []))
        ]
        qpos_missing_for_key: list[str] = []
        if accepted_frames and key not in qpos_keys_copied:
            qpos_missing_for_key = _copy_or_slice_qpos(
                server_root=server_root,
                output_root=output_root,
                robot=robot,
                sequence=sequence,
                frame_ids=accepted_frames,
                mode=args.copy_mode,
                dry_run=args.dry_run,
                slice_to_frames=args.slice_qpos_to_kept_frames,
            )
            qpos_keys_copied.add(key)
        for index, frame_id in enumerate(accepted_frames):
            missing = []
            missing.extend(
                _copy_shared_assets(
                    server_root=server_root,
                    output_root=output_root,
                    sequence=sequence,
                    frame_id=frame_id,
                    mode=args.copy_mode,
                    dry_run=args.dry_run,
                )
            )
            missing.extend(
                _copy_robot_assets(
                    server_root=server_root,
                    output_root=output_root,
                    robot=robot,
                    sequence=sequence,
                    frame_id=frame_id,
                    mode=args.copy_mode,
                    dry_run=args.dry_run,
                )
            )
            if index == 0:
                missing.extend(qpos_missing_for_key)
            row = {
                "robot": robot,
                "sequence": sequence,
                "frame_id": frame_id,
                "frame_name": _frame_name(frame_id),
                "is_demo": int(_in_ranges(frame_id, demo_by_key.get(key, []))),
            }
            filtered_rows.append(row)
            for path in missing:
                missing_rows.append({**row, "missing": path})

            if row["is_demo"]:
                demo_missing = []
                demo_missing.extend(
                    _copy_shared_assets(
                        server_root=server_root,
                        output_root=output_root,
                        sequence=sequence,
                        frame_id=frame_id,
                        mode=args.copy_mode,
                        dry_run=args.dry_run,
                        prefix=Path("demo"),
                    )
                )
                demo_missing.extend(
                    _copy_robot_assets(
                        server_root=server_root,
                        output_root=output_root,
                        robot=robot,
                        sequence=sequence,
                        frame_id=frame_id,
                        mode=args.copy_mode,
                        dry_run=args.dry_run,
                        prefix=Path("demo"),
                    )
                )
                if key not in demo_qpos_keys_copied:
                    demo_missing.extend(
                        _copy_or_slice_qpos(
                            server_root=server_root,
                            output_root=output_root,
                            robot=robot,
                            sequence=sequence,
                            frame_ids=demo_accepted_frames,
                            mode=args.copy_mode,
                            dry_run=args.dry_run,
                            slice_to_frames=args.slice_qpos_to_kept_frames,
                            prefix=Path("demo"),
                        )
                    )
                    demo_qpos_keys_copied.add(key)
                demo_rows.append(row)
                for path in demo_missing:
                    missing_rows.append({**row, "missing": path, "subset": "demo"})

    manifest_dir = output_root / "manifests"
    _write_csv(
        manifest_dir / "filtered_frames.csv",
        filtered_rows,
        ["robot", "sequence", "frame_id", "frame_name", "is_demo"],
    )
    _write_csv(
        manifest_dir / "demo_frames.csv",
        demo_rows,
        ["robot", "sequence", "frame_id", "frame_name", "is_demo"],
    )
    _write_csv(
        manifest_dir / "missing_assets.csv",
        missing_rows,
        ["robot", "sequence", "frame_id", "frame_name", "is_demo", "subset", "missing"],
    )
    summary = {
        "server_root": str(server_root),
        "output_root": str(output_root),
        "copy_mode": args.copy_mode,
        "dry_run": bool(args.dry_run),
        "slice_qpos_to_kept_frames": bool(args.slice_qpos_to_kept_frames),
        "keep_range_count": len(keep_ranges),
        "demo_range_count": len(demo_ranges),
        "reject_range_count": len(reject_ranges),
        "filtered_frame_count": len(filtered_rows),
        "demo_frame_count": len(demo_rows),
        "robot_sequence_count": len(qpos_keys_copied),
        "missing_asset_count": len(missing_rows),
    }
    if not args.dry_run:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "filter_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if missing_rows and not args.allow_missing:
        raise SystemExit(f"Missing assets: {len(missing_rows)}; see {manifest_dir / 'missing_assets.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-root", type=Path, default=DEFAULT_SERVER_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FILTERED_ROOT)
    parser.add_argument("--keep-ranges-csv", type=Path, required=True)
    parser.add_argument("--demo-ranges-csv", type=Path, default=None)
    parser.add_argument("--reject-ranges-csv", type=Path, default=None)
    parser.add_argument("--copy-mode", choices=["copy", "hardlink", "symlink"], default="hardlink")
    parser.add_argument(
        "--slice-qpos-to-kept-frames",
        action="store_true",
        help="Write qpos arrays sliced to accepted frame ids instead of copying whole sequence qpos.",
    )
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    filter_tree(build_parser().parse_args())


if __name__ == "__main__":
    main()
