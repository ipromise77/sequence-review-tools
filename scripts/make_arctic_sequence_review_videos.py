"""Build per-sequence review videos from ARCTIC server SAPIEN outputs.

The server output is laid out as:

    pseudo_gt/<robot>/<sequence>/frame_XXXXXX.jpg

This script stitches each sequence into an mp4 and writes a manifest that can be
loaded by ``serve_arctic_sequence_review.py``.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2


DEFAULT_SERVER_ROOT = Path("/path/to/server_outputs")


def _frame_id(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else -1


def _robots(root: Path, value: str | None) -> list[str]:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    pseudo_root = root / "pseudo_gt"
    return sorted(path.name for path in pseudo_root.iterdir() if path.is_dir())


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "robot",
        "sequence",
        "video",
        "frame_count",
        "first_frame",
        "last_frame",
        "pseudo_gt_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_videos(args: argparse.Namespace) -> None:
    server_root = args.server_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    video_root = output_root / "videos"
    video_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    for robot in _robots(server_root, args.robots):
        seq_root = server_root / "pseudo_gt" / robot
        if not seq_root.is_dir():
            print(f"skip missing robot dir: {seq_root}", flush=True)
            continue
        sequences = sorted(path for path in seq_root.iterdir() if path.is_dir())
        if args.limit_sequences is not None:
            sequences = sequences[: int(args.limit_sequences)]
        for index, sequence_dir in enumerate(sequences, start=1):
            frames = sorted(sequence_dir.glob("frame_*.jpg"), key=_frame_id)
            if not frames:
                print(f"skip empty: {robot}/{sequence_dir.name}", flush=True)
                continue
            video_dir = video_root / robot
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"{sequence_dir.name}.mp4"
            if args.manifest_only:
                print(f"[{robot} {index}/{len(sequences)}] indexed {sequence_dir.name} frames={len(frames)}", flush=True)
            elif video_path.is_file() and not args.overwrite:
                print(f"[{robot} {index}/{len(sequences)}] exists {video_path}", flush=True)
            else:
                first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
                if first is None:
                    print(f"skip unreadable first frame: {frames[0]}", flush=True)
                    continue
                height, width = first.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*args.fourcc),
                    float(args.fps),
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {video_path}")
                for frame_path in frames:
                    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                    if image is None:
                        raise RuntimeError(f"Failed to read frame: {frame_path}")
                    if image.shape[:2] != (height, width):
                        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                    writer.write(image)
                writer.release()
                print(
                    f"[{robot} {index}/{len(sequences)}] wrote {video_path} frames={len(frames)}",
                    flush=True,
                )
            manifest_rows.append(
                {
                    "robot": robot,
                    "sequence": sequence_dir.name,
                    "video": str(video_path),
                    "frame_count": len(frames),
                    "first_frame": _frame_id(frames[0]),
                    "last_frame": _frame_id(frames[-1]),
                    "pseudo_gt_dir": str(sequence_dir),
                }
            )
    manifest_path = args.manifest or (output_root / "review_video_manifest.csv")
    _write_csv(manifest_path, manifest_rows)
    print(f"manifest={manifest_path}", flush=True)
    print(f"videos={len(manifest_rows)}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-root", type=Path, default=DEFAULT_SERVER_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--robots", default=None, help="Comma-separated robot names. Default: all pseudo_gt dirs.")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--fourcc", default="mp4v")
    parser.add_argument("--limit-sequences", type=int, default=None)
    parser.add_argument("--manifest-only", action="store_true", help="Only write the review manifest; do not encode MP4.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = args.server_root / "_work" / "arctic_sequence_review"
    build_videos(args)


if __name__ == "__main__":
    main()
