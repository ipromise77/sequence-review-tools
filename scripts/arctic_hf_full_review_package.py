"""Review, package, verify, and upload server ARCTIC outputs for HF Full.

This utility is intentionally aligned with the public reviewer layout in
``HandEdit/HandEdit``:

    train-samples/Hand/
      originals/<id>.jpg
      pseudo_gt/<id>.jpg
      hand_masks/<id>.png
      object_masks/<id>.png
      robot_masks/<id>.png
      qpos/<id>/retargeted_pose_bimanual.npz
      qpos/<id>/retargeting_log_bimanual.txt
      metadata/train-samples.json

Server ARCTIC qpos is sequence-level. Packaging slices it into per-sample qpos
windows by matching the selected frame id against the frame ids recorded in the
sequence qpos log.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SERVER_ROOT = Path("/path/to/server_outputs")
DEFAULT_UPLOAD_ROOT = Path("/path/to/hf_staging")
DEFAULT_REPO_ID = "YOUR_ORG/YOUR_DATASET"
DEFAULT_SOURCE_DATASET = "arctic"
METADATA_REL = Path("train-samples") / "Hand" / "metadata" / "train-samples.json"
PROVENANCE_NAME = "arctic_hf_upload_provenance.csv"
CATEGORIES = ["hand_masks", "object_masks", "originals", "pseudo_gt", "robot_masks"]
QPOS_FILE = "retargeted_pose_bimanual.npz"
QPOS_LOG = "retargeting_log_bimanual.txt"
TRUTHY = {"1", "true", "yes", "y", "ok", "pass", "keep", "use", "x", "v", "√", "✓", "勾", "打勾"}


@dataclass(frozen=True)
class FrameAsset:
    sequence: str
    frame_id: int
    frame_name: str
    original: Path
    hand_mask: Path
    object_mask: Path
    bg: Path | None = None
    source_manifest: str = ""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["review_ok"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _copy_or_link(src: Path, dst: Path, mode: str) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    normalized = str(mode).strip().lower()
    if normalized == "copy":
        shutil.copy2(src, dst)
    elif normalized == "symlink":
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    elif normalized == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError("--copy-mode must be copy, symlink, or hardlink")


def _frame_name(frame_id: int) -> str:
    return f"frame_{int(frame_id):06d}"


def _parse_frame_id_from_name(path_or_name: str) -> int | None:
    stem = Path(str(path_or_name)).stem
    match = re.search(r"(\d+)$", stem)
    if not match:
        return None
    return int(match.group(1))


def _default_review_csv(server_root: Path) -> Path:
    return server_root / "manifests" / "arctic_hf_sequence_review.csv"


def _selected_manifest_candidates(server_root: Path, selected_manifest: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if selected_manifest is not None:
        candidates.append(selected_manifest)
    candidates.extend(
        [
            server_root / "manifest.csv",
            server_root / "manifests" / "selected_manifest.csv",
            server_root / "manifests" / "preprocess_manifest.csv",
        ]
    )
    seen: set[str] = set()
    result: list[Path] = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _load_frame_assets(
    *,
    server_root: Path,
    selected_manifest: Path | None = None,
    preprocess_manifest: Path | None = None,
) -> dict[str, dict[int, FrameAsset]]:
    candidates = _selected_manifest_candidates(server_root, selected_manifest)
    if preprocess_manifest is not None:
        candidates.insert(0, preprocess_manifest)

    assets: dict[str, dict[int, FrameAsset]] = {}
    used_any = False
    for manifest in candidates:
        if not manifest.is_file():
            continue
        rows = _read_csv(manifest)
        if not rows:
            continue
        fields = set(rows[0].keys())
        used_any = True
        for row in rows:
            status = str(row.get("status", "ok")).strip().lower()
            if status and status not in {"ok", "hand_mask_ok", "object_mask_ok", "bg_ok", "existing"}:
                continue
            sequence = str(
                row.get("sequence_key")
                or row.get("sequence")
                or row.get("capture_name")
                or ""
            ).strip()
            if not sequence:
                continue
            raw_frame = (
                row.get("frame_id")
                or row.get("export_frame_index")
                or row.get("frame_id_zero_based")
                or ""
            )
            frame_id = int(raw_frame) if str(raw_frame).strip() else None
            original_text = (
                row.get("original")
                or row.get("original_file")
                or row.get("source_original_path")
                or ""
            )
            if frame_id is None and original_text:
                frame_id = _parse_frame_id_from_name(original_text)
            if frame_id is None:
                continue
            frame_name = _frame_name(frame_id)
            original = Path(original_text) if original_text else server_root / "originals" / sequence / f"{frame_name}.jpg"
            hand_mask_text = row.get("hand_mask") or row.get("hand_mask_file") or row.get("human_mask_file") or ""
            object_mask_text = row.get("object_mask") or row.get("object_mask_file") or ""
            bg_text = row.get("bg") or row.get("bg_file") or ""
            hand_mask = Path(hand_mask_text) if hand_mask_text else server_root / "hand_masks" / sequence / f"{frame_name}.png"
            object_mask = Path(object_mask_text) if object_mask_text else server_root / "object_masks" / sequence / f"{frame_name}.png"
            bg = Path(bg_text) if bg_text else None
            assets.setdefault(sequence, {})[frame_id] = FrameAsset(
                sequence=sequence,
                frame_id=frame_id,
                frame_name=frame_name,
                original=original,
                hand_mask=hand_mask,
                object_mask=object_mask,
                bg=bg,
                source_manifest=str(manifest),
            )
    if not used_any:
        raise FileNotFoundError(
            "Could not find a selected/preprocess manifest. Pass --selected-manifest "
            "or --preprocess-manifest."
        )
    return assets


def _read_qpos_log(path: Path) -> dict[str, str]:
    info: dict[str, str] = {}
    if not path.is_file():
        return info
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        info[key.strip()] = value.strip()
    return info


def _qpos_frame_ids_from_log(path: Path) -> list[int]:
    info = _read_qpos_log(path)
    raw = info.get("frame_ids", "")
    if not raw:
        return []
    return [int(part) for part in raw.split(",") if part.strip()]


def _qpos_keys(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with np.load(str(path), allow_pickle=True) as payload:
        keys: list[str] = []
        for key in payload.files:
            text = str(key)
            if text.endswith("_root_pose") or text.endswith("_error"):
                continue
            arr = np.asarray(payload[key])
            if arr.ndim >= 2:
                keys.append(text)
        return sorted(keys)


def _robots_from_arg_or_dirs(server_root: Path, robots: str | None) -> list[str]:
    if robots:
        return [item.strip() for item in robots.split(",") if item.strip()]
    root = server_root / "pseudo_gt"
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.is_dir() else 0


def make_review(args: argparse.Namespace) -> None:
    server_root = args.server_root.expanduser().resolve()
    assets = _load_frame_assets(
        server_root=server_root,
        selected_manifest=args.selected_manifest,
        preprocess_manifest=args.preprocess_manifest,
    )
    robots = _robots_from_arg_or_dirs(server_root, args.robots)
    if not robots:
        raise ValueError(f"No robot folders found under {server_root / 'pseudo_gt'}")

    rows: list[dict[str, Any]] = []
    for robot in robots:
        pseudo_robot_dir = server_root / "pseudo_gt" / robot
        sequences = sorted(path.name for path in pseudo_robot_dir.iterdir() if path.is_dir()) if pseudo_robot_dir.is_dir() else []
        for sequence in sequences:
            frame_assets = assets.get(sequence, {})
            expected = len(frame_assets)
            pseudo_dir = server_root / "pseudo_gt" / robot / sequence
            mask_dir = server_root / "robot_masks" / robot / sequence
            rgba_dir = server_root / "robot_overlay_rgba" / robot / sequence
            qpos_dir = server_root / "qpos" / robot / sequence
            qpos_npz = qpos_dir / QPOS_FILE
            qpos_log = qpos_dir / QPOS_LOG
            qpos_frames = _qpos_frame_ids_from_log(qpos_log)
            qpos_keys = _qpos_keys(qpos_npz)
            jpg_count = _count_files(pseudo_dir, "*.jpg")
            mask_count = _count_files(mask_dir, "*.png")
            rgba_count = _count_files(rgba_dir, "*.png")
            complete = (
                expected > 0
                and jpg_count == expected
                and mask_count == expected
                and rgba_count == expected
                and qpos_npz.is_file()
                and qpos_log.is_file()
                and len(qpos_frames) > 0
                and bool(qpos_keys)
            )
            first_frame = min(frame_assets) if frame_assets else ""
            last_frame = max(frame_assets) if frame_assets else ""
            first_preview = next(iter(sorted(pseudo_dir.glob("*.jpg"))), "")
            rows.append(
                {
                    "review_ok": "",
                    "robot": robot,
                    "sequence": sequence,
                    "complete": int(bool(complete)),
                    "expected_frames": expected,
                    "pseudo_gt_count": jpg_count,
                    "robot_mask_count": mask_count,
                    "overlay_rgba_count": rgba_count,
                    "qpos_frame_count": len(qpos_frames),
                    "qpos_keys": ";".join(qpos_keys),
                    "first_frame": first_frame,
                    "last_frame": last_frame,
                    "preview": str(first_preview),
                    "pseudo_gt_dir": str(pseudo_dir),
                    "robot_mask_dir": str(mask_dir),
                    "overlay_rgba_dir": str(rgba_dir),
                    "qpos_npz": str(qpos_npz),
                    "qpos_log": str(qpos_log),
                    "review_note": "",
                }
            )
    out = args.review_csv or _default_review_csv(server_root)
    fields = [
        "review_ok",
        "robot",
        "sequence",
        "complete",
        "expected_frames",
        "pseudo_gt_count",
        "robot_mask_count",
        "overlay_rgba_count",
        "qpos_frame_count",
        "qpos_keys",
        "first_frame",
        "last_frame",
        "preview",
        "pseudo_gt_dir",
        "robot_mask_dir",
        "overlay_rgba_dir",
        "qpos_npz",
        "qpos_log",
        "review_note",
    ]
    _write_csv(out, rows, fields)
    print(f"review_csv={out}")
    print(f"robots={len(robots)} rows={len(rows)} complete={sum(int(r['complete']) for r in rows)}")


def _is_review_ok(value: str) -> bool:
    return str(value).strip().lower() in TRUTHY


def _load_reject_ranges(path: Path | None) -> dict[tuple[str, str], list[tuple[int, int, str]]]:
    if path is None or not path.is_file():
        return {}
    ranges: dict[tuple[str, str], list[tuple[int, int, str]]] = {}
    for row in _read_csv(path):
        robot = str(row.get("robot", "")).strip()
        sequence = str(row.get("sequence", "")).strip()
        if not robot or not sequence:
            continue
        start_raw = row.get("start_frame") or row.get("start") or row.get("begin") or ""
        end_raw = row.get("end_frame") or row.get("end") or start_raw
        if not str(start_raw).strip():
            continue
        start = int(start_raw)
        end = int(end_raw)
        if end < start:
            start, end = end, start
        reason = str(row.get("reason", "") or row.get("note", "")).strip()
        ranges.setdefault((robot, sequence), []).append((start, end, reason))
    return ranges


def _is_rejected_frame(
    reject_ranges: dict[tuple[str, str], list[tuple[int, int, str]]],
    robot: str,
    sequence: str,
    frame_id: int,
) -> tuple[bool, str]:
    for start, end, reason in reject_ranges.get((robot, sequence), []):
        if start <= int(frame_id) <= end:
            return True, reason
    return False, ""


def _hand_dir(upload_root: Path) -> Path:
    return upload_root / "train-samples" / "Hand"


def _load_existing_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _max_sample_id(samples: list[dict[str, Any]]) -> int:
    max_id = 0
    for sample in samples:
        sid = str(sample.get("id", "")).strip()
        if sid.isdigit():
            max_id = max(max_id, int(sid))
    return max_id


def _requested_window(frame_ids: list[int], target_frame: int, before: int, after: int, allow_clipped: bool) -> tuple[list[int], list[int], list[int], str]:
    frame_to_index = {int(frame_id): index for index, frame_id in enumerate(frame_ids)}
    requested = list(range(int(target_frame) - int(before), int(target_frame) + int(after) + 1))
    missing = [frame_id for frame_id in requested if frame_id not in frame_to_index]
    if missing and not allow_clipped:
        raise ValueError(
            f"Missing qpos frames around target={target_frame}: "
            f"missing={missing[:8]} total={len(missing)} available={frame_ids[:1]}..{frame_ids[-1:]}"
        )
    exported = [frame_id for frame_id in requested if frame_id in frame_to_index]
    if not exported:
        raise ValueError(f"No qpos frames available for target={target_frame}")
    indices = [frame_to_index[frame_id] for frame_id in exported]
    return indices, requested, exported, "clipped" if missing else "ok"


def _write_sample_qpos(
    *,
    source_npz: Path,
    source_log: Path,
    destination_dir: Path,
    sample_id: str,
    robot: str,
    sequence: str,
    frame_id: int,
    window_before: int,
    window_after: int,
    allow_clipped: bool,
) -> dict[str, Any]:
    frame_ids = _qpos_frame_ids_from_log(source_log)
    if not frame_ids:
        raise ValueError(f"No frame_ids found in qpos log: {source_log}")
    indices, requested, exported, status = _requested_window(
        frame_ids,
        frame_id,
        window_before,
        window_after,
        allow_clipped,
    )

    payload: dict[str, np.ndarray] = {}
    with np.load(str(source_npz), allow_pickle=True) as src:
        for key in src.files:
            text = str(key)
            if text.endswith("_root_pose") or text.endswith("_error"):
                continue
            arr = np.asarray(src[key], dtype=np.float32)
            if arr.ndim < 2:
                continue
            payload[text] = arr[np.asarray(indices, dtype=np.int32)].copy()
    if not payload:
        raise KeyError(f"No sample qpos arrays found in {source_npz}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(destination_dir / QPOS_FILE), **payload)
    log_lines = [
        "QPOS summary",
        "=" * 60,
        "dataset=arctic",
        f"sample_id={sample_id}",
        f"combo={robot}",
        f"sequence={sequence}",
        f"target_frame={int(frame_id)}",
        f"window_before={int(window_before)}",
        f"window_after={int(window_after)}",
        f"requested_start={requested[0]}",
        f"requested_end={requested[-1]}",
        f"exported_start={exported[0]}",
        f"exported_end={exported[-1]}",
        f"exported_frame_count={len(exported)}",
        f"status={status}",
        "qpos_keys=" + ";".join(sorted(payload)),
        f"source_qpos={source_npz}",
        f"source_qpos_log={source_log}",
        "frame_ids=" + ",".join(str(int(frame_id)) for frame_id in exported),
    ]
    for key, arr in sorted(payload.items()):
        log_lines.append(f"{key}: shape={tuple(int(dim) for dim in arr.shape)}")
    (destination_dir / QPOS_LOG).write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return {
        "requested_start": requested[0],
        "requested_end": requested[-1],
        "exported_start": exported[0],
        "exported_end": exported[-1],
        "exported_frame_count": len(exported),
        "qpos_status": status,
        "qpos_keys": ";".join(sorted(payload)),
    }


def _metadata_sample(sample_id: str, source_dataset: str, target_robot: str) -> dict[str, Any]:
    slash = "/"
    return {
        "id": sample_id,
        "qpos_dir": f"qpos{slash}{sample_id}",
        "source_dataset": source_dataset,
        "qpos_files": [
            f"qpos{slash}{sample_id}{slash}{QPOS_FILE}",
            f"qpos{slash}{sample_id}{slash}{QPOS_LOG}",
        ],
        "object_mask_available": True,
        "image_paths": {
            "originals": f"originals{slash}{sample_id}.jpg",
            "pseudo_gt": f"pseudo_gt{slash}{sample_id}.jpg",
            "robot_masks": f"robot_masks{slash}{sample_id}.png",
            "hand_masks": f"hand_masks{slash}{sample_id}.png",
            "object_masks": f"object_masks{slash}{sample_id}.png",
        },
        "target_robot": target_robot,
    }


def _write_metadata(path: Path, samples: list[dict[str, Any]]) -> None:
    target_counts = Counter(str(sample.get("target_robot", "")) for sample in samples)
    source_counts = Counter(str(sample.get("source_dataset", "")) for sample in samples)
    metadata = {
        "dataset": "Hand",
        "target_robot_counts": dict(sorted(target_counts.items())),
        "source_dataset_counts": dict(sorted(source_counts.items())),
        "qpos_layout": f"qpos/<id>/{QPOS_FILE} and qpos/<id>/{QPOS_LOG}",
        "categories": CATEGORIES,
        "samples": samples,
        "id_format": "zero-padded package-local sample id",
        "sample_count": len(samples),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")


def package(args: argparse.Namespace) -> None:
    server_root = args.server_root.expanduser().resolve()
    upload_root = args.upload_root.expanduser().resolve()
    hand_dir = _hand_dir(upload_root)
    if args.clean_hand and hand_dir.exists():
        shutil.rmtree(hand_dir)
    hand_dir.mkdir(parents=True, exist_ok=True)
    for child in ["originals", "pseudo_gt", "hand_masks", "object_masks", "robot_masks", "qpos", "metadata"]:
        (hand_dir / child).mkdir(parents=True, exist_ok=True)

    review_csv = args.review_csv or _default_review_csv(server_root)
    review_rows = [row for row in _read_csv(review_csv) if _is_review_ok(row.get("review_ok", ""))]
    if not review_rows:
        raise ValueError(f"No review_ok rows found in {review_csv}")
    if args.only_complete:
        review_rows = [row for row in review_rows if str(row.get("complete", "")).strip() in {"1", "true", "True"}]
    if not review_rows:
        raise ValueError("No review_ok complete rows remain after filtering")

    assets = _load_frame_assets(
        server_root=server_root,
        selected_manifest=args.selected_manifest,
        preprocess_manifest=args.preprocess_manifest,
    )
    keep_ranges = _load_reject_ranges(args.keep_ranges_csv)
    reject_ranges = _load_reject_ranges(args.reject_ranges_csv)
    existing = _load_existing_metadata(args.existing_metadata)
    existing_samples = list(existing.get("samples", [])) if existing else []
    next_id = args.start_id if args.start_id is not None else _max_sample_id(existing_samples) + 1

    metadata_samples = list(existing_samples)
    provenance_rows: list[dict[str, Any]] = []
    sample_index = next_id
    for row in review_rows:
        robot = str(row["robot"]).strip()
        sequence = str(row["sequence"]).strip()
        seq_assets = assets.get(sequence)
        if not seq_assets:
            raise KeyError(f"No frame assets for sequence={sequence}")
        qpos_npz = Path(row.get("qpos_npz") or server_root / "qpos" / robot / sequence / QPOS_FILE)
        qpos_log = Path(row.get("qpos_log") or server_root / "qpos" / robot / sequence / QPOS_LOG)
        for frame_id, frame in sorted(seq_assets.items()):
            if keep_ranges and not _is_rejected_frame(keep_ranges, robot, sequence, frame_id)[0]:
                provenance_rows.append(
                    {
                        "id": "",
                        "source_dataset": args.source_dataset,
                        "target_robot": robot,
                        "sequence": sequence,
                        "frame_id_zero_based": frame_id,
                        "frame_name": frame.frame_name,
                        "source_original": str(frame.original),
                        "source_hand_mask": str(frame.hand_mask),
                        "source_object_mask": str(frame.object_mask),
                        "source_pseudo_gt": str(server_root / "pseudo_gt" / robot / sequence / f"{frame.frame_name}.jpg"),
                        "source_robot_mask": str(server_root / "robot_masks" / robot / sequence / f"{frame.frame_name}.png"),
                        "source_sequence_qpos": str(qpos_npz),
                        "source_sequence_qpos_log": str(qpos_log),
                        "skipped": "1",
                        "skip_reason": "outside_keep_ranges",
                    }
                )
                continue
            rejected, reject_reason = _is_rejected_frame(reject_ranges, robot, sequence, frame_id)
            if rejected:
                provenance_rows.append(
                    {
                        "id": "",
                        "source_dataset": args.source_dataset,
                        "target_robot": robot,
                        "sequence": sequence,
                        "frame_id_zero_based": frame_id,
                        "frame_name": frame.frame_name,
                        "source_original": str(frame.original),
                        "source_hand_mask": str(frame.hand_mask),
                        "source_object_mask": str(frame.object_mask),
                        "source_pseudo_gt": str(server_root / "pseudo_gt" / robot / sequence / f"{frame.frame_name}.jpg"),
                        "source_robot_mask": str(server_root / "robot_masks" / robot / sequence / f"{frame.frame_name}.png"),
                        "source_sequence_qpos": str(qpos_npz),
                        "source_sequence_qpos_log": str(qpos_log),
                        "skipped": "1",
                        "skip_reason": reject_reason or "manual_reject_range",
                    }
                )
                continue
            sample_id = f"{sample_index:0{int(args.id_width)}d}"
            sample_index += 1
            pseudo_src = server_root / "pseudo_gt" / robot / sequence / f"{frame.frame_name}.jpg"
            robot_mask_src = server_root / "robot_masks" / robot / sequence / f"{frame.frame_name}.png"
            if not pseudo_src.is_file():
                raise FileNotFoundError(pseudo_src)
            if not robot_mask_src.is_file():
                raise FileNotFoundError(robot_mask_src)
            for label, src, dst in [
                ("originals", frame.original, hand_dir / "originals" / f"{sample_id}.jpg"),
                ("pseudo_gt", pseudo_src, hand_dir / "pseudo_gt" / f"{sample_id}.jpg"),
                ("hand_masks", frame.hand_mask, hand_dir / "hand_masks" / f"{sample_id}.png"),
                ("object_masks", frame.object_mask, hand_dir / "object_masks" / f"{sample_id}.png"),
                ("robot_masks", robot_mask_src, hand_dir / "robot_masks" / f"{sample_id}.png"),
            ]:
                _copy_or_link(src, dst, args.copy_mode)
            qpos_info = _write_sample_qpos(
                source_npz=qpos_npz,
                source_log=qpos_log,
                destination_dir=hand_dir / "qpos" / sample_id,
                sample_id=sample_id,
                robot=robot,
                sequence=sequence,
                frame_id=frame_id,
                window_before=args.window_before,
                window_after=args.window_after,
                allow_clipped=args.allow_clipped,
            )
            metadata_samples.append(_metadata_sample(sample_id, args.source_dataset, robot))
            provenance_rows.append(
                {
                    "id": sample_id,
                    "source_dataset": args.source_dataset,
                    "target_robot": robot,
                    "sequence": sequence,
                    "frame_id_zero_based": frame_id,
                    "frame_name": frame.frame_name,
                    "source_original": str(frame.original),
                    "source_hand_mask": str(frame.hand_mask),
                    "source_object_mask": str(frame.object_mask),
                    "source_pseudo_gt": str(pseudo_src),
                    "source_robot_mask": str(robot_mask_src),
                    "source_sequence_qpos": str(qpos_npz),
                    "source_sequence_qpos_log": str(qpos_log),
                    "skipped": "0",
                    "skip_reason": "",
                    **qpos_info,
                }
            )

    _write_metadata(upload_root / METADATA_REL, metadata_samples)
    _write_csv(hand_dir / "metadata" / PROVENANCE_NAME, provenance_rows)
    print(f"hand_dir={hand_dir}")
    print(f"new_samples={len(provenance_rows)} total_metadata_samples={len(metadata_samples)}")
    print(f"metadata={upload_root / METADATA_REL}")
    print(f"provenance={hand_dir / 'metadata' / PROVENANCE_NAME}")


def verify(args: argparse.Namespace) -> None:
    upload_root = args.upload_root.expanduser().resolve()
    hand_dir = _hand_dir(upload_root)
    metadata_path = upload_root / METADATA_REL
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    samples = list(metadata.get("samples", []))
    if args.only_provenance:
        provenance = hand_dir / "metadata" / PROVENANCE_NAME
        ids = {row["id"] for row in _read_csv(provenance)} if provenance.is_file() else set()
        samples = [sample for sample in samples if str(sample.get("id", "")) in ids]
    missing: list[str] = []
    bad_qpos: list[str] = []
    for sample in samples:
        sid = str(sample["id"])
        paths = [
            hand_dir / "originals" / f"{sid}.jpg",
            hand_dir / "pseudo_gt" / f"{sid}.jpg",
            hand_dir / "hand_masks" / f"{sid}.png",
            hand_dir / "object_masks" / f"{sid}.png",
            hand_dir / "robot_masks" / f"{sid}.png",
            hand_dir / "qpos" / sid / QPOS_FILE,
            hand_dir / "qpos" / sid / QPOS_LOG,
        ]
        for path in paths:
            if not path.is_file() or path.stat().st_size <= 0:
                missing.append(str(path))
        qpos_path = hand_dir / "qpos" / sid / QPOS_FILE
        if qpos_path.is_file():
            with np.load(str(qpos_path), allow_pickle=True) as payload:
                arrays = [np.asarray(payload[key]) for key in payload.files]
                if not arrays or any(arr.ndim < 2 or arr.shape[0] == 0 for arr in arrays):
                    bad_qpos.append(str(qpos_path))
    print(f"samples_checked={len(samples)}")
    print(f"missing={len(missing)} bad_qpos={len(bad_qpos)}")
    for item in missing[:20]:
        print(f"MISSING {item}")
    for item in bad_qpos[:20]:
        print(f"BAD_QPOS {item}")
    if missing or bad_qpos:
        raise SystemExit(1)


def _load_token(token_file: Path | None, token: str | None) -> str | None:
    if token:
        return token
    if token_file is not None and token_file.is_file():
        for line in token_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("HF_TOKEN")


def upload(args: argparse.Namespace) -> None:
    token = _load_token(args.token_file, args.token)
    if not token:
        raise ValueError("Missing HF token. Pass --token-file or set HF_TOKEN.")
    upload_root = args.upload_root.expanduser().resolve()
    ignore_patterns = ["_work/**", "logs/**", "*.tmp", "*.log", ".hf_upload_token.env"]
    print("Equivalent CLI:")
    print(
        "hf upload-large-folder "
        f"{args.repo_id} {upload_root} --type dataset --private "
        f"--num-workers {args.num_workers}"
    )
    if args.dry_run:
        return
    from huggingface_hub import HfApi

    HfApi(token=token).upload_large_folder(
        repo_id=args.repo_id,
        folder_path=str(upload_root),
        repo_type="dataset",
        private=True,
        ignore_patterns=ignore_patterns,
        num_workers=args.num_workers,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["make-review", "package", "verify", "upload"])
    parser.add_argument("--server-root", type=Path, default=DEFAULT_SERVER_ROOT)
    parser.add_argument("--selected-manifest", type=Path, default=None)
    parser.add_argument("--preprocess-manifest", type=Path, default=None)
    parser.add_argument("--review-csv", type=Path, default=None)
    parser.add_argument(
        "--keep-ranges-csv",
        type=Path,
        default=None,
        help="Optional CSV with robot,sequence,start_frame,end_frame rows. If set, package only frames inside these ranges.",
    )
    parser.add_argument(
        "--reject-ranges-csv",
        type=Path,
        default=None,
        help="Optional CSV with robot,sequence,start_frame,end_frame rows to skip during packaging.",
    )
    parser.add_argument("--robots", default=None, help="Comma-separated robot folders. Defaults to pseudo_gt dirs.")
    parser.add_argument("--upload-root", type=Path, default=DEFAULT_UPLOAD_ROOT)
    parser.add_argument("--source-dataset", default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--existing-metadata", type=Path, default=None)
    parser.add_argument("--start-id", type=int, default=None)
    parser.add_argument("--id-width", type=int, default=6)
    parser.add_argument("--window-before", type=int, default=20)
    parser.add_argument("--window-after", type=int, default=20)
    parser.add_argument("--allow-clipped", action="store_true")
    parser.add_argument("--copy-mode", choices=["copy", "symlink", "hardlink"], default="copy")
    parser.add_argument("--clean-hand", action="store_true")
    parser.add_argument("--only-complete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only-provenance", action="store_true", help="Verify only locally packaged new samples.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--token-file", type=Path, default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "make-review":
        make_review(args)
    elif args.stage == "package":
        package(args)
    elif args.stage == "verify":
        verify(args)
    elif args.stage == "upload":
        upload(args)
    else:
        raise AssertionError(args.stage)


if __name__ == "__main__":
    main()
