# Sequence Review Tools

Small server-side tools for reviewing rendered hand/robot sequence outputs in a
browser, marking good frame ranges, marking demo-quality ranges, and exporting a
filtered copy of the dataset.

The scripts were first used for ARCTIC, but the review UI only assumes this
basic layout:

```text
SERVER_ROOT/
  pseudo_gt/<robot_or_urdf>/<sequence>/frame_000000.jpg
  robot_masks/<robot_or_urdf>/<sequence>/frame_000000.png        # optional for export
  robot_overlay_rgba/<robot_or_urdf>/<sequence>/frame_000000.png # optional for export
  originals/<sequence>/frame_000000.jpg                         # optional for export
  hand_masks/<sequence>/frame_000000.png                        # optional for export
  object_masks/<sequence>/frame_000000.png                      # optional for export
  bg/<sequence>/frame_000000.jpg                                # optional for export
  qpos/<robot_or_urdf>/<sequence>/retargeted_pose_bimanual.npz  # optional for export
  qpos/<robot_or_urdf>/<sequence>/retargeting_log_bimanual.txt  # optional for export
```

For HOI4D, HOCap, or another dataset, create or symlink your rendered outputs
into the same `pseudo_gt/<robot>/<sequence>/frame_*.jpg` structure before using
the review UI.

## Install

Use an existing Python environment on the server:

```bash
python -m pip install -r requirements.txt
```

If OpenCV is already installed in your rendering environment, the scripts can
usually run without installing anything else except `numpy`.

## 1. Set Paths

```bash
export ROOT=/path/to/server_outputs
export REVIEW_ROOT="$ROOT/_work/sequence_review"
export KEEP_CSV="$ROOT/manifests/keep_ranges.csv"
export DEMO_CSV="$ROOT/manifests/demo_ranges.csv"
export REJECT_CSV="$ROOT/manifests/reject_ranges.csv"
export PY=python
```

## 2. Build The Review Manifest

The browser player reads JPG frames directly. `--manifest-only` is usually best
on headless servers because it avoids slow MP4 encoding and browser codec
problems.

```bash
"$PY" -u scripts/make_arctic_sequence_review_videos.py \
  --server-root "$ROOT" \
  --output-root "$REVIEW_ROOT" \
  --manifest-only
```

Optional filters:

```bash
# Only index one or more robot/URDF folders.
--robots ability,shadow,allegro

# Quick smoke test on the first few sequences.
--limit-sequences 5
```

## 3. Start The Review Web UI

```bash
"$PY" -u scripts/serve_arctic_sequence_review.py \
  --manifest "$REVIEW_ROOT/review_video_manifest.csv" \
  --keep-csv "$KEEP_CSV" \
  --demo-csv "$DEMO_CSV" \
  --reject-csv "$REJECT_CSV" \
  --host 0.0.0.0 \
  --port 8765
```

On a headless server, forward port `8765` with VSCode Remote Ports, SSH, or your
cluster notebook proxy. Open the forwarded address in your local browser.

If the port is already occupied:

```bash
pkill -f 'serve_arctic_sequence_review.py' || true
```

If `pkill` is unavailable, use a different port such as `8766`.

## 4. Review Workflow

The UI has a left sidebar with robot/URDF filtering and sequence search. The
main panel shows a JPG frame player and a draggable timeline.

Primary actions:

- `KEEP whole sequence + next`: write the whole sequence to `KEEP_CSV`, then go to the next sequence.
- `save KEEP selected range + next`: write only the selected range to `KEEP_CSV`, then go to the next sequence.
- `save DEMO selected range + next`: write the selected range to both `KEEP_CSV` and `DEMO_CSV`, then go to the next sequence.
- `KEEP whole + selected DEMO + next`: write the whole sequence to `KEEP_CSV`, write the selected range to `DEMO_CSV`, mirror that demo range into `KEEP_CSV`, then go to the next sequence.
- `save REJECT bad range`: write a bad local range to `REJECT_CSV`. This stays on the current sequence.
- `REJECT whole sequence + next`: write the whole sequence to `REJECT_CSV`, then go to the next sequence.

The server validates every save against the review manifest:

- `robot` and `sequence` must exist.
- `start_frame` and `end_frame` must be valid integers.
- ranges are sorted and clipped to the sequence frame span.
- demo ranges are mirrored into keep ranges, so demo frames are always retained.

## 5. Check Saved Review Results

This command is read-only:

```bash
"$PY" -u scripts/check_review_ranges.py \
  --manifest "$REVIEW_ROOT/review_video_manifest.csv" \
  --keep-csv "$KEEP_CSV" \
  --demo-csv "$DEMO_CSV" \
  --reject-csv "$REJECT_CSV"
```

Healthy output should end with:

```text
DEMO exact ranges not mirrored in KEEP: 0
errors: 0
```

## 6. Export A Filtered Dataset Tree

This does not modify the original server outputs.

```bash
export FILTERED_ROOT=/path/to/server_outputs_filtered

"$PY" -u scripts/filter_arctic_by_review_ranges.py \
  --server-root "$ROOT" \
  --output-root "$FILTERED_ROOT" \
  --keep-ranges-csv "$KEEP_CSV" \
  --demo-ranges-csv "$DEMO_CSV" \
  --reject-ranges-csv "$REJECT_CSV" \
  --copy-mode hardlink \
  --slice-qpos-to-kept-frames \
  --clean-output
```

Copy modes:

- `hardlink`: fast and storage-saving on the same filesystem.
- `copy`: safest across filesystems.
- `symlink`: useful for inspection, but less portable.

Use `--dry-run` first if you want to inspect what would be exported.

## 7. Optional Hugging Face Packaging

`scripts/arctic_hf_full_review_package.py` is project-specific glue for
packaging reviewed ARCTIC outputs into a HandEdit-style Hugging Face dataset
layout. Treat it as an example if your target dataset has a different schema.

Typical stages:

```bash
"$PY" -u scripts/arctic_hf_full_review_package.py --stage make-review --server-root "$ROOT"
"$PY" -u scripts/arctic_hf_full_review_package.py --stage package --server-root "$ROOT" --upload-root /path/to/staging
"$PY" -u scripts/arctic_hf_full_review_package.py --stage verify --upload-root /path/to/staging
```

Never commit Hugging Face or GitHub tokens into this repository.

## Notes For New Datasets

For HOI4D/HOCap-style migration:

1. Render or blend frames into `pseudo_gt/<robot>/<sequence>/frame_*.jpg`.
2. Put optional masks/background/qpos into the matching directories listed at the top.
3. Run the manifest builder with `--manifest-only`.
4. Review in the web UI.
5. Export with `filter_arctic_by_review_ranges.py`.
6. Adapt the optional HF packaging script only after the filtered tree looks correct.

The scripts intentionally keep filtering as CSV state. That makes it easy to
audit, back up, merge, and rerun exports without touching the original rendered
data.
