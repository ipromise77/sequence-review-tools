"""Serve a tiny ARCTIC sequence video review UI.

Run this on the headless server, forward the port through VSCode or SSH, and
use a local browser to mark keep/demo/reject frame ranges. The generated CSVs
can be consumed by ``filter_arctic_by_review_ranges.py`` and by the HF
packaging script.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import re
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


FIELDS = ["robot", "sequence", "start_frame", "end_frame", "purpose", "reason"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _append_range(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})


def _frame_id(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else -1


def make_handler(manifest: Path, keep_csv: Path, demo_csv: Path, reject_csv: Path):
    rows = _read_csv(manifest)
    row_by_key = {(row["robot"], row["sequence"]): row for row in rows}
    video_by_key = {(row["robot"], row["sequence"]): Path(row["video"]) for row in rows}
    frame_paths_by_key = {
        (row["robot"], row["sequence"]): sorted(Path(row["pseudo_gt_dir"]).glob("frame_*.jpg"), key=_frame_id)
        for row in rows
        if row.get("pseudo_gt_dir")
    }

    class Handler(BaseHTTPRequestHandler):
        def _route_path(self) -> str:
            path = urlparse(self.path).path.rstrip("/") or "/"
            for route in ("/api/rows", "/api/range", "/frame", "/video"):
                if path == route or path.endswith(route):
                    return route
            return path

        def _send(self, status: int, body: bytes, content_type: str = "text/html; charset=utf-8") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, content_type: str) -> None:
            size = path.stat().st_size
            range_header = self.headers.get("Range", "")
            if range_header.startswith("bytes="):
                raw = range_header.split("=", 1)[1].split(",", 1)[0]
                start_text, _, end_text = raw.partition("-")
                start = int(start_text) if start_text else 0
                end = int(end_text) if end_text else size - 1
                start = max(0, min(start, size - 1))
                end = max(start, min(end, size - 1))
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", content_type)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(length))
                self.end_headers()
                with path.open("rb") as handle:
                    handle.seek(start)
                    self.wfile.write(handle.read(length))
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with path.open("rb") as handle:
                shutil.copyfileobj(handle, self.wfile)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            route = self._route_path()
            if route == "/api/rows":
                body = json.dumps(
                    {
                        "rows": rows,
                        "keeps": _read_csv(keep_csv),
                        "demos": _read_csv(demo_csv),
                        "rejects": _read_csv(reject_csv),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
                return
            if route == "/video":
                query = parse_qs(parsed.query)
                robot = query.get("robot", [""])[0]
                sequence = query.get("sequence", [""])[0]
                video = video_by_key.get((robot, sequence))
                if video is None or not video.is_file():
                    self._send(404, b"missing video", "text/plain; charset=utf-8")
                    return
                self._send_file(video, mimetypes.guess_type(video.name)[0] or "video/mp4")
                return
            if route == "/frame":
                query = parse_qs(parsed.query)
                robot = query.get("robot", [""])[0]
                sequence = query.get("sequence", [""])[0]
                try:
                    index = int(query.get("idx", ["0"])[0])
                except ValueError:
                    index = 0
                frames = frame_paths_by_key.get((robot, sequence), [])
                if not frames:
                    self._send(404, b"missing frames", "text/plain; charset=utf-8")
                    return
                index = max(0, min(index, len(frames) - 1))
                frame = frames[index]
                self._send_file(frame, mimetypes.guess_type(frame.name)[0] or "image/jpeg")
                return
            self._send(200, self._page().encode("utf-8"))

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if self._route_path() != "/api/range":
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            kind = str(payload.get("kind", "")).strip().lower()
            target = {"keep": keep_csv, "demo": demo_csv, "reject": reject_csv}.get(kind)
            if target is None and kind != "keep_all_demo":
                self._send(400, b"kind must be keep, demo, reject, or keep_all_demo", "text/plain; charset=utf-8")
                return
            row = {field: str(payload.get(field, "")).strip() for field in FIELDS}
            if not row["robot"] or not row["sequence"] or not row["start_frame"] or not row["end_frame"]:
                self._send(400, b"robot, sequence, start_frame, end_frame are required", "text/plain; charset=utf-8")
                return
            manifest_row = row_by_key.get((row["robot"], row["sequence"]))
            if manifest_row is None:
                self._send(400, b"robot/sequence is not present in manifest", "text/plain; charset=utf-8")
                return
            try:
                start = int(row["start_frame"])
                end = int(row["end_frame"])
                first = int(manifest_row.get("first_frame") or 0)
                last = int(manifest_row.get("last_frame") or first)
            except ValueError:
                self._send(400, b"start_frame/end_frame must be integers", "text/plain; charset=utf-8")
                return
            start, end = sorted((start, end))
            if end < first or start > last:
                self._send(400, b"selected range is outside the sequence frame span", "text/plain; charset=utf-8")
                return
            row["start_frame"] = str(max(first, start))
            row["end_frame"] = str(min(last, end))
            row["purpose"] = kind
            written = []
            if kind == "keep_all_demo":
                keep_all_row = dict(row)
                keep_all_row["start_frame"] = str(first)
                keep_all_row["end_frame"] = str(last)
                keep_all_row["purpose"] = "keep"
                keep_all_row["reason"] = keep_all_row["reason"] or "whole_sequence_keep_with_demo"
                _append_range(keep_csv, keep_all_row)
                written.append(str(keep_csv))

                demo_keep_row = dict(row)
                demo_keep_row["purpose"] = "keep"
                demo_keep_row["reason"] = demo_keep_row["reason"] or "demo_marked_keep"
                if demo_keep_row["start_frame"] != keep_all_row["start_frame"] or demo_keep_row["end_frame"] != keep_all_row["end_frame"]:
                    _append_range(keep_csv, demo_keep_row)

                demo_row = dict(row)
                demo_row["purpose"] = "demo"
                _append_range(demo_csv, demo_row)
                written.append(str(demo_csv))
                body = json.dumps(
                    {"ok": True, "written": written, "keep_all": keep_all_row, "demo": demo_row},
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send(200, body, "application/json")
                return
            if kind == "demo":
                keep_row = dict(row)
                keep_row["purpose"] = "keep"
                keep_row["reason"] = keep_row["reason"] or "demo_marked_keep"
                _append_range(keep_csv, keep_row)
                written.append(str(keep_csv))
            _append_range(target, row)
            written.append(str(target))
            body = json.dumps({"ok": True, "written": written, "row": row}, ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json")

        def _page(self) -> str:
            return f"""<!doctype html>
<meta charset="utf-8">
<title>ARCTIC Sequence Review</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 16px; background: #f6f6f3; color: #111; }}
.layout {{ display: grid; grid-template-columns: 380px 1fr; gap: 16px; }}
.list {{ height: 88vh; overflow: auto; border: 1px solid #ccc; background: white; }}
.item {{ padding: 8px 10px; border-bottom: 1px solid #eee; cursor: pointer; }}
.item:hover, .item.active {{ background: #e8f0ff; }}
.item.keepMarked {{ border-left: 6px solid #35a853; }}
.item.demoMarked {{ border-left: 6px solid #8b5cf6; }}
.item.rejectMarked {{ border-left: 6px solid #d63f31; }}
.item.fullKeep {{ background: #dcf8e2; }}
.item.fullReject {{ background: #ffe0df; color: #6b1410; }}
.item.fullDemo {{ background: #efe5ff; }}
.item.active {{ outline: 2px solid #2b61d1; outline-offset: -2px; }}
video {{ width: 100%; max-height: 68vh; background: #222; }}
.framePlayer {{ background: #111; display: flex; align-items: center; justify-content: center; min-height: 320px; max-height: 68vh; }}
.framePlayer img {{ max-width: 100%; max-height: 68vh; display: block; }}
input, button, textarea {{ font: inherit; margin: 4px; }}
input {{ width: 110px; }}
textarea {{ width: 620px; height: 56px; }}
button.keep {{ background: #d9f5df; }}
button.demo {{ background: #eadcff; }}
button.reject {{ background: #ffd7d7; }}
.small {{ color: #666; font-size: 12px; }}
.row {{ margin: 6px 0; }}
.tools {{ display: flex; align-items: center; flex-wrap: wrap; gap: 4px; }}
.filters {{ display: grid; gap: 6px; margin: 8px 0; }}
.filters select, .filters input {{ width: 100%; box-sizing: border-box; margin: 0; }}
.group {{ padding: 6px 10px; background: #eee; border-bottom: 1px solid #d4d4d4; font-weight: 700; }}
.timeline {{ position: relative; height: 30px; margin: 10px 4px; background: #ddd; border: 1px solid #aaa; cursor: crosshair; user-select: none; }}
.timelineSelection {{ position: absolute; top: 0; bottom: 0; background: rgba(50, 130, 240, 0.42); display: none; }}
.timelinePlayhead {{ position: absolute; top: -3px; bottom: -3px; width: 2px; background: #111; }}
pre {{ background: white; border: 1px solid #ccc; padding: 8px; max-height: 180px; overflow: auto; }}
</style>
<div class="layout">
  <div>
    <h3>Sequences / 序列</h3>
    <div class="small">Keep CSV: {html.escape(str(keep_csv))}</div>
    <div class="small">Demo CSV: {html.escape(str(demo_csv))}</div>
    <div class="small">Reject CSV: {html.escape(str(reject_csv))}</div>
    <div class="filters">
      <label>Robot / URDF
        <select id="robotFilter"></select>
      </label>
      <label>Search sequence / 搜索序列
        <input id="searchBox" placeholder="s01__box..." />
      </label>
      <div class="small" id="listCount"></div>
    </div>
    <div id="list" class="list"></div>
  </div>
  <div>
    <h3 id="title">Pick a sequence</h3>
    <h3>Frame player / 帧序列播放器</h3>
    <div class="framePlayer"><img id="frameImg" alt="current frame"></div>
    <div class="tools">
      <button id="prevFrame">prev / 上一帧</button>
      <button id="playFrames">play / 播放</button>
      <button id="nextFrame">next / 下一帧</button>
      <label>fps / 帧率 <input id="fps" type="number" value="15" min="1" max="60"></label>
      <label>frame / 当前帧 <input id="frameInput" type="number" value="0"></label>
      <span class="small" id="frameStatus"></span>
    </div>
    <div class="small">Drag timeline to select range / 鼠标拖动时间轴选择帧段</div>
    <div id="timeline" class="timeline">
      <div id="timelineSelection" class="timelineSelection"></div>
      <div id="timelinePlayhead" class="timelinePlayhead"></div>
    </div>
    <details>
      <summary>MP4 preview / MP4 预览（如果浏览器支持该编码才可播放）</summary>
      <video id="video" controls></video>
    </details>
    <div class="small" id="meta"></div>
    <h3>Mark Frame Range / 标注帧范围</h3>
    <div class="row">
      <label>start / 起始帧 <input id="start" type="number"></label>
      <label>end / 结束帧 <input id="end" type="number"></label>
      <label>half-window / 半窗口 <input id="half" type="number" value="20"></label>
    </div>
    <div class="row">
      <button id="setStart">set start = current / 起始=当前</button>
      <button id="setEnd">set end = current / 结束=当前</button>
      <button id="centerWindow">current +/- half-window / 当前±半窗口</button>
    </div>
    <div class="row">
      <button class="keep" onclick="saveRange('keep')">save KEEP selected range + next / 保留选中段并下一条</button>
      <button class="demo" onclick="saveRange('demo')">save DEMO selected range + next / 保留选中段并标记demo，然后下一条</button>
      <button class="reject" onclick="saveRange('reject')">save REJECT bad range / 保存坏帧</button>
    </div>
    <div class="row">
      <button class="keep" onclick="saveWholeSequence('keep')">KEEP whole sequence + next / 保留整段并下一条</button>
      <button class="demo" onclick="saveWholeWithSelectedDemo()">KEEP whole + selected DEMO + next / 保留整段并将选中段标记demo，然后下一条</button>
      <button class="reject" onclick="saveWholeSequence('reject')">REJECT whole sequence + next / 不保留整段并下一条</button>
    </div>
    <textarea id="reason" placeholder="reason / note / 原因备注"></textarea>
    <h3>Saved ranges / 已保存范围</h3>
    <pre id="ranges"></pre>
  </div>
</div>
<script>
let rows = [];
let selected = null;
let frameIndex = 0;
let timer = null;
let draggingTimeline = false;
let dragStartFrame = 0;
let rangeData = {{keeps: [], demos: [], rejects: []}};
const framePreload = new Map();
const PRELOAD_AHEAD = 5;
const MAX_PRELOAD = 120;
async function loadRows() {{
  const data = await (await fetch('api/rows')).json();
  rows = data.rows;
  rangeData = {{keeps: data.keeps || [], demos: data.demos || [], rejects: data.rejects || []}};
  populateRobots();
  renderList();
  function fmt(name, arr) {{
    return name + '\\n' + arr.map(
      r => `${{r.robot}},${{r.sequence}},${{r.start_frame}}-${{r.end_frame}} ${{r.reason||''}}`
    ).join('\\n');
  }}
  document.getElementById('ranges').textContent =
    fmt('KEEP', data.keeps) + '\\n\\n' + fmt('DEMO', data.demos) + '\\n\\n' + fmt('REJECT', data.rejects);
  if (!selected && rows.length) selectRowByKey(rows[0].robot, rows[0].sequence);
}}
function populateRobots() {{
  const select = document.getElementById('robotFilter');
  const current = select.value || 'ALL';
  const robots = [...new Set(rows.map(row => row.robot))].sort();
  select.innerHTML = '<option value="ALL">All robots / 全部URDF</option>' +
    robots.map(robot => `<option value="${{robot}}">${{robot}}</option>`).join('');
  select.value = robots.includes(current) ? current : 'ALL';
}}
function visibleRows() {{
  const robot = document.getElementById('robotFilter').value;
  const query = document.getElementById('searchBox').value.trim().toLowerCase();
  return rows.filter(row => {{
    if (robot !== 'ALL' && row.robot !== robot) return false;
    if (query && !`${{row.robot}} ${{row.sequence}}`.toLowerCase().includes(query)) return false;
    return true;
  }});
}}
function sameKey(row, robot, sequence) {{
  return row.robot === robot && row.sequence === sequence;
}}
function coversWholeSequence(range, row) {{
  const start = Number(range.start_frame || range.start || 0);
  const end = Number(range.end_frame || range.end || start);
  return sameKey(range, row.robot, row.sequence) &&
    Math.min(start, end) <= Number(row.first_frame || 0) &&
    Math.max(start, end) >= Number(row.last_frame || 0);
}}
function sequenceStatus(row) {{
  const hasKeep = rangeData.keeps.some(range => sameKey(range, row.robot, row.sequence));
  const hasDemo = rangeData.demos.some(range => sameKey(range, row.robot, row.sequence));
  const hasReject = rangeData.rejects.some(range => sameKey(range, row.robot, row.sequence));
  const fullKeep = rangeData.keeps.some(range => coversWholeSequence(range, row));
  const fullDemo = rangeData.demos.some(range => coversWholeSequence(range, row));
  const fullReject = rangeData.rejects.some(range => coversWholeSequence(range, row));
  return {{hasKeep, hasDemo, hasReject, fullKeep, fullDemo, fullReject}};
}}
function nextVisibleAfterCurrent() {{
  const shown = visibleRows();
  if (!selected || !shown.length) return null;
  const currentIndex = shown.findIndex(row => sameKey(row, selected.robot, selected.sequence));
  if (currentIndex < 0 || currentIndex + 1 >= shown.length) return null;
  return shown[currentIndex + 1];
}}
function renderList() {{
  const list = document.getElementById('list');
  list.innerHTML = '';
  const shown = visibleRows();
  document.getElementById('listCount').textContent = `${{shown.length}} / ${{rows.length}} sequences`;
  let previousRobot = '';
  shown.forEach((row, i) => {{
    if (row.robot !== previousRobot) {{
      previousRobot = row.robot;
      const group = document.createElement('div');
      group.className = 'group';
      group.textContent = `Robot / URDF: ${{row.robot}}`;
      list.appendChild(group);
    }}
    const div = document.createElement('div');
    div.className = 'item';
    const status = sequenceStatus(row);
    if (status.hasKeep) div.classList.add('keepMarked');
    if (status.hasDemo) div.classList.add('demoMarked');
    if (status.hasReject) div.classList.add('rejectMarked');
    if (status.fullKeep) div.classList.add('fullKeep');
    if (status.fullDemo) div.classList.add('fullDemo');
    if (status.fullReject) div.classList.add('fullReject');
    div.textContent = `${{i+1}}. ${{row.robot}} / ${{row.sequence}} (${{row.frame_count}})`;
    if (status.fullReject) div.textContent += '  [REJECT all / 整段剔除]';
    else if (status.fullDemo) div.textContent += '  [DEMO all / 整段demo优选]';
    else if (status.fullKeep && status.hasDemo) div.textContent += '  [KEEP all + DEMO / 整段保留+demo]';
    else if (status.fullKeep) div.textContent += '  [KEEP all / 整段保留]';
    else if (status.hasDemo) div.textContent += '  [DEMO / demo优选]';
    else if (status.hasKeep) div.textContent += '  [KEEP / 已保留]';
    else if (status.hasReject) div.textContent += '  [REJECT marked / 已标坏段]';
    div.onclick = () => selectRowByKey(row.robot, row.sequence);
    if (selected && selected.robot === row.robot && selected.sequence === row.sequence) div.classList.add('active');
    list.appendChild(div);
  }});
}}
function selectRowByKey(robot, sequence) {{
  const row = rows.find(item => item.robot === robot && item.sequence === sequence);
  if (!row) return;
  selected = row;
  stopFrames();
  frameIndex = 0;
  framePreload.clear();
  renderList();
  document.getElementById('title').textContent = `${{selected.robot}} / ${{selected.sequence}}`;
  document.getElementById('meta').textContent = `frames ${{selected.first_frame}}-${{selected.last_frame}}, count=${{selected.frame_count}}`;
  document.getElementById('video').src = `video?robot=${{encodeURIComponent(selected.robot)}}&sequence=${{encodeURIComponent(selected.sequence)}}`;
  showFrame(0);
}}
function frameUrl(i) {{
  if (!selected) return '';
  const robot = encodeURIComponent(selected.robot);
  const sequence = encodeURIComponent(selected.sequence);
  return `frame?robot=${{robot}}&sequence=${{sequence}}&idx=${{i}}`;
}}
function preloadFrames(start, count) {{
  if (!selected) return;
  const maxCount = Number(selected.frame_count || 1);
  for (let i = start; i < Math.min(maxCount, start + count); i++) {{
    const url = frameUrl(i);
    if (framePreload.has(url)) continue;
    const img = new Image();
    img.src = url;
    framePreload.set(url, img);
  }}
  if (framePreload.size > MAX_PRELOAD) {{
    for (const key of framePreload.keys()) {{
      framePreload.delete(key);
      if (framePreload.size <= MAX_PRELOAD) break;
    }}
  }}
}}
function showFrame(i) {{
  if (!selected) return;
  const count = Number(selected.frame_count || 1);
  frameIndex = Math.max(0, Math.min(Number(i) || 0, count - 1));
  document.getElementById('frameImg').src = frameUrl(frameIndex);
  document.getElementById('frameInput').value = currentFrame();
  document.getElementById('frameStatus').textContent = `${{frameIndex + 1}} / ${{count}}`;
  preloadFrames(frameIndex + 1, PRELOAD_AHEAD);
  updateTimeline();
}}
function stepFrame(delta) {{
  if (!selected) return;
  const count = Number(selected.frame_count || 1);
  const next = frameIndex + delta;
  if (next >= count) {{
    stopFrames();
    showFrame(count - 1);
  }} else {{
    showFrame(next);
  }}
}}
function playFrames() {{
  if (timer) {{
    stopFrames();
    return;
  }}
  document.getElementById('playFrames').textContent = 'pause / 暂停';
  const fps = Math.max(1, Number(document.getElementById('fps').value || 15));
  timer = setInterval(() => stepFrame(1), 1000 / fps);
}}
function stopFrames() {{
  if (timer) clearInterval(timer);
  timer = null;
  const button = document.getElementById('playFrames');
  if (button) button.textContent = 'play / 播放';
}}
function currentFrame() {{
  if (!selected) return 0;
  const first = Number(selected.first_frame || 0);
  return Math.round(first + frameIndex);
}}
function frameFromTimelineEvent(event) {{
  if (!selected) return 0;
  const rect = document.getElementById('timeline').getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / Math.max(rect.width, 1)));
  const first = Number(selected.first_frame || 0);
  const count = Number(selected.frame_count || 1);
  return Math.round(first + ratio * (count - 1));
}}
function updateSelectedRange(start, end) {{
  if (!selected) return;
  const focus = end;
  if (end < start) [start, end] = [end, start];
  document.getElementById('start').value = start;
  document.getElementById('end').value = end;
  const first = Number(selected.first_frame || 0);
  showFrame(focus - first);
  updateTimeline();
}}
function updateTimeline() {{
  const selection = document.getElementById('timelineSelection');
  const playhead = document.getElementById('timelinePlayhead');
  if (!selected) {{
    selection.style.display = 'none';
    playhead.style.left = '0%';
    return;
  }}
  const count = Math.max(Number(selected.frame_count || 1), 1);
  const headPct = count <= 1 ? 0 : (frameIndex / (count - 1)) * 100;
  playhead.style.left = `${{headPct}}%`;
  const startRaw = document.getElementById('start').value;
  const endRaw = document.getElementById('end').value;
  if (startRaw === '' || endRaw === '') {{
    selection.style.display = 'none';
    return;
  }}
  const first = Number(selected.first_frame || 0);
  const start = Math.max(0, Math.min(Number(startRaw) - first, count - 1));
  const end = Math.max(0, Math.min(Number(endRaw) - first, count - 1));
  const left = Math.min(start, end) / Math.max(count - 1, 1) * 100;
  const right = Math.max(start, end) / Math.max(count - 1, 1) * 100;
  selection.style.display = 'block';
  selection.style.left = `${{left}}%`;
  selection.style.width = `${{Math.max(0.4, right - left)}}%`;
}}
document.getElementById('setStart').onclick = () => {{ document.getElementById('start').value = currentFrame(); updateTimeline(); }};
document.getElementById('setEnd').onclick = () => {{ document.getElementById('end').value = currentFrame(); updateTimeline(); }};
document.getElementById('prevFrame').onclick = () => stepFrame(-1);
document.getElementById('nextFrame').onclick = () => stepFrame(1);
document.getElementById('playFrames').onclick = playFrames;
document.getElementById('frameInput').onchange = () => {{
  if (!selected) return;
  const first = Number(selected.first_frame || 0);
  showFrame(Number(document.getElementById('frameInput').value || first) - first);
}};
document.getElementById('centerWindow').onclick = () => {{
  const c = currentFrame();
  const h = Number(document.getElementById('half').value || 0);
  document.getElementById('start').value = c - h;
  document.getElementById('end').value = c + h;
  updateTimeline();
}};
document.getElementById('start').onchange = updateTimeline;
document.getElementById('end').onchange = updateTimeline;
document.getElementById('robotFilter').onchange = renderList;
document.getElementById('searchBox').oninput = renderList;
document.getElementById('timeline').onpointerdown = event => {{
  if (!selected) return;
  draggingTimeline = true;
  dragStartFrame = frameFromTimelineEvent(event);
  updateSelectedRange(dragStartFrame, dragStartFrame);
  document.getElementById('timeline').setPointerCapture(event.pointerId);
}};
document.getElementById('timeline').onpointermove = event => {{
  if (!draggingTimeline) return;
  updateSelectedRange(dragStartFrame, frameFromTimelineEvent(event));
}};
document.getElementById('timeline').onpointerup = event => {{
  if (!draggingTimeline) return;
  draggingTimeline = false;
  updateSelectedRange(dragStartFrame, frameFromTimelineEvent(event));
}};
async function saveRange(kind) {{
  if (!selected) return alert('pick sequence first');
  const next = (kind === 'keep' || kind === 'demo') ? nextVisibleAfterCurrent() : null;
  const body = {{
    kind,
    robot: selected.robot,
    sequence: selected.sequence,
    start_frame: document.getElementById('start').value,
    end_frame: document.getElementById('end').value,
    purpose: kind,
    reason: document.getElementById('reason').value,
  }};
  const res = await fetch('api/range', {{method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify(body)}});
  if (!res.ok) return alert(await res.text());
  document.getElementById('reason').value = '';
  await loadRows();
  if (next) selectRowByKey(next.robot, next.sequence);
}}
async function saveWholeWithSelectedDemo() {{
  if (!selected) return alert('pick sequence first');
  const next = nextVisibleAfterCurrent();
  const body = {{
    kind: 'keep_all_demo',
    robot: selected.robot,
    sequence: selected.sequence,
    start_frame: document.getElementById('start').value,
    end_frame: document.getElementById('end').value,
    purpose: 'keep_all_demo',
    reason: document.getElementById('reason').value,
  }};
  const res = await fetch('api/range', {{method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify(body)}});
  if (!res.ok) return alert(await res.text());
  document.getElementById('reason').value = '';
  await loadRows();
  if (next) selectRowByKey(next.robot, next.sequence);
}}
async function saveWholeSequence(kind) {{
  if (!selected) return alert('pick sequence first');
  const next = nextVisibleAfterCurrent();
  const start = Number(selected.first_frame || 0);
  const end = Number(selected.last_frame || start);
  document.getElementById('start').value = start;
  document.getElementById('end').value = end;
  updateTimeline();
  const body = {{
    kind,
    robot: selected.robot,
    sequence: selected.sequence,
    start_frame: start,
    end_frame: end,
    purpose: kind,
    reason: document.getElementById('reason').value || (kind === 'keep' ? 'whole_sequence_keep' : 'whole_sequence_reject'),
  }};
  const res = await fetch('api/range', {{method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify(body)}});
  if (!res.ok) return alert(await res.text());
  document.getElementById('reason').value = '';
  await loadRows();
  if (next) selectRowByKey(next.robot, next.sequence);
}}
loadRows();
</script>"""

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--keep-csv", type=Path, default=None)
    parser.add_argument("--demo-csv", type=Path, default=None)
    parser.add_argument("--reject-csv", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    keep_csv = args.keep_csv or (args.manifest.parent / "keep_ranges.csv")
    demo_csv = args.demo_csv or (args.manifest.parent / "demo_ranges.csv")
    reject_csv = args.reject_csv or (args.manifest.parent / "reject_ranges.csv")
    server = ThreadingHTTPServer((args.host, int(args.port)), make_handler(args.manifest, keep_csv, demo_csv, reject_csv))
    print(f"open http://{args.host}:{args.port}", flush=True)
    print(f"keep_csv={keep_csv}", flush=True)
    print(f"demo_csv={demo_csv}", flush=True)
    print(f"reject_csv={reject_csv}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
