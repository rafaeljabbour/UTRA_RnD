"""Main capture, track, display and record loop."""

from __future__ import annotations

import contextlib
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from utra_tracker.config import Config
from utra_tracker.model import load_model, resolve_device
from utra_tracker.records import CsvWriter, Detection, result_to_detections, write_run_json
from utra_tracker.sources import open_capture, parse_source

WINDOW = "utra-track"
QUIT_KEYS = {ord("q"), ord("Q"), 27}


@dataclass
class RunStats:
    frames: int = 0
    detections: int = 0
    track_ids: set[int] = field(default_factory=set)
    started: float = field(default_factory=time.perf_counter)
    stop_reason: str = "end_of_stream"

    def summary(self) -> dict[str, Any]:
        elapsed = time.perf_counter() - self.started
        return {
            "frames": self.frames,
            "detections": self.detections,
            "unique_tracks": len(self.track_ids),
            "track_ids": sorted(self.track_ids),
            "elapsed_s": round(elapsed, 2),
            "fps": round(self.frames / elapsed, 2) if elapsed > 0 else 0.0,
            "stop_reason": self.stop_reason,
        }


def make_run_dir(base: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base) / stamp
    suffix = 1
    while run_dir.exists():
        run_dir = Path(base) / f"{stamp}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def run(cfg: Config, max_frames: int | None = None) -> Path:
    """Track robots from the configured source until it ends, the user quits or Ctrl+C."""
    import cv2

    source = parse_source(cfg.source)
    model = load_model(cfg)
    device = resolve_device(cfg.detect.device)
    cap = open_capture(source)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    run_dir = make_run_dir(cfg.output.dir)
    print(f"Source: {source.kind} {source.value}  device: {device}  output: {run_dir}")

    stats = RunStats()
    csv_writer = CsvWriter(run_dir / "detections.csv")
    video_writer = None
    show = cfg.display and _display_available()
    smoothed_fps = 0.0
    try:
        while max_frames is None or stats.frames < max_frames:
            tick = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            result = model.track(
                frame,
                persist=True,
                tracker=cfg.tracker,
                conf=cfg.detect.conf,
                iou=cfg.detect.iou,
                imgsz=cfg.detect.imgsz,
                device=device,
                verbose=False,
            )[0]
            time_s = _frame_time(stats, source.is_live, video_fps)
            detections = result_to_detections(result, stats.frames, time_s)
            csv_writer.write(detections)
            stats.frames += 1
            stats.detections += len(detections)
            stats.track_ids.update(d.track_id for d in detections if d.track_id >= 0)

            dt = time.perf_counter() - tick
            smoothed_fps = 1 / dt if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 / dt
            if not (show or cfg.output.save_video):
                continue
            annotated = annotate(frame, detections, smoothed_fps)
            # Video stays off by default because students may be on camera and faces must never be stored.
            if cfg.output.save_video:
                if video_writer is None:
                    video_writer = _open_writer(run_dir / "annotated.mp4", annotated, video_fps)
                video_writer.write(annotated)
            if show:
                key = _show(annotated)
                if key is None:
                    show = False
                elif key in QUIT_KEYS:
                    stats.stop_reason = "user_quit"
                    break
    except KeyboardInterrupt:
        stats.stop_reason = "interrupted"
        print("\nStopping.")
    finally:
        cap.release()
        csv_writer.close()
        if video_writer is not None:
            video_writer.release()
        if cfg.display:
            _close_windows()
        if max_frames is not None and stats.frames >= max_frames:
            stats.stop_reason = "max_frames"
        summary = stats.summary()
        write_run_json(run_dir / "run.json", cfg.to_dict(), summary)
    print(
        f"Done: {summary['frames']} frames, {summary['detections']} detections, "
        f"{summary['unique_tracks']} tracks, {summary['fps']} fps. Saved to {run_dir}"
    )
    return run_dir


def annotate(frame: Any, detections: list[Detection], fps: float) -> Any:
    """Draw boxes, track IDs and an FPS counter on a copy of the frame."""
    import cv2

    out = frame.copy()
    scale = max(out.shape[0] / 720, 0.6)
    thick = max(int(2 * scale), 1)
    for det in detections:
        color = _track_color(det.track_id)
        p1, p2 = (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2))
        cv2.rectangle(out, p1, p2, color, thick)
        label = f"id {det.track_id} {det.conf:.2f}" if det.track_id >= 0 else f"{det.conf:.2f}"
        _label(out, label, (p1[0], max(p1[1] - 6, 16)), color, scale)
    _label(out, f"FPS {fps:.1f}", (10, int(30 * scale)), (40, 40, 40), scale)
    return out


def _frame_time(stats: RunStats, live: bool, video_fps: float) -> float:
    if not live and video_fps > 0:
        return stats.frames / video_fps
    return time.perf_counter() - stats.started


def _display_available() -> bool:
    """Qt aborts the process without a display, so check before opening a window."""
    has_display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    if sys.platform.startswith("linux") and not has_display:
        print("No display found, running headless. Use --no-show to silence this.")
        return False
    return True


def _show(frame: Any) -> int | None:
    """Show a frame and return the pressed key, or None if the window failed."""
    import cv2

    try:
        cv2.imshow(WINDOW, frame)
        return cv2.waitKey(1) & 0xFF
    except cv2.error:
        print("Could not open a window, continuing headless.")
        return None


def _close_windows() -> None:
    import cv2

    with contextlib.suppress(cv2.error):
        cv2.destroyAllWindows()


def _open_writer(path: Path, frame: Any, fps: float) -> Any:
    import cv2

    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps if fps > 0 else 30.0, (width, height))


def _label(img: Any, text: str, org: tuple[int, int], color: tuple[int, int, int], scale: float) -> None:
    import cv2

    font, size, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, max(int(2 * scale), 1)
    (w, h), base = cv2.getTextSize(text, font, size, thick)
    x, y = org
    cv2.rectangle(img, (x, y - h - base), (x + w + 4, y + base), color, -1)
    cv2.putText(img, text, (x + 2, y), font, size, (255, 255, 255), thick, cv2.LINE_AA)


def _track_color(track_id: int) -> tuple[int, int, int]:
    if track_id < 0:
        return (128, 128, 128)
    palette = ((0, 90, 230), (0, 170, 60), (200, 60, 0), (160, 0, 160), (0, 140, 200), (30, 30, 200))
    return palette[track_id % len(palette)]
