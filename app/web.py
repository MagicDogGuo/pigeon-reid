"""Local Flask monitor: MJPEG stream + visit stats + recent captures."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Generator, Optional

import cv2
from flask import Flask, Response, jsonify, render_template, send_from_directory

from app.config import AppConfig, CONFIG, ROOT_DIR
from app.counter import VisitCounter, VisitStats
from app.detector import Detector
from app.saver import CaptureSaver

RECENT_CAPTURE_LIMIT = 12
JPEG_QUALITY = 80
MJPEG_BOUNDARY = b"frame"


@dataclass
class MonitorSnapshot:
    """Thread-safe view of the latest processed frame and stats."""

    jpeg_bytes: Optional[bytes]
    stats: Optional[VisitStats]
    recent: list[str]
    error: Optional[str]


class MonitorRuntime:
    """Background webcam loop shared by Flask routes."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._jpeg_bytes: Optional[bytes] = None
        self._stats: Optional[VisitStats] = None
        self._recent: Deque[str] = deque(maxlen=RECENT_CAPTURE_LIMIT)
        self._error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="pigeon-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def snapshot(self) -> MonitorSnapshot:
        with self._lock:
            return MonitorSnapshot(
                jpeg_bytes=self._jpeg_bytes,
                stats=self._stats,
                recent=list(self._recent),
                error=self._error,
            )

    def mjpeg_frames(self) -> Generator[bytes, None, None]:
        while not self._stop.is_set():
            snap = self.snapshot()
            if snap.jpeg_bytes is not None:
                yield (
                    b"--" + MJPEG_BOUNDARY + b"\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + snap.jpeg_bytes
                    + b"\r\n"
                )
            time.sleep(0.05)

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._error = message

    def _publish(
        self,
        jpeg_bytes: bytes,
        stats: VisitStats,
        saved_rel: Optional[str],
    ) -> None:
        with self._lock:
            self._jpeg_bytes = jpeg_bytes
            self._stats = stats
            self._error = None
            if saved_rel is not None:
                self._recent.appendleft(saved_rel)

    def _encode_jpeg(self, frame) -> Optional[bytes]:
        ok, buf = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        if not ok:
            return None
        return buf.tobytes()

    def _relative_capture(self, path: Path) -> str:
        return path.relative_to(self._config.captures_dir).as_posix()

    def _run_loop(self) -> None:
        detector = Detector(self._config)
        counter = VisitCounter(self._config)
        saver = CaptureSaver(self._config)
        try:
            detector.open()
        except Exception as exc:  # noqa: BLE001 — surface to UI
            self._set_error(f"camera open failed: {exc}")
            return

        try:
            while not self._stop.is_set():
                detection = detector.read()
                if detection is None:
                    self._set_error("camera read failed")
                    time.sleep(0.2)
                    continue

                stats = counter.update(detection.box_count)
                saved = saver.maybe_save(detection.annotated_frame, stats)
                jpeg = self._encode_jpeg(detection.annotated_frame)
                if jpeg is None:
                    self._set_error("jpeg encode failed")
                    continue

                saved_rel = (
                    self._relative_capture(saved) if saved is not None else None
                )
                self._publish(jpeg, stats, saved_rel)
        except Exception as exc:  # noqa: BLE001 — surface to UI
            self._set_error(f"monitor loop failed: {exc}")
        finally:
            detector.close()


def create_app(
    config: AppConfig | None = None,
    *,
    runtime: MonitorRuntime | None = None,
    start_runtime: bool = True,
) -> Flask:
    """Build Flask app; config and runtime are injectable for tests."""
    cfg = config or CONFIG
    monitor = runtime or MonitorRuntime(cfg)

    app = Flask(
        __name__,
        template_folder=str(ROOT_DIR / "templates"),
        static_folder=str(ROOT_DIR / "static"),
    )
    app.config["PIGEON_CONFIG"] = cfg
    app.config["PIGEON_RUNTIME"] = monitor

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        return Response(
            monitor.mjpeg_frames(),
            mimetype=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY.decode()}",
        )

    @app.route("/api/stats")
    def api_stats():
        snap = monitor.snapshot()
        stats = snap.stats
        payload = {
            "visits_today": 0 if stats is None else stats.visits_today,
            "concurrent_count": 0 if stats is None else stats.concurrent_count,
            "in_visit": False if stats is None else stats.in_visit,
            "local_date": None if stats is None else stats.local_date.isoformat(),
            "recent": snap.recent,
            "error": snap.error,
            "has_frame": snap.jpeg_bytes is not None,
        }
        return jsonify(payload)

    @app.route("/captures/<path:rel_path>")
    def serve_capture(rel_path: str):
        captures_dir = cfg.captures_dir.resolve()
        target = (captures_dir / rel_path).resolve()
        if not str(target).startswith(str(captures_dir)):
            return ("Not found", 404)
        if not target.is_file():
            return ("Not found", 404)
        return send_from_directory(target.parent, target.name)

    if start_runtime:
        monitor.start()

    return app


def main() -> None:
    app = create_app(CONFIG)
    print(
        f"pigeon monitor → http://127.0.0.1:5000 "
        f"model={CONFIG.model_path.name} conf={CONFIG.conf}"
    )
    # threaded=True: MJPEG + /api/stats while the worker holds the camera.
    # use_reloader=False: avoid opening the webcam twice.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
