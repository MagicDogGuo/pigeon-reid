"""Local Flask monitor: MJPEG stream + visit stats + recent captures."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Generator, Optional

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from app.config import (
    MODEL_PRESETS,
    AppConfig,
    CONFIG,
    ROOT_DIR,
    get_model_preset,
)
from app.counter import VisitCounter, VisitStats
from app.detector import Detector, list_camera_indices
from app.saver import CaptureSaver
from app.visit_store import VisitCsvStore

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
    camera_index: int
    model_id: str
    frame_width: Optional[int]
    frame_height: Optional[int]
    switching: bool


class MonitorRuntime:
    """Background webcam loop shared by Flask routes."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._jpeg_bytes: Optional[bytes] = None
        self._stats: Optional[VisitStats] = None
        self._recent: Deque[str] = deque(maxlen=RECENT_CAPTURE_LIMIT)
        self._error: Optional[str] = None
        self._camera_index = config.camera_index
        self._model_id = config.model_id
        self._frame_width: Optional[int] = None
        self._frame_height: Optional[int] = None
        self._switch_to: Optional[int] = None
        self._switch_model_to: Optional[str] = None
        self._switching = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def camera_index(self) -> int:
        with self._lock:
            return self._camera_index

    @property
    def model_id(self) -> str:
        with self._lock:
            return self._model_id

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

    def request_camera(self, camera_index: int) -> int:
        """Ask the worker to switch cameras; returns the requested index."""
        if camera_index < 0:
            raise ValueError(f"camera_index must be >= 0, got {camera_index}")
        with self._lock:
            if camera_index == self._camera_index and not self._switching:
                return camera_index
            self._switch_to = camera_index
            self._switching = True
            self._error = f"switching to camera {camera_index}…"
        return camera_index

    def request_model(self, model_id: str) -> str:
        """Ask the worker to switch YOLO model; returns the requested id."""
        preset = get_model_preset(model_id)
        with self._lock:
            if model_id == self._model_id and not self._switching:
                return model_id
            self._switch_model_to = preset.id
            self._switching = True
            self._error = f"switching to model {preset.label}…"
        return preset.id

    def list_cameras(self) -> list[int]:
        return list_camera_indices(self._config.camera_probe_max)

    def list_models(self) -> list[dict]:
        return [
            {
                "id": preset.id,
                "label": preset.label,
                "conf": preset.conf,
                "classes": None if preset.classes is None else list(preset.classes),
            }
            for preset in MODEL_PRESETS.values()
        ]

    def snapshot(self) -> MonitorSnapshot:
        with self._lock:
            return MonitorSnapshot(
                jpeg_bytes=self._jpeg_bytes,
                stats=self._stats,
                recent=list(self._recent),
                error=self._error,
                camera_index=self._camera_index,
                model_id=self._model_id,
                frame_width=self._frame_width,
                frame_height=self._frame_height,
                switching=self._switching,
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
            self._switching = False
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

    def _sync_frame_size(self, detector: Detector) -> None:
        size = detector.frame_size
        with self._lock:
            if size is None:
                self._frame_width = None
                self._frame_height = None
            else:
                self._frame_width, self._frame_height = size

    def _apply_pending_switch(self, detector: Detector) -> None:
        with self._lock:
            target = self._switch_to
            self._switch_to = None
        if target is None:
            return
        try:
            detector.switch_camera(target)
            self._sync_frame_size(detector)
            with self._lock:
                self._camera_index = detector.camera_index
                self._jpeg_bytes = None
                size = detector.frame_size
                size_txt = f" {size[0]}x{size[1]}" if size else ""
                self._error = f"switched to camera {detector.camera_index}{size_txt}"
                self._switching = False
        except Exception as exc:  # noqa: BLE001 — surface to UI
            self._sync_frame_size(detector)
            with self._lock:
                self._camera_index = detector.camera_index
                self._switching = False
                self._error = f"camera switch failed: {exc}"

    def _apply_pending_model(self, detector: Detector) -> None:
        with self._lock:
            target = self._switch_model_to
            self._switch_model_to = None
        if target is None:
            return
        try:
            preset = get_model_preset(target)
            detector.switch_model(preset.path, preset.conf, preset.classes)
            with self._lock:
                self._model_id = preset.id
                self._jpeg_bytes = None
                self._error = f"switched to model {preset.label}"
                self._switching = False
        except Exception as exc:  # noqa: BLE001 — surface to UI
            with self._lock:
                self._switching = False
                self._error = f"model switch failed: {exc}"

    def _run_loop(self) -> None:
        detector = Detector(self._config)
        store = VisitCsvStore(self._config.visits_csv_path)
        counter = VisitCounter(self._config, store=store)
        saver = CaptureSaver(self._config)
        try:
            detector.open()
            self._sync_frame_size(detector)
            with self._lock:
                self._camera_index = detector.camera_index
                self._model_id = self._config.model_id
        except Exception as exc:  # noqa: BLE001 — surface to UI
            self._set_error(f"camera open failed: {exc}")
            return

        try:
            while not self._stop.is_set():
                self._apply_pending_switch(detector)
                self._apply_pending_model(detector)

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
    # Local monitor: pick up HTML/CSS edits without restarting.
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
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
            "camera_index": snap.camera_index,
            "model_id": snap.model_id,
            "frame_width": snap.frame_width,
            "frame_height": snap.frame_height,
            "switching": snap.switching,
        }
        return jsonify(payload)

    @app.route("/api/cameras")
    def api_cameras():
        cameras = monitor.list_cameras()
        return jsonify(
            {
                "cameras": cameras,
                "current": monitor.camera_index,
                "probe_max": cfg.camera_probe_max,
            }
        )

    @app.route("/api/camera", methods=["POST"])
    def api_camera():
        body = request.get_json(silent=True) or {}
        raw = body.get("index", body.get("camera_index"))
        try:
            index = int(raw)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "index must be an integer"}), 400
        if index < 0:
            return jsonify({"ok": False, "error": "index must be >= 0"}), 400
        requested = monitor.request_camera(index)
        return jsonify({"ok": True, "requested": requested, "current": monitor.camera_index})

    @app.route("/api/models")
    def api_models():
        return jsonify(
            {
                "models": monitor.list_models(),
                "current": monitor.model_id,
            }
        )

    @app.route("/api/model", methods=["POST"])
    def api_model():
        body = request.get_json(silent=True) or {}
        raw = body.get("id", body.get("model_id"))
        if not isinstance(raw, str) or not raw.strip():
            return jsonify({"ok": False, "error": "id must be a non-empty string"}), 400
        model_id = raw.strip()
        if model_id not in MODEL_PRESETS:
            return jsonify(
                {
                    "ok": False,
                    "error": f"unknown model id: {model_id}",
                    "known": sorted(MODEL_PRESETS),
                }
            ), 400
        requested = monitor.request_model(model_id)
        return jsonify({"ok": True, "requested": requested, "current": monitor.model_id})

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
        f"model={CONFIG.model_id} ({CONFIG.model_path.name}) conf={CONFIG.conf} "
        f"camera={CONFIG.camera_index} "
        f"size={CONFIG.camera_width}x{CONFIG.camera_height}"
    )
    # threaded=True: MJPEG + /api/stats while the worker holds the camera.
    # use_reloader=False: avoid opening the webcam twice.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
