"""Webcam + YOLO pigeon detector (no UI)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from app.config import AppConfig, CONFIG


@dataclass(frozen=True)
class DetectionFrame:
    """One camera (or still) frame after YOLO inference."""

    frame: np.ndarray
    annotated_frame: np.ndarray
    box_count: int


class Detector:
    """OpenCV webcam + Ultralytics YOLO detection loop."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._camera_index = config.camera_index
        self._model_path = config.model_path
        self._conf = config.conf
        self._classes = config.classes
        self._model = YOLO(str(config.model_path))
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_size: Optional[tuple[int, int]] = None

    @property
    def camera_index(self) -> int:
        return self._camera_index

    @property
    def frame_size(self) -> Optional[tuple[int, int]]:
        """Actual capture size as (width, height), or None if not open."""
        return self._frame_size

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def conf(self) -> float:
        return self._conf

    @property
    def classes(self) -> Optional[tuple[int, ...]]:
        return self._classes

    def open(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            return
        cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Failed to open webcam index {self._camera_index}"
            )
        width = self._config.camera_width
        height = self._config.camera_height
        if width > 0 and height > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        for _ in range(self._config.camera_warmup_frames):
            cap.read()
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_size = (actual_w, actual_h) if actual_w > 0 and actual_h > 0 else None
        self._cap = cap

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frame_size = None

    def switch_camera(self, camera_index: int) -> None:
        """Close current capture (if any) and open a different camera index."""
        if camera_index < 0:
            raise ValueError(f"camera_index must be >= 0, got {camera_index}")
        if (
            self._cap is not None
            and self._cap.isOpened()
            and camera_index == self._camera_index
        ):
            return
        previous = self._camera_index
        self.close()
        self._camera_index = camera_index
        try:
            self.open()
        except Exception:
            self._camera_index = previous
            try:
                self.open()
            except Exception:
                pass
            raise

    def switch_model(
        self,
        path: Path,
        conf: float,
        classes: Optional[tuple[int, ...]],
    ) -> None:
        """Load a different YOLO weights file and detection filters."""
        if not path.is_file():
            raise FileNotFoundError(f"model weights not found: {path}")
        if conf <= 0 or conf > 1:
            raise ValueError(f"conf must be in (0, 1], got {conf}")
        previous_path = self._model_path
        previous_conf = self._conf
        previous_classes = self._classes
        previous_model = self._model
        try:
            self._model = YOLO(str(path))
            self._model_path = path
            self._conf = conf
            self._classes = classes
        except Exception:
            self._model = previous_model
            self._model_path = previous_path
            self._conf = previous_conf
            self._classes = previous_classes
            raise

    def predict_frame(self, frame: np.ndarray) -> DetectionFrame:
        predict_kwargs = {
            "source": frame,
            "device": self._config.device,
            "conf": self._conf,
            "verbose": False,
        }
        if self._classes is not None:
            predict_kwargs["classes"] = list(self._classes)
        result = self._model.predict(**predict_kwargs)[0]
        boxes = result.boxes
        box_count = 0 if boxes is None else len(boxes)
        annotated = result.plot()
        return DetectionFrame(
            frame=frame,
            annotated_frame=annotated,
            box_count=box_count,
        )

    def read(self) -> Optional[DetectionFrame]:
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Detector is not open; call open() first")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return self.predict_frame(frame)

    def frames(self) -> Generator[DetectionFrame, None, None]:
        self.open()
        try:
            while True:
                detection = self.read()
                if detection is None:
                    break
                yield detection
        finally:
            self.close()

    def __enter__(self) -> "Detector":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def list_camera_indices(probe_max: int) -> list[int]:
    """Return candidate camera indices 0..probe_max.

    Windows DSHOW probing of missing devices can hang, so the UI offers this
    fixed range and open/switch validates the chosen index.
    """
    if probe_max < 0:
        raise ValueError(f"probe_max must be >= 0, got {probe_max}")
    return list(range(probe_max + 1))


def main() -> None:
    """No-UI loop: print box_count each frame. Ctrl+C to stop."""
    print(
        f"model={CONFIG.model_path} conf={CONFIG.conf} "
        f"device={CONFIG.device} camera={CONFIG.camera_index} "
        f"size={CONFIG.camera_width}x{CONFIG.camera_height}"
    )
    detector = Detector(CONFIG)
    try:
        frame_idx = 0
        for detection in detector.frames():
            frame_idx += 1
            print(f"frame={frame_idx} boxes={detection.box_count}")
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
