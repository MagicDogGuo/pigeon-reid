"""Webcam + YOLO pigeon detector (no UI)."""

from __future__ import annotations

from dataclasses import dataclass
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
    """OpenCV webcam + Ultralytics YOLO pigeon detection loop."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._model = YOLO(str(config.model_path))
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            return
        cap = cv2.VideoCapture(self._config.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Failed to open webcam index {self._config.camera_index}"
            )
        for _ in range(self._config.camera_warmup_frames):
            cap.read()
        self._cap = cap

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def predict_frame(self, frame: np.ndarray) -> DetectionFrame:
        result = self._model.predict(
            source=frame,
            device=self._config.device,
            conf=self._config.conf,
            verbose=False,
        )[0]
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


def main() -> None:
    """No-UI loop: print box_count each frame. Ctrl+C to stop."""
    print(
        f"model={CONFIG.model_path} conf={CONFIG.conf} "
        f"device={CONFIG.device} camera={CONFIG.camera_index}"
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
