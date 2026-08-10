"""Central config for the local pigeon monitor (injected into services)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelPreset:
    """One selectable YOLO detection profile."""

    id: str
    label: str
    path: Path
    conf: float
    classes: Optional[tuple[int, ...]]


MODEL_PRESETS: dict[str, ModelPreset] = {
    "pigeon": ModelPreset(
        id="pigeon",
        label="Pigeon",
        path=ROOT_DIR / "data" / "models" / "pigeon.pt",
        conf=0.6,
        classes=None,
    ),
    "bird": ModelPreset(
        id="bird",
        label="Bird (COCO YOLO)",
        path=ROOT_DIR / "data" / "models" / "yolov8n.pt",
        conf=0.25,
        classes=(14,),
    ),
}

DEFAULT_MODEL_ID = "pigeon"


def get_model_preset(model_id: str) -> ModelPreset:
    try:
        return MODEL_PRESETS[model_id]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_PRESETS))
        raise KeyError(f"unknown model_id={model_id!r}; expected one of: {known}") from exc


@dataclass(frozen=True)
class AppConfig:
    model_id: str
    model_path: Path
    conf: float
    classes: Optional[tuple[int, ...]]
    camera_index: int
    device: str
    visit_gap_sec: float
    confirm_sec: float
    save_interval_sec: float
    captures_dir: Path
    camera_warmup_frames: int
    camera_probe_max: int


_DEFAULT_PRESET = get_model_preset(DEFAULT_MODEL_ID)

CONFIG = AppConfig(
    model_id=_DEFAULT_PRESET.id,
    model_path=_DEFAULT_PRESET.path,
    conf=_DEFAULT_PRESET.conf,
    classes=_DEFAULT_PRESET.classes,
    camera_index=0,
    device="cpu",
    visit_gap_sec=30.0,
    confirm_sec=2.0,
    save_interval_sec=10.0,
    captures_dir=ROOT_DIR / "data" / "captures",
    camera_warmup_frames=5,
    camera_probe_max=5,
)
