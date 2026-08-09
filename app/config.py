"""Central config for the local pigeon monitor (injected into services)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AppConfig:
    model_path: Path
    camera_index: int
    conf: float
    device: str
    visit_gap_sec: float
    confirm_sec: float
    save_interval_sec: float
    captures_dir: Path
    camera_warmup_frames: int
    camera_probe_max: int


CONFIG = AppConfig(
    model_path=ROOT_DIR / "data" / "models" / "pigeon.pt",
    camera_index=0,
    conf=0.45,
    device="cpu",
    visit_gap_sec=30.0,
    confirm_sec=2.0,
    save_interval_sec=10.0,
    captures_dir=ROOT_DIR / "data" / "captures",
    camera_warmup_frames=5,
    camera_probe_max=5,
)
