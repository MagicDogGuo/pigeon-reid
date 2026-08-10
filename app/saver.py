"""Auto-save visit frames under data/captures/YYYY-MM-DD/."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from app.config import AppConfig, CONFIG
from app.counter import VisitStats

# Injected clock for tests; production uses datetime.now (local timezone).
NowFn = Callable[[], datetime]

# Filename example: visit003_20260809_171530_n2.jpg
_CAPTURE_TS_RE = re.compile(r"_(\d{8})_(\d{6})_")


@dataclass(frozen=True)
class RecentCapture:
    """One capture file relative to captures_dir, with display date/time."""

    path: str
    date: str
    captured_at: str

    @classmethod
    def from_path(cls, captures_dir: Path, path: Path) -> "RecentCapture":
        rel = path.relative_to(captures_dir).as_posix()
        captured = _parse_capture_datetime(path)
        if captured is None:
            captured = datetime.fromtimestamp(path.stat().st_mtime)
        day = (
            path.parent.name
            if _looks_like_iso_date(path.parent.name)
            else captured.date().isoformat()
        )
        return cls(
            path=rel,
            date=day,
            captured_at=captured.strftime("%Y-%m-%d %H:%M:%S"),
        )


def list_recent_captures(
    captures_dir: Path,
    *,
    limit: int = 10,
) -> list[RecentCapture]:
    """Return the newest JPEG captures under captures_dir (newest first)."""
    if limit < 1:
        return []
    if not captures_dir.is_dir():
        return []

    scored: list[tuple[datetime, float, Path]] = []
    for path in captures_dir.rglob("*.jpg"):
        if not path.is_file():
            continue
        try:
            path.relative_to(captures_dir)
        except ValueError:
            continue
        captured = _parse_capture_datetime(path)
        mtime = path.stat().st_mtime
        scored.append((captured or datetime.fromtimestamp(mtime), mtime, path))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [
        RecentCapture.from_path(captures_dir, path)
        for _captured, _mtime, path in scored[:limit]
    ]


def _looks_like_iso_date(name: str) -> bool:
    try:
        datetime.strptime(name, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _parse_capture_datetime(path: Path) -> Optional[datetime]:
    match = _CAPTURE_TS_RE.search(path.name)
    if match is None:
        return None
    try:
        return datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y%m%d_%H%M%S")
    except ValueError:
        return None


class CaptureSaver:
    """Save JPEGs on visit start and at most once per save_interval_sec.

    Filename example: visit003_20260809_171530_n2.jpg
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        now_fn: Optional[NowFn] = None,
    ) -> None:
        self._captures_dir = config.captures_dir
        self._save_interval_sec = config.save_interval_sec
        self._now_fn = now_fn or datetime.now
        self._visit_index = 0
        self._local_date = self._now_fn().date()
        self._last_save_at: Optional[datetime] = None

    def maybe_save(
        self,
        frame: np.ndarray,
        stats: VisitStats,
    ) -> Optional[Path]:
        """Save when a visit starts, or when the interval elapses mid-visit.

        Only saves while pigeons are present (concurrent_count >= 1).
        Returns the written path, or None if nothing was saved.
        """
        if not stats.in_visit or stats.concurrent_count < 1:
            return None

        now = self._now_fn()
        if stats.local_date != self._local_date:
            self._local_date = stats.local_date
            self._visit_index = 0
            self._last_save_at = None

        should_save = False
        if stats.visit_started:
            self._visit_index += 1
            should_save = True
        elif self._last_save_at is not None and (
            now - self._last_save_at
        ) >= timedelta(seconds=self._save_interval_sec):
            should_save = True

        if not should_save:
            return None

        day_dir = self._captures_dir / stats.local_date.isoformat()
        day_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"visit{self._visit_index:03d}_"
            f"{now.strftime('%Y%m%d_%H%M%S')}_"
            f"n{stats.concurrent_count}.jpg"
        )
        path = day_dir / filename
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            raise RuntimeError(f"Failed to write JPEG: {path}")
        self._last_save_at = now
        return path


class _FakeClock:
    """Mutable clock for deterministic smoke tests."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _demo() -> None:
    """Deterministic smoke: start save after confirm, interval save, no empty-gap save, new visit."""
    import shutil
    import tempfile
    from dataclasses import replace

    from app.counter import VisitCounter

    tmp = Path(tempfile.mkdtemp(prefix="pigeon-reid-saver-"))
    try:
        clock = _FakeClock(datetime(2026, 8, 9, 17, 15, 30))
        cfg = replace(CONFIG, captures_dir=tmp, save_interval_sec=10.0, confirm_sec=2.0)
        counter = VisitCounter(cfg, now_fn=clock)
        saver = CaptureSaver(cfg, now_fn=clock)
        frame = np.zeros((48, 64, 3), dtype=np.uint8)

        s = counter.update(0)
        p = saver.maybe_save(frame, s)
        _assert(p is None, "idle: no save")

        s = counter.update(2)
        p = saver.maybe_save(frame, s)
        _assert(p is None, "confirming: no save yet")

        clock.now = datetime(2026, 8, 9, 17, 15, 31)
        s = counter.update(2)
        p = saver.maybe_save(frame, s)
        _assert(p is None, "still confirming < 2s")

        clock.now = datetime(2026, 8, 9, 17, 15, 32)
        s = counter.update(2)
        p = saver.maybe_save(frame, s)
        _assert(p is not None and p.exists(), "confirmed visit start saves")
        _assert(p.name == "visit001_20260809_171532_n2.jpg", f"name={p.name}")
        _assert(p.parent.name == "2026-08-09", "day folder")

        clock.now = datetime(2026, 8, 9, 17, 15, 37)
        s = counter.update(2)
        p = saver.maybe_save(frame, s)
        _assert(p is None, "interval < 10s: no save")

        clock.now = datetime(2026, 8, 9, 17, 15, 42)
        s = counter.update(3)
        p = saver.maybe_save(frame, s)
        _assert(
            p is not None and p.name == "visit001_20260809_171542_n3.jpg",
            "interval save same visit",
        )

        clock.now = datetime(2026, 8, 9, 17, 15, 43)
        s = counter.update(0)
        p = saver.maybe_save(frame, s)
        _assert(p is None, "zero gap: no save")

        clock.now = datetime(2026, 8, 9, 17, 16, 13)
        s = counter.update(0)
        _assert(s.visit_ended, "visit ended after gap")

        clock.now = datetime(2026, 8, 9, 17, 16, 14)
        s = counter.update(1)
        p = saver.maybe_save(frame, s)
        _assert(p is None, "new presence confirming")

        clock.now = datetime(2026, 8, 9, 17, 16, 16)
        s = counter.update(1)
        p = saver.maybe_save(frame, s)
        _assert(
            p is not None and p.name.startswith("visit002_"),
            "new visit increments index",
        )

        print("saver smoke OK")
        print(
            f"captures_dir={cfg.captures_dir} "
            f"save_interval_sec={cfg.save_interval_sec} last={p.name}"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    _demo()
