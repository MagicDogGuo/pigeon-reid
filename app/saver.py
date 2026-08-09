"""Auto-save visit frames under data/captures/YYYY-MM-DD/."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from app.config import AppConfig, CONFIG
from app.counter import VisitStats

# Injected clock for tests; production uses datetime.now (local timezone).
NowFn = Callable[[], datetime]


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
    """Deterministic smoke: start save, interval save, no empty-gap save, new visit."""
    import shutil
    import tempfile
    from dataclasses import replace

    from app.counter import VisitCounter

    tmp = Path(tempfile.mkdtemp(prefix="pigeon-reid-saver-"))
    try:
        clock = _FakeClock(datetime(2026, 8, 9, 17, 15, 30))
        cfg = replace(CONFIG, captures_dir=tmp, save_interval_sec=10.0)
        counter = VisitCounter(cfg, now_fn=clock)
        saver = CaptureSaver(cfg, now_fn=clock)
        frame = np.zeros((48, 64, 3), dtype=np.uint8)

        s = counter.update(0)
        p = saver.maybe_save(frame, s)
        _assert(p is None, "idle: no save")

        s = counter.update(2)
        p = saver.maybe_save(frame, s)
        _assert(p is not None and p.exists(), "visit start saves")
        _assert(p.name == "visit001_20260809_171530_n2.jpg", f"name={p.name}")
        _assert(p.parent.name == "2026-08-09", "day folder")

        clock.now = datetime(2026, 8, 9, 17, 15, 35)
        s = counter.update(2)
        p = saver.maybe_save(frame, s)
        _assert(p is None, "interval < 10s: no save")

        clock.now = datetime(2026, 8, 9, 17, 15, 40)
        s = counter.update(3)
        p = saver.maybe_save(frame, s)
        _assert(
            p is not None and p.name == "visit001_20260809_171540_n3.jpg",
            "interval save same visit",
        )

        clock.now = datetime(2026, 8, 9, 17, 15, 41)
        s = counter.update(0)
        p = saver.maybe_save(frame, s)
        _assert(p is None, "zero gap: no save")

        clock.now = datetime(2026, 8, 9, 17, 16, 11)
        s = counter.update(0)
        _assert(s.visit_ended, "visit ended after gap")

        clock.now = datetime(2026, 8, 9, 17, 16, 12)
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
