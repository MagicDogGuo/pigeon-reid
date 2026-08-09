"""Visit session state machine: concurrent count + visits today."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable, Optional

from app.config import AppConfig, CONFIG

# Injected clock for tests; production uses datetime.now (local timezone).
NowFn = Callable[[], datetime]


@dataclass(frozen=True)
class VisitStats:
    """Snapshot after one update() call."""

    concurrent_count: int
    visits_today: int
    in_visit: bool
    visit_started: bool
    visit_ended: bool
    local_date: date


class VisitCounter:
    """Track live box count and daily visit sessions.

    Rules:
    - concurrent_count = current frame box_count
    - 0 -> N starts a visit and adds N to visits_today
    - same visit does not add again (count may change mid-visit)
    - continuous visit_gap_sec of zeros ends the visit
    - local calendar day rollover resets visits_today
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        now_fn: Optional[NowFn] = None,
    ) -> None:
        self._visit_gap_sec = config.visit_gap_sec
        self._now_fn = now_fn or datetime.now
        self._visits_today = 0
        self._in_visit = False
        self._zero_since: Optional[datetime] = None
        self._local_date = self._now_fn().date()

    def update(self, box_count: int) -> VisitStats:
        if box_count < 0:
            raise ValueError(f"box_count must be >= 0, got {box_count}")

        now = self._now_fn()
        today = now.date()
        if today != self._local_date:
            self._local_date = today
            self._visits_today = 0
            self._in_visit = False
            self._zero_since = None

        visit_started = False
        visit_ended = False
        concurrent = box_count

        if box_count >= 1:
            self._zero_since = None
            if not self._in_visit:
                self._in_visit = True
                self._visits_today += box_count
                visit_started = True
        else:
            if self._in_visit:
                if self._zero_since is None:
                    self._zero_since = now
                elif (now - self._zero_since) >= timedelta(
                    seconds=self._visit_gap_sec
                ):
                    self._in_visit = False
                    self._zero_since = None
                    visit_ended = True

        return VisitStats(
            concurrent_count=concurrent,
            visits_today=self._visits_today,
            in_visit=self._in_visit,
            visit_started=visit_started,
            visit_ended=visit_ended,
            local_date=self._local_date,
        )


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
    """Deterministic smoke: enter, stay, leave 30s, re-enter, day rollover."""
    clock = _FakeClock(datetime(2026, 8, 9, 12, 0, 0))
    counter = VisitCounter(CONFIG, now_fn=clock)

    s = counter.update(0)
    _assert(s.visits_today == 0 and not s.in_visit, "idle")

    s = counter.update(3)
    _assert(
        s.visits_today == 3
        and s.concurrent_count == 3
        and s.visit_started
        and s.in_visit,
        "enter +3 concurrent=3",
    )

    s = counter.update(2)
    _assert(s.visits_today == 3 and not s.visit_started, "same visit no add")

    s = counter.update(0)
    _assert(s.in_visit and not s.visit_ended, "zero starts gap timer")

    clock.now = datetime(2026, 8, 9, 12, 0, 29)
    s = counter.update(0)
    _assert(s.in_visit and not s.visit_ended, "gap < 30s still in visit")

    clock.now = datetime(2026, 8, 9, 12, 0, 30)
    s = counter.update(0)
    _assert(not s.in_visit and s.visit_ended and s.visits_today == 3, "gap end")

    clock.now = datetime(2026, 8, 9, 12, 0, 31)
    s = counter.update(1)
    _assert(s.visits_today == 4 and s.visit_started, "re-enter +1")

    clock.now = datetime(2026, 8, 10, 0, 0, 1)
    s = counter.update(0)
    _assert(s.visits_today == 0 and not s.in_visit, "local day rollover")

    s = counter.update(1)
    _assert(s.visits_today == 1 and s.visit_started, "new day first visit")

    print("counter smoke OK")
    print(
        f"visit_gap_sec={CONFIG.visit_gap_sec} "
        f"final visits_today={s.visits_today} concurrent={s.concurrent_count}"
    )


if __name__ == "__main__":
    _demo()
