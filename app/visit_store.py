"""Persist daily visit totals to a local CSV file."""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Optional

CSV_FIELDNAMES = ("date", "visits_today", "updated_at")


class VisitCsvStore:
    """Load/save visits_today by local calendar date (one row per day)."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def load_visits(self, day: date) -> int:
        rows = self._read_rows()
        key = day.isoformat()
        for row in rows:
            if row.get("date") == key:
                try:
                    return max(0, int(row.get("visits_today", "0")))
                except (TypeError, ValueError):
                    return 0
        return 0

    def save_visits(
        self,
        day: date,
        visits_today: int,
        *,
        updated_at: Optional[datetime] = None,
    ) -> None:
        if visits_today < 0:
            raise ValueError(f"visits_today must be >= 0, got {visits_today}")
        when = updated_at or datetime.now()
        key = day.isoformat()
        rows = self._read_rows()
        payload = {
            "date": key,
            "visits_today": str(visits_today),
            "updated_at": when.isoformat(timespec="seconds"),
        }
        replaced = False
        for idx, row in enumerate(rows):
            if row.get("date") == key:
                rows[idx] = payload
                replaced = True
                break
        if not replaced:
            rows.append(payload)
        rows.sort(key=lambda r: r.get("date") or "")
        self._write_rows(rows)

    def _read_rows(self) -> list[dict[str, str]]:
        if not self._path.is_file():
            return []
        with self._path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            return [
                {name: (row.get(name) or "") for name in CSV_FIELDNAMES}
                for row in reader
            ]

    def _write_rows(self, rows: list[dict[str, str]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in CSV_FIELDNAMES})
