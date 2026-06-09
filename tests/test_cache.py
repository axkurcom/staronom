from __future__ import annotations

import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from typing import Optional, Sequence

from core.cache import (
    load_stars_cache,
    save_stars_cache,
    stars_cache_is_fresh,
)
from core.cli import load_star_series
from core.models import RepoRef


class StarsCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = RepoRef.parse("owner/repo")
        self.now = dt.datetime(2026, 1, 3, 12, tzinfo=dt.timezone.utc)

    def test_cache_roundtrip_and_freshness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stars.json"
            days = [dt.date(2026, 1, 2), dt.date(2026, 1, 3)]
            counts = [2, 1]

            save_stars_cache(path, self.repo, days, counts, fetched_at_utc=self.now)
            loaded = load_stars_cache(path, self.repo, now_utc=self.now)

            self.assertEqual(loaded.repo, self.repo.full_name)
            self.assertEqual(loaded.days, days)
            self.assertEqual(loaded.counts, counts)
            self.assertTrue(
                stars_cache_is_fresh(loaded, now_utc=self.now, ttl_hours=6.0)
            )

    def test_invalid_cache_payloads_are_rejected(self) -> None:
        base_payload = {
            "schema_version": 1,
            "repo": self.repo.full_name,
            "fetched_at_utc": "2026-01-03T12:00:00Z",
            "days": ["2026-01-02", "2026-01-03"],
            "counts": [2, 1],
        }
        invalid_payloads = [
            {**base_payload, "schema_version": 99},
            {**base_payload, "repo": "other/repo"},
            {**base_payload, "counts": [1]},
            {**base_payload, "days": ["2026-01-01", "2026-01-03"]},
            {**base_payload, "counts": [1, -1]},
            {**base_payload, "days": ["2026-01-02", "2026-01-04"]},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stars.json"
            for payload in invalid_payloads:
                with self.subTest(payload=payload):
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_stars_cache(path, self.repo, now_utc=self.now)

    def test_fresh_cache_hit_avoids_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stars.json"
            save_stars_cache(
                path,
                self.repo,
                [dt.date(2026, 1, 3)],
                [4],
                fetched_at_utc=self.now,
            )

            def fail_fetch(_: RepoRef, __: Optional[str]) -> Sequence[dt.date]:
                raise AssertionError("fetcher should not be called")

            result = load_star_series(
                repo=self.repo,
                token=None,
                out_dir=tmp,
                now_utc=self.now,
                cache_ttl_hours=6.0,
                stars_cache_path=str(path),
                refresh_cache=False,
                offline_cache=False,
                no_cache=False,
                fetcher=fail_fetch,
            )

            self.assertEqual(result.source, "cache")
            self.assertEqual(result.series.counts, [4])

    def test_stale_auto_cache_fetches_and_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stars.json"
            save_stars_cache(
                path,
                self.repo,
                [dt.date(2026, 1, 2)],
                [5],
                fetched_at_utc=self.now - dt.timedelta(days=1),
            )
            calls = []

            def fetch(_: RepoRef, __: Optional[str]) -> Sequence[dt.date]:
                calls.append(True)
                return [dt.date(2026, 1, 2), dt.date(2026, 1, 3)]

            result = load_star_series(
                repo=self.repo,
                token=None,
                out_dir=tmp,
                now_utc=self.now,
                cache_ttl_hours=6.0,
                stars_cache_path=str(path),
                refresh_cache=False,
                offline_cache=False,
                no_cache=False,
                fetcher=fetch,
            )

            self.assertEqual(calls, [True])
            self.assertEqual(result.source, "api")
            self.assertEqual(result.series.days, [dt.date(2026, 1, 2), dt.date(2026, 1, 3)])
            self.assertEqual(result.series.counts, [1, 1])

            loaded = load_stars_cache(path, self.repo, now_utc=self.now)
            self.assertEqual(loaded.counts, [1, 1])

    def test_offline_cache_uses_stale_data_and_extends_trailing_zeros(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stars.json"
            save_stars_cache(
                path,
                self.repo,
                [dt.date(2026, 1, 1)],
                [2],
                fetched_at_utc=self.now - dt.timedelta(days=2),
            )

            def fail_fetch(_: RepoRef, __: Optional[str]) -> Sequence[dt.date]:
                raise AssertionError("fetcher should not be called")

            result = load_star_series(
                repo=self.repo,
                token=None,
                out_dir=tmp,
                now_utc=self.now,
                cache_ttl_hours=6.0,
                stars_cache_path=str(path),
                refresh_cache=False,
                offline_cache=True,
                no_cache=False,
                fetcher=fail_fetch,
            )

            self.assertEqual(result.source, "cache")
            self.assertEqual(
                result.series.days,
                [dt.date(2026, 1, 1), dt.date(2026, 1, 2), dt.date(2026, 1, 3)],
            )
            self.assertEqual(result.series.counts, [2, 0, 0])

    def test_refresh_cache_ignores_fresh_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stars.json"
            save_stars_cache(
                path,
                self.repo,
                [dt.date(2026, 1, 3)],
                [9],
                fetched_at_utc=self.now,
            )
            calls = []

            def fetch(_: RepoRef, __: Optional[str]) -> Sequence[dt.date]:
                calls.append(True)
                return [dt.date(2026, 1, 3)]

            result = load_star_series(
                repo=self.repo,
                token=None,
                out_dir=tmp,
                now_utc=self.now,
                cache_ttl_hours=6.0,
                stars_cache_path=str(path),
                refresh_cache=True,
                offline_cache=False,
                no_cache=False,
                fetcher=fetch,
            )

            self.assertEqual(calls, [True])
            self.assertEqual(result.source, "api")
            self.assertEqual(result.series.counts, [1])


if __name__ == "__main__":
    unittest.main()
