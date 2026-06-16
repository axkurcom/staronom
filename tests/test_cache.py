from __future__ import annotations

import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from typing import Optional, Sequence

from core.cache import (
    LEGACY_STARS_CACHE_SCHEMA_VERSION,
    STARS_CACHE_SCHEMA_VERSION,
    load_stars_cache,
    save_stars_cache,
    stars_cache_is_fresh,
)
from core.cli import load_star_series
from core.github_client import RepoMetadata, StargazerPage
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
            self.assertEqual(loaded.start_day, days[0])
            self.assertEqual(loaded.current_stargazers_count, 3)
            self.assertTrue(
                stars_cache_is_fresh(loaded, now_utc=self.now, ttl_hours=6.0)
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], STARS_CACHE_SCHEMA_VERSION)
            self.assertEqual(payload["start_day"], "2026-01-02")
            self.assertNotIn("days", payload)

    def test_v1_cache_is_read_and_marked_for_migration(self) -> None:
        payload = {
            "schema_version": LEGACY_STARS_CACHE_SCHEMA_VERSION,
            "repo": self.repo.full_name,
            "fetched_at_utc": "2026-01-03T12:00:00Z",
            "days": ["2026-01-02", "2026-01-03"],
            "counts": [2, 1],
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stars.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_stars_cache(path, self.repo, now_utc=self.now)

            self.assertEqual(loaded.days, [dt.date(2026, 1, 2), dt.date(2026, 1, 3)])
            self.assertEqual(loaded.counts, [2, 1])
            self.assertEqual(loaded.migrated_from_schema, LEGACY_STARS_CACHE_SCHEMA_VERSION)

    def test_invalid_cache_payloads_are_rejected(self) -> None:
        base_v2 = {
            "schema_version": STARS_CACHE_SCHEMA_VERSION,
            "repo": self.repo.full_name,
            "fetched_at_utc": "2026-01-03T12:00:00Z",
            "checked_at_utc": "2026-01-03T12:00:00Z",
            "full_scan_at_utc": "2026-01-03T12:00:00Z",
            "start_day": "2026-01-02",
            "counts": [2, 1],
            "current_stargazers_count": 3,
            "repo_etag": '"abc"',
        }
        invalid_payloads = [
            {**base_v2, "schema_version": 99},
            {**base_v2, "repo": "other/repo"},
            {**base_v2, "counts": []},
            {**base_v2, "counts": [1, -1]},
            {**base_v2, "start_day": "2026-01-04"},
            {**base_v2, "current_stargazers_count": -1},
            {**base_v2, "repo_etag": 123},
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
            self.assertEqual(result.detail, "cache hit")
            self.assertEqual(result.series.counts, [4])

    def test_stale_cache_with_repo_304_avoids_stargazer_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stars.json"
            save_stars_cache(
                path,
                self.repo,
                [dt.date(2026, 1, 2)],
                [5],
                fetched_at_utc=self.now - dt.timedelta(days=1),
                repo_etag='"old"',
            )

            def fail_fetch(_: RepoRef, __: Optional[str]) -> Sequence[dt.date]:
                raise AssertionError("full fetcher should not be called")

            def metadata(repo: RepoRef, token: Optional[str], etag: Optional[str]) -> RepoMetadata:
                self.assertEqual(etag, '"old"')
                return RepoMetadata(None, '"new"', not_modified=True)

            def fail_page(_: RepoRef, __: Optional[str], ___: int) -> StargazerPage:
                raise AssertionError("page fetcher should not be called")

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
                repo_metadata_fetcher=metadata,
                stargazer_page_fetcher=fail_page,
            )

            self.assertEqual(result.source, "cache")
            self.assertEqual(result.detail, "repo metadata 304")
            loaded = load_stars_cache(path, self.repo, now_utc=self.now)
            self.assertEqual(loaded.checked_at_utc, self.now)
            self.assertEqual(loaded.repo_etag, '"new"')

    def test_stale_cache_with_same_star_count_avoids_stargazer_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stars.json"
            save_stars_cache(
                path,
                self.repo,
                [dt.date(2026, 1, 2)],
                [5],
                fetched_at_utc=self.now - dt.timedelta(days=1),
            )

            def fail_fetch(_: RepoRef, __: Optional[str]) -> Sequence[dt.date]:
                raise AssertionError("full fetcher should not be called")

            def metadata(_: RepoRef, __: Optional[str], ___: Optional[str]) -> RepoMetadata:
                return RepoMetadata(5, '"same"', not_modified=False)

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
                repo_metadata_fetcher=metadata,
            )

            self.assertEqual(result.source, "cache")
            self.assertEqual(result.detail, "repo star count unchanged")
            self.assertEqual(result.series.counts, [5, 0])

            loaded = load_stars_cache(path, self.repo, now_utc=self.now)
            self.assertEqual(loaded.current_stargazers_count, 5)
            self.assertEqual(loaded.repo_etag, '"same"')

    def test_small_star_count_increase_scans_and_merges_tail_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stars.json"
            save_stars_cache(
                path,
                self.repo,
                [dt.date(2026, 1, 1), dt.date(2026, 1, 2)],
                [2, 1],
                fetched_at_utc=self.now - dt.timedelta(days=1),
            )

            def fail_fetch(_: RepoRef, __: Optional[str]) -> Sequence[dt.date]:
                raise AssertionError("full fetcher should not be called")

            def metadata(_: RepoRef, __: Optional[str], ___: Optional[str]) -> RepoMetadata:
                return RepoMetadata(5, '"tail"', not_modified=False)

            page_calls = []

            def page_fetch(_: RepoRef, __: Optional[str], page: int) -> StargazerPage:
                page_calls.append(page)
                return StargazerPage(
                    page=page,
                    dates=[
                        dt.date(2026, 1, 1),
                        dt.date(2026, 1, 1),
                        dt.date(2026, 1, 2),
                        dt.date(2026, 1, 3),
                        dt.date(2026, 1, 3),
                    ],
                    links={},
                )

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
                repo_metadata_fetcher=metadata,
                stargazer_page_fetcher=page_fetch,
            )

            self.assertEqual(result.source, "tail")
            self.assertEqual(page_calls, [1])
            self.assertEqual(result.series.counts, [2, 1, 2])
            loaded = load_stars_cache(path, self.repo, now_utc=self.now)
            self.assertEqual(loaded.counts, [2, 1, 2])
            self.assertEqual(loaded.full_scan_at_utc, self.now - dt.timedelta(days=1))

    def test_count_decrease_falls_back_to_full_refresh(self) -> None:
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

            def metadata(_: RepoRef, __: Optional[str], ___: Optional[str]) -> RepoMetadata:
                return RepoMetadata(3, '"lower"', not_modified=False)

            def fetch(_: RepoRef, __: Optional[str]) -> Sequence[dt.date]:
                calls.append(True)
                return [dt.date(2026, 1, 2), dt.date(2026, 1, 3), dt.date(2026, 1, 3)]

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
                repo_metadata_fetcher=metadata,
            )

            self.assertEqual(calls, [True])
            self.assertEqual(result.source, "api")
            self.assertEqual(result.detail, "full refresh")
            self.assertEqual(result.series.counts, [1, 2])

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
            self.assertEqual(result.detail, "offline cache")
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
            self.assertEqual(result.detail, "full refresh")
            self.assertEqual(result.series.counts, [1])


if __name__ == "__main__":
    unittest.main()
