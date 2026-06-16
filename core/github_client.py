from __future__ import annotations

import datetime as dt
import collections
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

from core.date_utils import iso_to_date
from core.features import EventSignals
from core.models import RepoRef

GITHUB_API = "https://api.github.com"
GITHUB_JSON_ACCEPT = "application/vnd.github+json"
GITHUB_STAR_ACCEPT = "application/vnd.github.star+json"
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_EVENT_PAGES = 10
DEFAULT_RELEASE_PAGE_LIMIT = 50
_LINK_RE = re.compile(r'<([^>]+)>;\s*rel="([^"]+)"')


@dataclass(frozen=True)
class RepoMetadata:
    stargazers_count: Optional[int]
    etag: Optional[str]
    not_modified: bool = False


@dataclass(frozen=True)
class StargazerPage:
    page: int
    dates: List[dt.date]
    links: Dict[str, str]


def gh_headers(
    token: Optional[str],
    *,
    accept: str = GITHUB_JSON_ACCEPT,
) -> Dict[str, str]:
    headers = {"Accept": accept}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def parse_link_header(link_header: Optional[str]) -> Dict[str, str]:
    if not link_header:
        return {}

    out: Dict[str, str] = {}
    for part in link_header.split(","):
        match = _LINK_RE.search(part)
        if match:
            out[match.group(2)] = match.group(1)
    return out


def _get_json(
    session: requests.Session, url: str, headers: Dict[str, str]
) -> Tuple[Any, requests.Response]:
    response = session.get(url, headers=headers, timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json(), response


def _parse_stargazer_dates(payload: Any) -> List[dt.date]:
    if not isinstance(payload, list):
        raise ValueError("Unexpected GitHub API response format for stargazers")

    dates: List[dt.date] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        starred_at = item.get("starred_at")
        if starred_at:
            dates.append(iso_to_date(starred_at))
    return dates


def fetch_repo_metadata(
    repo: RepoRef,
    token: Optional[str],
    etag: Optional[str] = None,
) -> RepoMetadata:
    headers = gh_headers(token)
    if etag:
        headers["If-None-Match"] = etag
    url = f"{GITHUB_API}/repos/{repo.owner}/{repo.repo}"

    with requests.Session() as session:
        response = session.get(url, headers=headers, timeout=DEFAULT_TIMEOUT_SECONDS)
        if response.status_code == 304:
            return RepoMetadata(
                stargazers_count=None,
                etag=response.headers.get("ETag") or etag,
                not_modified=True,
            )
        response.raise_for_status()
        payload = response.json()

    if not isinstance(payload, dict):
        raise ValueError("Unexpected GitHub API response format for repository")
    count = payload.get("stargazers_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("Repository response has invalid stargazers_count")
    return RepoMetadata(
        stargazers_count=count,
        etag=response.headers.get("ETag"),
        not_modified=False,
    )


def fetch_stargazer_page(
    repo: RepoRef,
    token: Optional[str],
    page: int,
    *,
    per_page: int = 100,
) -> StargazerPage:
    if page <= 0:
        raise ValueError("page must be > 0")
    if per_page <= 0 or per_page > 100:
        raise ValueError("per_page must be in [1, 100]")

    headers = gh_headers(token, accept=GITHUB_STAR_ACCEPT)
    url = (
        f"{GITHUB_API}/repos/{repo.owner}/{repo.repo}/stargazers"
        f"?per_page={per_page}&page={page}"
    )
    with requests.Session() as session:
        payload, response = _get_json(session, url, headers)
    return StargazerPage(
        page=page,
        dates=_parse_stargazer_dates(payload),
        links=parse_link_header(response.headers.get("Link")),
    )


def fetch_stars(repo: RepoRef, token: Optional[str]) -> List[dt.date]:
    headers = gh_headers(token, accept=GITHUB_STAR_ACCEPT)
    url = f"{GITHUB_API}/repos/{repo.owner}/{repo.repo}/stargazers?per_page=100"
    dates: List[dt.date] = []

    with requests.Session() as session:
        while url:
            payload, response = _get_json(session, url, headers)
            dates.extend(_parse_stargazer_dates(payload))

            url = parse_link_header(response.headers.get("Link")).get("next")

    return dates


def fetch_release_dates(
    repo: RepoRef, token: Optional[str], page_limit: int = DEFAULT_RELEASE_PAGE_LIMIT
) -> List[dt.date]:
    headers = gh_headers(token)
    url = f"{GITHUB_API}/repos/{repo.owner}/{repo.repo}/releases?per_page=100"
    dates: List[dt.date] = []
    pages = 0

    with requests.Session() as session:
        while url and pages < page_limit:
            payload, response = _get_json(session, url, headers)
            if not isinstance(payload, list):
                raise ValueError("Unexpected GitHub API response format for releases")

            for item in payload:
                stamp = item.get("published_at") or item.get("created_at")
                if stamp:
                    dates.append(iso_to_date(stamp))

            url = parse_link_header(response.headers.get("Link")).get("next")
            pages += 1

    return dates


def fetch_recent_repo_events(
    repo: RepoRef, token: Optional[str], max_pages: int = DEFAULT_EVENT_PAGES
) -> List[Dict[str, Any]]:
    headers = gh_headers(token)
    events: List[Dict[str, Any]] = []

    with requests.Session() as session:
        for page in range(1, max_pages + 1):
            url = f"{GITHUB_API}/repos/{repo.owner}/{repo.repo}/events?per_page=100&page={page}"
            payload, _ = _get_json(session, url, headers)
            if not isinstance(payload, list):
                raise ValueError("Unexpected GitHub API response format for events")
            if not payload:
                break
            events.extend(item for item in payload if isinstance(item, dict))

    return events


def fetch_event_signals(repo: RepoRef, token: Optional[str]) -> EventSignals:
    release_counter = collections.Counter(fetch_release_dates(repo, token))
    commits_counter: collections.Counter[dt.date] = collections.Counter()
    issues_counter: collections.Counter[dt.date] = collections.Counter()
    prs_counter: collections.Counter[dt.date] = collections.Counter()

    for event in fetch_recent_repo_events(repo, token):
        created_at = event.get("created_at")
        if not created_at:
            continue
        day = iso_to_date(created_at)
        event_type = event.get("type")
        payload = event.get("payload") or {}

        if event_type == "PushEvent":
            size = payload.get("size")
            if not isinstance(size, int):
                commits = payload.get("commits")
                size = len(commits) if isinstance(commits, list) else 0
            commits_counter[day] += max(size, 0)
        elif event_type == "IssuesEvent" and payload.get("action") == "opened":
            issues_counter[day] += 1
        elif event_type == "PullRequestEvent" and payload.get("action") == "opened":
            prs_counter[day] += 1

    return EventSignals(
        releases=release_counter,
        commits=commits_counter,
        issues=issues_counter,
        prs=prs_counter,
    )
