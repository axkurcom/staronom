from __future__ import annotations

import datetime as dt
import re
from typing import Dict, List, Optional

import requests

from core.date_utils import iso_to_date
from core.models import RepoRef

GITHUB_API = "https://api.github.com"
DEFAULT_TIMEOUT_SECONDS = 20
_LINK_RE = re.compile(r'<([^>]+)>;\s*rel="([^"]+)"')


def gh_headers(token: Optional[str]) -> Dict[str, str]:
    headers = {"Accept": "application/vnd.github.star+json"}
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


def fetch_stars(repo: RepoRef, token: Optional[str]) -> List[dt.date]:
    headers = gh_headers(token)
    url = f"{GITHUB_API}/repos/{repo.owner}/{repo.repo}/stargazers?per_page=100"
    dates: List[dt.date] = []

    with requests.Session() as session:
        while url:
            response = session.get(url, headers=headers, timeout=DEFAULT_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError("Unexpected GitHub API response format for stargazers")

            for item in payload:
                starred_at = item.get("starred_at")
                if starred_at:
                    dates.append(iso_to_date(starred_at))

            url = parse_link_header(response.headers.get("Link")).get("next")

    return dates
