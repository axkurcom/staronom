from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

Streak = Tuple[int, int, int]
WeekStat = Tuple[Tuple[int, int], float, int]


@dataclass(frozen=True)
class RepoRef:
    owner: str
    repo: str

    @staticmethod
    def parse(value: str) -> "RepoRef":
        if "/" not in value:
            raise ValueError("Expected repository format: owner/repo")

        owner, repo = value.split("/", 1)
        owner = owner.strip()
        repo = repo.strip()
        if not owner or not repo:
            raise ValueError("Both owner and repo must be non-empty")

        return RepoRef(owner=owner, repo=repo)

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"


@dataclass(frozen=True)
class AnalysisResult:
    first_day: dt.date
    last_day: dt.date
    days: List[dt.date]
    counts: List[int]
    total_stars: int
    average_per_day: float
    median_per_day: float
    stdev_per_day: float
    nonzero_days: int
    max_day_count: int
    max_day_dates: List[dt.date]
    percentile_by_p: Dict[int, int]
    ma7: List[Optional[float]]
    ma28: List[Optional[float]]
    med7: List[Optional[float]]
    med28: List[Optional[float]]
    slope: float
    intercept: float
    r2: float
    streak_nonzero: Streak
    streak_zero: Streak
    weekly: List[WeekStat]
