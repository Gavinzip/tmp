#!/usr/bin/env python3
"""Temporal validation windows for time-series backtests."""

from __future__ import annotations

import pandas as pd


def expanding_yearly_windows(
    index: pd.DatetimeIndex,
    min_train_years: int = 3,
    test_years: int = 1,
) -> list[dict[str, pd.Timestamp]]:
    years = sorted(set(index.year.tolist()))
    if len(years) < min_train_years + test_years:
        return []

    windows: list[dict[str, pd.Timestamp]] = []
    first_year = years[0]
    last_year = years[-1]

    for train_end_year in range(first_year + min_train_years - 1, last_year - test_years + 1):
        train_start = pd.Timestamp(year=first_year, month=1, day=1)
        train_end = pd.Timestamp(year=train_end_year, month=12, day=31)
        test_start = pd.Timestamp(year=train_end_year + 1, month=1, day=1)
        test_end = pd.Timestamp(year=train_end_year + test_years, month=12, day=31)

        windows.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )

    return windows
