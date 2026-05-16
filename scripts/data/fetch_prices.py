#!/usr/bin/env python3
"""Fetch daily Yahoo Finance prices for the 5-stock universe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class UniverseItem:
    ticker: str
    name: str


DEFAULT_UNIVERSE: list[UniverseItem] = [
    UniverseItem("AAPL", "Apple"),
    UniverseItem("MSFT", "Microsoft"),
    UniverseItem("V", "Visa"),
    UniverseItem("MA", "Mastercard"),
    UniverseItem("JPM", "JPMorgan Chase"),
]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    if "Date" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "Date"})
    keep_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[keep_cols]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    return df


def fetch_price_data(
    output_dir: Path,
    tickers: Iterable[str],
    start: str = "2006-01-01",
    end: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    close_frames: list[pd.Series] = []
    raw_paths: dict[str, Path] = {}

    for ticker in tickers:
        df = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            raise RuntimeError(f"No price data from Yahoo Finance for {ticker}")

        norm = _normalize(df)
        raw_path = raw_dir / f"{ticker}.csv"
        norm.to_csv(raw_path, index=False)
        raw_paths[ticker] = raw_path

        close_series = norm.set_index("Date")["Close"].rename(ticker)
        close_frames.append(close_series)

    close_df = pd.concat(close_frames, axis=1).sort_index()
    close_df = close_df.dropna(how="all")

    close_df.to_csv(output_dir / "prices_close_wide.csv", index_label="Date")
    return close_df, raw_paths
