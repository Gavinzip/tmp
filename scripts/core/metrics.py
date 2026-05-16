#!/usr/bin/env python3
"""Backtest metrics helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def annualized_return(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    days = (equity.index[-1] - equity.index[0]).days
    if days <= 0:
        return 0.0
    years = days / 365.25
    if years <= 0:
        return 0.0
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def cumulative_return(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def max_drawdown(equity: pd.Series) -> float:
    equity = equity.dropna()
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def annualized_volatility(daily_returns: pd.Series) -> float:
    daily_returns = daily_returns.dropna()
    if len(daily_returns) < 2:
        return 0.0
    return float(daily_returns.std(ddof=0) * math.sqrt(252.0))


def summarize(equity: pd.Series) -> dict[str, float]:
    rets = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "annualized_return": annualized_return(equity),
        "cumulative_return": cumulative_return(equity),
        "max_drawdown": max_drawdown(equity),
        "volatility": annualized_volatility(rets),
    }
