from __future__ import annotations

import pandas as pd


def buy_and_hold(price: pd.Series, initial_capital: float = 10_000.0) -> pd.Series:
    s = price.dropna()
    if s.empty:
        return pd.Series(dtype=float, name="equity")
    shares = initial_capital / s.iloc[0]
    equity = shares * s
    equity.name = "equity"
    return equity


def dca_equal_budget(
    price: pd.Series,
    initial_capital: float = 10_000.0,
    frequency: str = "M",
) -> pd.Series:
    s = price.dropna()
    if s.empty:
        return pd.Series(dtype=float, name="equity")

    if frequency.upper() == "M":
        buckets = s.index.to_period("M")
    elif frequency.upper() == "W":
        buckets = s.index.to_period("W")
    else:
        raise ValueError("frequency must be 'M' or 'W'")

    contribution_days = s.groupby(buckets).head(1).index
    if len(contribution_days) == 0:
        return buy_and_hold(s, initial_capital=initial_capital)

    amount_per_contribution = initial_capital / len(contribution_days)
    shares = 0.0
    cash_left = initial_capital
    contribution_set = set(contribution_days)

    rows = []
    for dt, px in s.items():
        if dt in contribution_set:
            invest = min(amount_per_contribution, cash_left)
            shares += invest / px
            cash_left -= invest
        equity = shares * px + cash_left
        rows.append((dt, equity))

    equity_series = pd.Series(
        [v for _, v in rows],
        index=pd.DatetimeIndex([k for k, _ in rows]),
        name="equity",
    )
    return equity_series

