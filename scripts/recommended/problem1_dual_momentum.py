from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DualMomentumConfig:
    momentum_lookback: int = 252
    trend_lookback: int = 200
    vol_lookback: int = 63
    top_n: int = 2
    target_vol: float = 0.16
    vol_cap: float = 0.24
    max_gross_exposure: float = 1.0
    rebalance_every: int = 21
    commission_bps_one_way: float = 10.0


def _safe_portfolio_vol(cov: pd.DataFrame, w: pd.Series) -> float:
    if cov.empty or w.empty:
        return float("nan")
    v = w.to_numpy(dtype=float)
    c = cov.to_numpy(dtype=float)
    val = float(v @ c @ v)
    if not np.isfinite(val) or val <= 0:
        return float("nan")
    return math.sqrt(val)


def build_target_weights(prices: pd.DataFrame, cfg: DualMomentumConfig) -> pd.DataFrame:
    px = prices.sort_index().copy()
    px = px.ffill().dropna(how="any")
    if px.empty:
        return pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float).fillna(0.0)

    ret = px.pct_change()
    mom = px.pct_change(cfg.momentum_lookback)
    trend_ma = px.rolling(cfg.trend_lookback).mean()
    vol = ret.rolling(cfg.vol_lookback).std(ddof=0) * math.sqrt(252.0)

    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    prev = pd.Series(0.0, index=px.columns, dtype=float)

    start_i = max(cfg.momentum_lookback, cfg.trend_lookback, cfg.vol_lookback) + 1
    for i in range(len(px)):
        if i < start_i:
            w.iloc[i] = prev
            continue
        if i % cfg.rebalance_every != 0:
            w.iloc[i] = prev
            continue

        mom_i = mom.iloc[i]
        trend_i = px.iloc[i] > trend_ma.iloc[i]
        vol_i = vol.iloc[i]
        eligible = (mom_i > 0.0) & trend_i & vol_i.notna()

        new_w = pd.Series(0.0, index=px.columns, dtype=float)
        if eligible.any():
            selected = mom_i[eligible].sort_values(ascending=False).head(cfg.top_n).index
            iv = (1.0 / vol_i[selected].clip(lower=1e-8)).replace([np.inf, -np.inf], np.nan).dropna()
            if not iv.empty:
                base = iv / iv.sum()
                ret_win = ret[selected].iloc[i - cfg.vol_lookback + 1 : i + 1].dropna(how="any")
                cov = ret_win.cov() * 252.0 if len(ret_win) > 2 else pd.DataFrame()
                port_vol = _safe_portfolio_vol(cov, base)
                if np.isfinite(port_vol) and port_vol > 0:
                    scale = min(cfg.target_vol / port_vol, cfg.vol_cap / port_vol, cfg.max_gross_exposure)
                else:
                    scale = 0.0
                new_w.loc[selected] = base * max(0.0, scale)

        prev = new_w
        w.iloc[i] = prev

    return w.reindex(prices.index).ffill().fillna(0.0)


def run_dual_momentum_portfolio_backtest(
    prices: pd.DataFrame,
    cfg: DualMomentumConfig,
    initial_capital: float = 10_000.0,
) -> dict[str, Any]:
    px = prices.sort_index().copy()
    px = px.ffill().dropna(how="any")
    if px.empty:
        return {
            "equity": pd.Series(dtype=float, name="equity"),
            "weights": pd.DataFrame(),
            "n_rebalances": 0,
            "turnover": 0.0,
        }

    target_w = build_target_weights(px, cfg)
    ret = px.pct_change().fillna(0.0)

    # Apply target weights from t-1 to returns at t (no look-ahead).
    eff_w = target_w.shift(1).fillna(0.0)
    port_ret_gross = (eff_w * ret).sum(axis=1)

    turnover = (target_w - target_w.shift(1).fillna(0.0)).abs().sum(axis=1)
    fee_rate = cfg.commission_bps_one_way / 10_000.0
    fee_drag = turnover * fee_rate

    port_ret_net = port_ret_gross - fee_drag
    equity = (1.0 + port_ret_net).cumprod() * float(initial_capital)
    equity.name = "equity"

    n_reb = int((turnover > 1e-10).sum())
    return {
        "equity": equity,
        "weights": target_w,
        "n_rebalances": n_reb,
        "turnover": float(turnover.sum()),
    }
