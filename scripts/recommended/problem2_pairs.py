from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


@dataclass(frozen=True)
class PairInfo:
    a: str
    b: str
    corr: float
    coint_pvalue: float


@dataclass(frozen=True)
class PairsConfig:
    lookback: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.0
    stop_z: float = 3.0
    max_holding_days: int = 60
    min_abs_corr: float = 0.70
    max_coint_pvalue: float = 0.10
    beta_lookback: int = 252
    regime_lookback: int = 100
    use_regime_filter: bool = False
    regime_adf_pvalue_max: float = 0.20
    min_half_life_days: float = 2.0
    max_half_life_days: float = 90.0
    cooldown_days: int = 0
    initial_capital: float = 10_000.0
    fee_bps: float = 5.0


def describe_pair(train_prices: pd.DataFrame, a: str, b: str) -> PairInfo:
    pair = train_prices[[a, b]].dropna()
    if len(pair) < 200:
        raise ValueError(f"not enough data to evaluate pair {a}-{b}")
    corr = float(pair[a].corr(pair[b]))
    try:
        _, pvalue, _ = coint(pair[a], pair[b])
    except Exception as exc:
        raise ValueError(f"cointegration test failed for pair {a}-{b}") from exc
    if not np.isfinite(float(pvalue)):
        raise ValueError(f"cointegration p-value is not finite for pair {a}-{b}")
    return PairInfo(a=a, b=b, corr=corr, coint_pvalue=float(pvalue))


def rank_candidate_pairs(train_prices: pd.DataFrame, cfg: PairsConfig | None = None) -> list[PairInfo]:
    if cfg is None:
        cfg = PairsConfig()
    cols = list(train_prices.columns)
    out: list[PairInfo] = []
    for a, b in combinations(cols, 2):
        try:
            info = describe_pair(train_prices, a, b)
        except ValueError:
            continue
        if abs(info.corr) < cfg.min_abs_corr:
            continue
        if info.coint_pvalue > cfg.max_coint_pvalue:
            continue
        out.append(info)

    out.sort(key=lambda x: (x.coint_pvalue, -abs(x.corr)))
    return out


def _fit_beta(train_x: pd.Series, train_y: pd.Series) -> float:
    df = pd.concat([train_x, train_y], axis=1).dropna()
    if len(df) < 50:
        return 1.0
    x = np.log(df.iloc[:, 0].astype(float))
    y = np.log(df.iloc[:, 1].astype(float))
    model = sm.OLS(x, sm.add_constant(y)).fit()
    beta = float(model.params.iloc[1])
    return beta if np.isfinite(beta) else 1.0


def _rolling_beta(lx: pd.Series, ly: pd.Series, base_beta: float, lookback: int) -> pd.Series:
    cov = lx.rolling(lookback).cov(ly)
    var = ly.rolling(lookback).var(ddof=0)
    beta = cov / var.replace(0.0, np.nan)
    beta = beta.replace([np.inf, -np.inf], np.nan).fillna(base_beta)
    return beta.clip(lower=-4.0, upper=4.0)


def _estimate_half_life(spread_window: pd.Series) -> float:
    s = spread_window.dropna()
    if len(s) < 30:
        return float("nan")

    lag = s.shift(1).dropna()
    ds = s.diff().dropna()
    idx = lag.index.intersection(ds.index)
    if len(idx) < 20:
        return float("nan")

    x = lag.loc[idx].to_numpy(dtype=float)
    y = ds.loc[idx].to_numpy(dtype=float)
    x = x - np.nanmean(x)
    denom = float(np.dot(x, x))
    if not np.isfinite(denom) or denom <= 1e-12:
        return float("nan")

    kappa = float(np.dot(x, y) / denom)
    if not np.isfinite(kappa) or kappa >= 0.0:
        return float("inf")

    return float(-np.log(2.0) / kappa)


def _is_mean_reversion_regime(
    spread: pd.Series,
    i: int,
    cfg: PairsConfig,
    cache: dict[int, bool],
) -> bool:
    if not cfg.use_regime_filter:
        return True
    if i in cache:
        return cache[i]
    if i < cfg.regime_lookback:
        cache[i] = False
        return False

    win = spread.iloc[i - cfg.regime_lookback + 1 : i + 1].dropna()
    if len(win) < max(40, cfg.lookback):
        cache[i] = False
        return False

    half_life = _estimate_half_life(win)
    if not np.isfinite(half_life):
        cache[i] = False
        return False
    if half_life < cfg.min_half_life_days or half_life > cfg.max_half_life_days:
        cache[i] = False
        return False

    try:
        adf_p = float(adfuller(win.to_numpy(dtype=float), autolag="AIC")[1])
    except Exception:
        cache[i] = False
        return False

    ok = np.isfinite(adf_p) and adf_p <= cfg.regime_adf_pvalue_max
    cache[i] = bool(ok)
    return bool(ok)


def run_pairs_backtest(
    x_price: pd.Series,
    y_price: pd.Series,
    train_end: pd.Timestamp,
    cfg: PairsConfig,
) -> dict[str, Any]:
    pair = pd.concat([x_price, y_price], axis=1).dropna()
    pair.columns = ["x", "y"]
    if len(pair) < 250:
        return {
            "equity": pd.Series(dtype=float, name="equity"),
            "zscore": pd.Series(dtype=float, name="zscore"),
            "beta": 1.0,
            "n_trades": 0,
        }

    train = pair[pair.index <= train_end]
    beta = _fit_beta(train["x"], train["y"])

    lx = np.log(pair["x"])
    ly = np.log(pair["y"])
    beta_series = _rolling_beta(lx, ly, base_beta=beta, lookback=cfg.beta_lookback)
    spread = lx - beta_series * ly
    zscore = (spread - spread.rolling(cfg.lookback).mean()) / spread.rolling(cfg.lookback).std(ddof=0)
    zscore = zscore.replace([np.inf, -np.inf], np.nan)

    equity = pd.Series(index=pair.index, dtype=float, name="equity")
    equity.iloc[0] = cfg.initial_capital

    state = 0  # 0: flat, +1: long spread, -1: short spread
    trade_count = 0
    fee_rate = cfg.fee_bps / 10_000.0
    ret_x = pair["x"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_y = pair["y"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    holding_days = 0
    cooldown = 0
    regime_cache: dict[int, bool] = {}

    for i in range(1, len(pair)):
        dt = pair.index[i]
        prev_equity = float(equity.iloc[i - 1])
        z = zscore.iloc[i]
        prev_z = zscore.iloc[i - 1]

        if not np.isfinite(z):
            equity.iloc[i] = prev_equity
            continue

        # Temporal validation discipline: trade only in the out-of-sample segment.
        if dt <= train_end:
            if state != 0:
                prev_equity *= (1.0 - fee_rate)
                state = 0
                holding_days = 0
                cooldown = cfg.cooldown_days
            equity.iloc[i] = prev_equity
            continue

        if state == 0:
            if cooldown > 0:
                cooldown -= 1
                equity.iloc[i] = prev_equity
                continue

            crossed_long = np.isfinite(prev_z) and prev_z > -cfg.entry_z and z <= -cfg.entry_z
            crossed_short = np.isfinite(prev_z) and prev_z < cfg.entry_z and z >= cfg.entry_z

            if crossed_long or crossed_short:
                if not _is_mean_reversion_regime(spread, i, cfg, regime_cache):
                    equity.iloc[i] = prev_equity
                    continue

            if crossed_long:
                state = +1
                holding_days = 0
                prev_equity *= (1.0 - fee_rate)
                trade_count += 1
            elif crossed_short:
                state = -1
                holding_days = 0
                prev_equity *= (1.0 - fee_rate)
                trade_count += 1

            equity.iloc[i] = prev_equity
            continue

        beta_i = float(beta_series.iloc[i])
        if not np.isfinite(beta_i):
            beta_i = beta

        # market-neutral daily return approximation:
        # long spread: +ret_x - beta * ret_y
        # short spread: -ret_x + beta * ret_y
        pair_ret_raw = ret_x.iloc[i] - beta_i * ret_y.iloc[i]
        pair_ret = (state * pair_ret_raw) / (1.0 + abs(beta_i))
        marked = prev_equity * (1.0 + pair_ret)
        holding_days += 1

        should_exit = (
            abs(z) <= cfg.exit_z
            or abs(z) >= cfg.stop_z
            or holding_days >= cfg.max_holding_days
        )
        if should_exit:
            marked *= (1.0 - fee_rate)
            state = 0
            holding_days = 0
            cooldown = cfg.cooldown_days

        # Guardrail for pathological paths; keep equity strictly positive
        # so downstream annualized-return math remains well-defined.
        equity.iloc[i] = max(marked, 1.0)

    equity = equity.ffill().dropna()
    return {
        "equity": equity,
        "zscore": zscore,
        "beta": beta,
        "n_trades": trade_count,
    }
