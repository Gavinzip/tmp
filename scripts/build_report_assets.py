#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.recommended.problem1_ga import (
    GeneticAlgorithmConfig,
    SingleStockRuleConfig,
    optimize_single_stock_ga,
    run_trend_stop_strategy,
)
from scripts.recommended.problem2_ga import PairsGeneticAlgorithmConfig, optimize_pairs_ga
from scripts.recommended.problem2_pairs import PairsConfig, describe_pair, rank_candidate_pairs

P2_FIXED_PAIR = ("V", "MA")


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
    else:
        buckets = s.index.to_period("W")

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
        rows.append((dt, shares * px + cash_left))
    return pd.Series([v for _, v in rows], index=pd.DatetimeIndex([k for k, _ in rows]), name="equity")


def summarize(equity: pd.Series) -> dict[str, float]:
    s = equity.dropna().astype(float)
    if len(s) < 2:
        return {
            "annualized_return": float("nan"),
            "cumulative_return": float("nan"),
            "max_drawdown": float("nan"),
            "volatility": float("nan"),
        }
    ret = s.pct_change().dropna()
    n = len(ret)
    ann = (s.iloc[-1] / s.iloc[0]) ** (252 / max(n, 1)) - 1
    cum = s.iloc[-1] / s.iloc[0] - 1
    running_max = s.cummax()
    dd = s / running_max - 1
    mdd = float(dd.min())
    vol = float(ret.std(ddof=0) * math.sqrt(252))
    return {
        "annualized_return": float(ann),
        "cumulative_return": float(cum),
        "max_drawdown": mdd,
        "volatility": vol,
    }


def _fit_beta(train_x: pd.Series, train_y: pd.Series) -> float:
    df = pd.concat([train_x, train_y], axis=1).dropna()
    if len(df) < 50:
        return 1.0
    x = np.log(df.iloc[:, 0].astype(float))
    y = np.log(df.iloc[:, 1].astype(float))
    model = sm.OLS(x, sm.add_constant(y)).fit()
    beta = float(model.params.iloc[1])
    return beta if np.isfinite(beta) else 1.0


@dataclass(frozen=True)
class P2Cfg:
    lookback: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.0
    stop_z: float = 3.0
    max_holding_days: int = 60
    beta_lookback: int = 252
    regime_lookback: int = 100
    use_regime_filter: bool = False
    regime_adf_pvalue_max: float = 0.20
    min_half_life_days: float = 2.0
    max_half_life_days: float = 90.0
    cooldown_days: int = 0
    initial_capital: float = 10_000.0
    fee_bps: float = 5.0


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
    if not np.isfinite(kappa) or kappa >= 0:
        return float("inf")
    return float(-np.log(2.0) / kappa)


def _regime_detail(spread: pd.Series, i: int, cfg: P2Cfg, cache: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if not cfg.use_regime_filter:
        return {"ok": True, "reason": "diagnostic_only", "half_life": float("nan"), "adf_pvalue": float("nan")}
    if i in cache:
        return cache[i]
    out: dict[str, Any] = {"ok": False, "reason": "insufficient_window", "half_life": float("nan"), "adf_pvalue": float("nan")}
    if i < cfg.regime_lookback:
        cache[i] = out
        return out
    win = spread.iloc[i - cfg.regime_lookback + 1 : i + 1].dropna()
    if len(win) < max(40, cfg.lookback):
        cache[i] = out
        return out
    hl = _estimate_half_life(win)
    out["half_life"] = hl
    if not np.isfinite(hl) or hl < cfg.min_half_life_days or hl > cfg.max_half_life_days:
        out["reason"] = "half_life_out_of_range"
        cache[i] = out
        return out
    try:
        adf_p = float(adfuller(win.to_numpy(dtype=float), autolag="AIC")[1])
    except Exception:
        out["reason"] = "adf_error"
        cache[i] = out
        return out
    out["adf_pvalue"] = adf_p
    if not np.isfinite(adf_p) or adf_p > cfg.regime_adf_pvalue_max:
        out["reason"] = "adf_pvalue_too_high"
        cache[i] = out
        return out
    out["ok"] = True
    out["reason"] = "pass"
    cache[i] = out
    return out


def run_pairs_with_logs(
    x_price: pd.Series,
    y_price: pd.Series,
    train_end: pd.Timestamp,
    cfg: P2Cfg,
) -> dict[str, Any]:
    pair = pd.concat([x_price, y_price], axis=1).dropna()
    pair.columns = ["x", "y"]
    train = pair[pair.index <= train_end]
    beta = _fit_beta(train["x"], train["y"])
    lx = np.log(pair["x"])
    ly = np.log(pair["y"])
    beta_series = _rolling_beta(lx, ly, beta, cfg.beta_lookback)
    spread = lx - beta_series * ly
    zscore = (spread - spread.rolling(cfg.lookback).mean()) / spread.rolling(cfg.lookback).std(ddof=0)
    zscore = zscore.replace([np.inf, -np.inf], np.nan)
    ret_x = pair["x"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_y = pair["y"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    equity = pd.Series(index=pair.index, dtype=float, name="equity")
    equity.iloc[0] = cfg.initial_capital
    position = pd.Series(0.0, index=pair.index, name="position")
    state = 0
    holding_days = 0
    cooldown = 0
    fee_rate = cfg.fee_bps / 10_000.0
    regime_cache: dict[int, dict[str, Any]] = {}
    trade_rows: list[dict[str, Any]] = []
    reject_rows: list[dict[str, Any]] = []
    open_trade: dict[str, Any] | None = None

    for i in range(1, len(pair)):
        dt = pair.index[i]
        prev_equity = float(equity.iloc[i - 1])
        z = float(zscore.iloc[i]) if np.isfinite(zscore.iloc[i]) else float("nan")
        prev_z = float(zscore.iloc[i - 1]) if np.isfinite(zscore.iloc[i - 1]) else float("nan")

        if not np.isfinite(z):
            equity.iloc[i] = prev_equity
            position.iloc[i] = state
            continue

        if dt <= train_end:
            equity.iloc[i] = prev_equity
            position.iloc[i] = 0.0
            continue

        if state == 0:
            if cooldown > 0:
                cooldown -= 1
                equity.iloc[i] = prev_equity
                position.iloc[i] = state
                continue

            crossed_long = np.isfinite(prev_z) and prev_z > -cfg.entry_z and z <= -cfg.entry_z
            crossed_short = np.isfinite(prev_z) and prev_z < cfg.entry_z and z >= cfg.entry_z
            if crossed_long or crossed_short:
                reg = _regime_detail(spread, i, cfg, regime_cache)
                if not reg["ok"]:
                    reject_rows.append(
                        {
                            "date": dt,
                            "side": "long" if crossed_long else "short",
                            "zscore": z,
                            "reason": reg["reason"],
                            "half_life": reg["half_life"],
                            "adf_pvalue": reg["adf_pvalue"],
                        }
                    )
                    equity.iloc[i] = prev_equity
                    position.iloc[i] = state
                    continue

            if crossed_long:
                state = +1
                holding_days = 0
                prev_equity *= (1.0 - fee_rate)
                open_trade = {"entry_date": dt, "side": "long_spread", "entry_z": z, "entry_equity": prev_equity}
            elif crossed_short:
                state = -1
                holding_days = 0
                prev_equity *= (1.0 - fee_rate)
                open_trade = {"entry_date": dt, "side": "short_spread", "entry_z": z, "entry_equity": prev_equity}
            equity.iloc[i] = prev_equity
            position.iloc[i] = state
            continue

        beta_i = float(beta_series.iloc[i]) if np.isfinite(beta_series.iloc[i]) else beta
        pair_ret_raw = ret_x.iloc[i] - beta_i * ret_y.iloc[i]
        pair_ret = (state * pair_ret_raw) / (1.0 + abs(beta_i))
        marked = prev_equity * (1.0 + pair_ret)
        holding_days += 1
        exit_reason = None
        if abs(z) <= cfg.exit_z:
            exit_reason = "mean_reversion"
        elif abs(z) >= cfg.stop_z:
            exit_reason = "stop_z"
        elif holding_days >= cfg.max_holding_days:
            exit_reason = "max_holding_days"

        if exit_reason is not None:
            marked *= (1.0 - fee_rate)
            if open_trade is not None:
                trade_rows.append(
                    {
                        "entry_date": open_trade["entry_date"],
                        "exit_date": dt,
                        "side": open_trade["side"],
                        "entry_z": open_trade["entry_z"],
                        "exit_z": z,
                        "holding_days": holding_days,
                        "exit_reason": exit_reason,
                        "entry_equity": open_trade["entry_equity"],
                        "exit_equity": marked,
                        "trade_return": marked / open_trade["entry_equity"] - 1.0,
                    }
                )
            open_trade = None
            state = 0
            holding_days = 0
            cooldown = cfg.cooldown_days
        equity.iloc[i] = max(marked, 1.0)
        position.iloc[i] = state

    if open_trade is not None:
        last_dt = pair.index[-1]
        last_z = float(zscore.iloc[-1]) if np.isfinite(zscore.iloc[-1]) else float("nan")
        last_equity = float(equity.ffill().iloc[-1])
        trade_rows.append(
            {
                "entry_date": open_trade["entry_date"],
                "exit_date": last_dt,
                "side": open_trade["side"],
                "entry_z": open_trade["entry_z"],
                "exit_z": last_z,
                "holding_days": holding_days,
                "exit_reason": "open_at_end",
                "entry_equity": open_trade["entry_equity"],
                "exit_equity": last_equity,
                "trade_return": last_equity / open_trade["entry_equity"] - 1.0,
            }
        )

    signal_df = pd.DataFrame(
        {
            "x_price": pair["x"],
            "y_price": pair["y"],
            "spread": spread,
            "zscore": zscore,
            "position": position.ffill().fillna(0.0),
            "equity": equity.ffill(),
        }
    )
    return {
        "equity": equity.ffill().dropna(),
        "zscore": zscore,
        "signal_df": signal_df.dropna(subset=["equity"]),
        "trade_log": pd.DataFrame(trade_rows),
        "reject_log": pd.DataFrame(reject_rows),
    }


def _worst_drawdown_rows(equity: pd.Series, top_n: int = 3) -> pd.DataFrame:
    s = equity.dropna().astype(float)
    if s.empty:
        return pd.DataFrame(columns=["date", "drawdown", "peak_date", "peak_value", "equity"])
    rolling_max = s.cummax()
    dd = s / rolling_max - 1.0
    worst = dd.nsmallest(top_n)
    rows = []
    for dt, val in worst.items():
        peak_dt = s.loc[:dt].idxmax()
        rows.append(
            {
                "date": dt,
                "drawdown": float(val),
                "peak_date": peak_dt,
                "peak_value": float(s.loc[peak_dt]),
                "equity": float(s.loc[dt]),
            }
        )
    return pd.DataFrame(rows)


def plot_problem1_single_stock_gallery() -> None:
    files = [
        ("AAPL", FIGURES_DIR / "reco_problem1_AAPL_vs_benchmarks.png"),
        ("MSFT", FIGURES_DIR / "reco_problem1_MSFT_vs_benchmarks.png"),
        ("V", FIGURES_DIR / "reco_problem1_V_vs_benchmarks.png"),
        ("MA", FIGURES_DIR / "reco_problem1_MA_vs_benchmarks.png"),
        ("JPM", FIGURES_DIR / "reco_problem1_JPM_vs_benchmarks.png"),
    ]
    if not all(path.exists() for _, path in files):
        return

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    for ax, (_, path) in zip(axes.ravel(), files):
        ax.imshow(plt.imread(path))
        ax.set_aspect("auto")
        ax.axis("off")
    axes.ravel()[-1].axis("off")
    fig.suptitle("Problem 1: Per-Stock Portfolio Value Lines", fontsize=18, y=0.995)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "reco_problem1_all_single_stock_lines.png", dpi=170)
    plt.close(fig)


def _rebased_test_equity(
    equity: pd.Series,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    initial_capital: float = 10_000.0,
) -> pd.Series:
    s = equity[(equity.index >= test_start) & (equity.index <= test_end)].dropna().astype(float)
    if len(s) < 2 or s.iloc[0] == 0:
        return pd.Series(dtype=float, name="equity")
    out = s / s.iloc[0] * initial_capital
    out.name = "equity"
    return out


def build_problem1_five_stock_outputs(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixed_cfg = SingleStockRuleConfig()
    ga_cfg = GeneticAlgorithmConfig()
    summary_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    ga_rows: list[dict[str, Any]] = []
    train_start = pd.Timestamp("2006-01-01")
    train_end = pd.Timestamp("2015-12-31")
    start = pd.Timestamp("2016-01-01")
    tickers = list(prices.columns)
    for seed_offset, t in enumerate(tickers, start=1):
        full = prices[t].dropna()
        train = full[(full.index >= train_start) & (full.index <= train_end)]
        s = full[full.index >= start]
        ga_result = optimize_single_stock_ga(
            train_price=train,
            ga_cfg=ga_cfg,
            seed=10_000 + seed_offset,
        )
        result = run_trend_stop_strategy(s, cfg=ga_result.config, initial_capital=10_000.0)
        fixed_result = run_trend_stop_strategy(s, cfg=fixed_cfg, initial_capital=10_000.0)
        strat = result["equity"]
        fixed = fixed_result["equity"]
        bh = buy_and_hold(s, 10_000.0)
        dca = dca_equal_budget(s, 10_000.0, "M")
        signal_df = result["signal_df"].copy()
        signal_df["ticker"] = t
        signal_df["strategy"] = "ga_trend_stop_momentum"
        signal_rows.append(signal_df.reset_index(names="date"))

        metrics_by_name: dict[str, dict[str, float]] = {}
        for name, eq in [
            ("ga_trend_stop_momentum", strat),
            ("fixed_trend_stop_momentum", fixed),
            ("buy_and_hold", bh),
            ("dca", dca),
        ]:
            m = summarize(eq)
            metrics_by_name[name] = m
            summary_rows.append({"ticker": t, "strategy": name, **m})

        ga_rows.append(
            {
                "ticker": t,
                "train_start": train.index[0].date().isoformat(),
                "train_end": train.index[-1].date().isoformat(),
                "test_start": s.index[0].date().isoformat(),
                "test_end": s.index[-1].date().isoformat(),
                "ga_trailing_stop_pct": ga_result.config.trailing_stop_pct,
                "ga_reentry_ma": ga_result.config.reentry_ma,
                "ga_momentum_lookback": ga_result.config.momentum_lookback,
                "fixed_trailing_stop_pct": fixed_cfg.trailing_stop_pct,
                "fixed_reentry_ma": fixed_cfg.reentry_ma,
                "fixed_momentum_lookback": fixed_cfg.momentum_lookback,
                "train_fitness": ga_result.fitness,
                "train_annualized_return": ga_result.train_metrics["annualized_return"],
                "train_cumulative_return": ga_result.train_metrics["cumulative_return"],
                "train_max_drawdown": ga_result.train_metrics["max_drawdown"],
                "train_volatility": ga_result.train_metrics["volatility"],
                "train_turnover": ga_result.turnover,
                "train_avg_exposure": ga_result.avg_exposure,
                "test_ga_annualized_return": metrics_by_name["ga_trend_stop_momentum"]["annualized_return"],
                "test_ga_cumulative_return": metrics_by_name["ga_trend_stop_momentum"]["cumulative_return"],
                "test_ga_max_drawdown": metrics_by_name["ga_trend_stop_momentum"]["max_drawdown"],
                "test_ga_volatility": metrics_by_name["ga_trend_stop_momentum"]["volatility"],
                "test_fixed_annualized_return": metrics_by_name["fixed_trend_stop_momentum"]["annualized_return"],
                "test_fixed_cumulative_return": metrics_by_name["fixed_trend_stop_momentum"]["cumulative_return"],
                "test_fixed_max_drawdown": metrics_by_name["fixed_trend_stop_momentum"]["max_drawdown"],
                "test_fixed_volatility": metrics_by_name["fixed_trend_stop_momentum"]["volatility"],
                "test_cumulative_delta_vs_fixed": metrics_by_name["ga_trend_stop_momentum"]["cumulative_return"]
                - metrics_by_name["fixed_trend_stop_momentum"]["cumulative_return"],
                "test_mdd_delta_vs_fixed": metrics_by_name["ga_trend_stop_momentum"]["max_drawdown"]
                - metrics_by_name["fixed_trend_stop_momentum"]["max_drawdown"],
                "ga_beats_fixed_cumulative": metrics_by_name["ga_trend_stop_momentum"]["cumulative_return"]
                > metrics_by_name["fixed_trend_stop_momentum"]["cumulative_return"],
                "ga_beats_dca_cumulative": metrics_by_name["ga_trend_stop_momentum"]["cumulative_return"]
                > metrics_by_name["dca"]["cumulative_return"],
                "ga_beats_buy_hold_cumulative": metrics_by_name["ga_trend_stop_momentum"]["cumulative_return"]
                > metrics_by_name["buy_and_hold"]["cumulative_return"],
                "ga_population_size": ga_result.population_size,
                "ga_generations": ga_result.generations,
            }
        )

        plt.figure(figsize=(11, 5))
        plt.plot(strat.index, strat.values, label=f"{t} GA Trend-Stop Strategy", linewidth=2.0)
        plt.plot(fixed.index, fixed.values, label=f"{t} Previous Fixed Rule", linewidth=1.5, linestyle="--")
        plt.plot(bh.index, bh.values, label=f"{t} Buy&Hold", linewidth=1.5)
        plt.plot(dca.index, dca.values, label=f"{t} DCA", linewidth=1.5)
        plt.title(f"Problem 1 ({t}, since 2016): GA Strategy vs Fixed Rule and Benchmarks")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value (USD)")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"reco_problem1_{t}_vs_benchmarks.png", dpi=160)
        plt.close()

    plot_problem1_single_stock_gallery()

    summary_df = pd.DataFrame(summary_rows)
    signal_df_all = pd.concat(signal_rows, ignore_index=True) if signal_rows else pd.DataFrame()
    ga_df = pd.DataFrame(ga_rows)
    return summary_df, signal_df_all, ga_df


def build_problem1_ga_walkforward(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Expanding walk-forward test for the single-stock GA rule.

    Data starts in 2006, so the first test year is 2011 after a five-year
    training history. Each test year uses parameters selected only from data
    available before that year.
    """

    fixed_cfg = SingleStockRuleConfig()
    ga_cfg = GeneticAlgorithmConfig(population_size=8, generations=5, elite_count=2)
    detail_rows: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []
    first_year = 2011
    last_year = int(prices.index.max().year)

    for ticker_idx, ticker in enumerate(prices.columns, start=1):
        full = prices[ticker].dropna().astype(float)
        if full.empty:
            continue
        first_date = full.index.min()
        for test_year in range(first_year, last_year + 1):
            train_end = pd.Timestamp(f"{test_year - 1}-12-31")
            test_start = pd.Timestamp(f"{test_year}-01-01")
            test_end = min(pd.Timestamp(f"{test_year}-12-31"), full.index.max())
            train = full[(full.index >= first_date) & (full.index <= train_end)]
            test = full[(full.index >= test_start) & (full.index <= test_end)]
            if len(train) < 252 * 4 or len(test) < 30:
                continue
            print(f"P1 walk-forward GA: {ticker} train<= {train_end.date()} test {test_year}", flush=True)

            ga_result = optimize_single_stock_ga(
                train_price=train,
                ga_cfg=ga_cfg,
                seed=30_000 + ticker_idx * 100 + test_year,
            )
            live_window = full[(full.index >= first_date) & (full.index <= test_end)]
            ga_full = run_trend_stop_strategy(live_window, cfg=ga_result.config, initial_capital=10_000.0)["equity"]
            fixed_full = run_trend_stop_strategy(live_window, cfg=fixed_cfg, initial_capital=10_000.0)["equity"]
            ga_test = _rebased_test_equity(ga_full, test_start, test_end)
            fixed_test = _rebased_test_equity(fixed_full, test_start, test_end)
            bh_test = buy_and_hold(test, 10_000.0)
            dca_test = dca_equal_budget(test, 10_000.0, "M")

            period = "pre_2016_internal_wf" if test_year <= 2015 else "post_2016_oos_wf"
            metrics_by_name: dict[str, dict[str, float]] = {}
            for strategy_name, eq in [
                ("ga_walkforward", ga_test),
                ("fixed_rule", fixed_test),
                ("buy_and_hold", bh_test),
                ("dca", dca_test),
            ]:
                m = summarize(eq)
                metrics_by_name[strategy_name] = m
                detail_rows.append(
                    {
                        "ticker": ticker,
                        "test_year": test_year,
                        "period": period,
                        "strategy": strategy_name,
                        "train_start": train.index[0].date().isoformat(),
                        "train_end": train.index[-1].date().isoformat(),
                        "test_start": test.index[0].date().isoformat(),
                        "test_end": test.index[-1].date().isoformat(),
                        **m,
                    }
                )

            param_rows.append(
                {
                    "ticker": ticker,
                    "test_year": test_year,
                    "period": period,
                    "train_start": train.index[0].date().isoformat(),
                    "train_end": train.index[-1].date().isoformat(),
                    "test_start": test.index[0].date().isoformat(),
                    "test_end": test.index[-1].date().isoformat(),
                    "ga_trailing_stop_pct": ga_result.config.trailing_stop_pct,
                    "ga_reentry_ma": ga_result.config.reentry_ma,
                    "ga_momentum_lookback": ga_result.config.momentum_lookback,
                    "train_fitness": ga_result.fitness,
                    "train_annualized_return": ga_result.train_metrics["annualized_return"],
                    "train_cumulative_return": ga_result.train_metrics["cumulative_return"],
                    "train_max_drawdown": ga_result.train_metrics["max_drawdown"],
                    "train_volatility": ga_result.train_metrics["volatility"],
                    "test_ga_cumulative_return": metrics_by_name["ga_walkforward"]["cumulative_return"],
                    "test_fixed_cumulative_return": metrics_by_name["fixed_rule"]["cumulative_return"],
                    "test_dca_cumulative_return": metrics_by_name["dca"]["cumulative_return"],
                    "test_buy_hold_cumulative_return": metrics_by_name["buy_and_hold"]["cumulative_return"],
                    "ga_beats_fixed_cumulative": metrics_by_name["ga_walkforward"]["cumulative_return"]
                    > metrics_by_name["fixed_rule"]["cumulative_return"],
                    "ga_beats_dca_cumulative": metrics_by_name["ga_walkforward"]["cumulative_return"]
                    > metrics_by_name["dca"]["cumulative_return"],
                    "ga_beats_buy_hold_cumulative": metrics_by_name["ga_walkforward"]["cumulative_return"]
                    > metrics_by_name["buy_and_hold"]["cumulative_return"],
                    "ga_population_size": ga_result.population_size,
                    "ga_generations": ga_result.generations,
                }
            )

    detail_df = pd.DataFrame(detail_rows)
    params_df = pd.DataFrame(param_rows)
    if detail_df.empty:
        summary_df = pd.DataFrame()
    else:
        summary_df = (
            detail_df.groupby(["ticker", "period", "strategy"], as_index=False)[
                ["annualized_return", "cumulative_return", "max_drawdown", "volatility"]
            ]
            .mean()
            .sort_values(["ticker", "period", "strategy"])
        )

        post = detail_df[detail_df["period"] == "post_2016_oos_wf"]
        if not post.empty:
            pivot = post.groupby(["ticker", "strategy"], as_index=False)["cumulative_return"].mean()
            pivot = pivot.pivot(index="ticker", columns="strategy", values="cumulative_return")
            ax = pivot[["ga_walkforward", "fixed_rule", "buy_and_hold", "dca"]].plot(
                kind="bar",
                figsize=(11, 5),
                width=0.78,
            )
            ax.set_title("Problem 1 GA Walk-Forward (2016-2026): Mean Yearly Cumulative Return")
            ax.set_xlabel("Ticker")
            ax.set_ylabel("Mean yearly cumulative return")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(loc="best")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "reco_problem1_ga_walkforward_cumulative.png", dpi=160)
            plt.close()

    return detail_df, summary_df, params_df


def _p2cfg_from_pairs_config(cfg: PairsConfig) -> P2Cfg:
    return P2Cfg(
        lookback=cfg.lookback,
        entry_z=cfg.entry_z,
        exit_z=cfg.exit_z,
        stop_z=cfg.stop_z,
        max_holding_days=cfg.max_holding_days,
        beta_lookback=cfg.beta_lookback,
        regime_lookback=cfg.regime_lookback,
        use_regime_filter=cfg.use_regime_filter,
        regime_adf_pvalue_max=cfg.regime_adf_pvalue_max,
        min_half_life_days=cfg.min_half_life_days,
        max_half_life_days=cfg.max_half_life_days,
        cooldown_days=cfg.cooldown_days,
        initial_capital=cfg.initial_capital,
        fee_bps=cfg.fee_bps,
    )


def _filter_trade_log_to_test_window(
    log: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> pd.DataFrame:
    if log.empty:
        return log
    out = log.copy()
    out["entry_date"] = pd.to_datetime(out["entry_date"])
    out["exit_date"] = pd.to_datetime(out["exit_date"])
    return out[(out["entry_date"] >= test_start) & (out["entry_date"] <= test_end)].copy()


def _filter_reject_log_to_test_window(
    log: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> pd.DataFrame:
    if log.empty:
        return log
    out = log.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out[(out["date"] >= test_start) & (out["date"] <= test_end)].copy()


def _pair_benchmarks_for_test(
    pair_px: pd.DataFrame,
    a: str,
    b: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> tuple[pd.Series, pd.Series]:
    pair_test = pair_px[(pair_px.index >= test_start) & (pair_px.index <= test_end)].dropna()
    if len(pair_test) < 2:
        empty = pd.Series(dtype=float, name="equity")
        return empty, empty

    bh_a = buy_and_hold(pair_test[a], initial_capital=10_000.0)
    bh_b = buy_and_hold(pair_test[b], initial_capital=10_000.0)
    bh_pair = (bh_a.reindex(pair_test.index).ffill() + bh_b.reindex(pair_test.index).ffill()) / 2.0
    bh_pair.name = "equity"

    dca_a = dca_equal_budget(pair_test[a], initial_capital=10_000.0, frequency="M")
    dca_b = dca_equal_budget(pair_test[b], initial_capital=10_000.0, frequency="M")
    dca_pair = (dca_a.reindex(pair_test.index).ffill() + dca_b.reindex(pair_test.index).ffill()) / 2.0
    dca_pair.name = "equity"
    return bh_pair, dca_pair


def _parse_selected_pairs(selected_pairs: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in str(selected_pairs).split(","):
        item = item.strip()
        if not item or "-" not in item:
            continue
        a, b = item.split("-", 1)
        out.append((a, b))
    return out


def _p2cfg_from_param_row(row: pd.Series) -> P2Cfg:
    raw_use_filter = row.get("use_regime_filter", False)
    if isinstance(raw_use_filter, str):
        use_filter = raw_use_filter.strip().lower() in {"1", "true", "yes"}
    else:
        use_filter = bool(raw_use_filter)
    return P2Cfg(
        lookback=int(row["lookback"]),
        entry_z=float(row["entry_z"]),
        exit_z=float(row["exit_z"]),
        stop_z=float(row["stop_z"]),
        max_holding_days=int(row["max_holding_days"]),
        use_regime_filter=use_filter,
        regime_adf_pvalue_max=float(row["regime_adf_pvalue_max"]),
        max_half_life_days=float(row["max_half_life_days"]),
    )


def _rank_pairs_for_selection(train_prices: pd.DataFrame) -> list[Any]:
    screen_cfg = PairsConfig(min_abs_corr=0.55, max_coint_pvalue=0.35)
    return rank_candidate_pairs(train_prices, cfg=screen_cfg)


def plot_problem2_price_position_chart(
    prices: pd.DataFrame,
    label: str,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    a: str,
    b: str,
    fixed_cfg: P2Cfg,
    ga_cfg: P2Cfg | None,
) -> None:
    pair_px = prices[[a, b]].dropna()
    pair_px = pair_px[pair_px.index <= test_end]
    pair_test = pair_px[(pair_px.index >= test_start) & (pair_px.index <= test_end)].dropna()
    if len(pair_test) < 2:
        return

    fixed_result = run_pairs_with_logs(pair_px[a], pair_px[b], train_end=train_end, cfg=fixed_cfg)
    fixed_eq = _rebased_test_equity(fixed_result["equity"], test_start, test_end)
    fixed_sig = fixed_result["signal_df"].loc[test_start:test_end]
    ga_eq = pd.Series(dtype=float, name="equity")
    ga_sig = pd.DataFrame()
    if ga_cfg is not None:
        ga_result = run_pairs_with_logs(pair_px[a], pair_px[b], train_end=train_end, cfg=ga_cfg)
        ga_eq = _rebased_test_equity(ga_result["equity"], test_start, test_end)
        ga_sig = ga_result["signal_df"].loc[test_start:test_end]

    bh_pair, dca_pair = _pair_benchmarks_for_test(pair_px, a, b, test_start, test_end)
    norm = pair_test[[a, b]].divide(pair_test[[a, b]].iloc[0]).multiply(100.0)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.25, 1.0]},
    )

    axes[0].plot(norm.index, norm[a], label=f"{a} price (rebased=100)", linewidth=1.8)
    axes[0].plot(norm.index, norm[b], label=f"{b} price (rebased=100)", linewidth=1.8)
    axes[0].set_ylabel("Price index")
    axes[0].set_title(f"Problem 2 {label} {a}-{b}: Price, Portfolio Value, Z-score and Position")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    if not ga_eq.empty:
        axes[1].plot(ga_eq.index, ga_eq.values, label="GA pairs strategy", linewidth=2.2)
    axes[1].plot(fixed_eq.index, fixed_eq.values, label="Fixed pairs strategy", linewidth=1.8, linestyle="--")
    axes[1].plot(bh_pair.index, bh_pair.values, label="Buy&Hold pair 50/50", linewidth=1.5)
    axes[1].plot(dca_pair.index, dca_pair.values, label="DCA pair 50/50", linewidth=1.5)
    axes[1].set_ylabel("Portfolio value")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    z_source = ga_sig if not ga_sig.empty else fixed_sig
    z_label = "GA z-score" if not ga_sig.empty else "Fixed z-score"
    entry = ga_cfg.entry_z if ga_cfg is not None else fixed_cfg.entry_z
    exit_z = ga_cfg.exit_z if ga_cfg is not None else fixed_cfg.exit_z
    axes[2].plot(z_source.index, z_source["zscore"], label=z_label, linewidth=1.5, color="#345995")
    axes[2].axhline(entry, color="#b23a48", linestyle="--", linewidth=1.0, label="+entry")
    axes[2].axhline(-entry, color="#b23a48", linestyle="--", linewidth=1.0, label="-entry")
    axes[2].axhline(exit_z, color="#50723c", linestyle=":", linewidth=1.0, label="+exit")
    axes[2].axhline(-exit_z, color="#50723c", linestyle=":", linewidth=1.0, label="-exit")
    axes[2].set_ylabel("Z-score")
    axes[2].grid(alpha=0.25)

    pos_ax = axes[2].twinx()
    if not ga_sig.empty:
        pos_ax.step(ga_sig.index, ga_sig["position"], where="post", label="GA position", color="#111111", linewidth=1.7)
    pos_ax.step(fixed_sig.index, fixed_sig["position"], where="post", label="Fixed position", color="#d17a22", linewidth=1.4, linestyle="--")
    pos_ax.set_ylabel("Position\n(+1 long, -1 short)")
    pos_ax.set_ylim(-1.25, 1.25)

    lines, labels = axes[2].get_legend_handles_labels()
    pos_lines, pos_labels = pos_ax.get_legend_handles_labels()
    axes[2].legend(lines + pos_lines, labels + pos_labels, loc="best")
    axes[2].set_xlabel("Date")

    safe_pair = f"{a}_{b}".lower()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"reco_problem2_{label}_{safe_pair}_price_position.png", dpi=170)
    plt.close(fig)


def plot_problem2_ga_equity_overview(prices: pd.DataFrame, params_df: pd.DataFrame) -> None:
    fixed_cfg = P2Cfg()
    panels: list[tuple[str, pd.Series, pd.Series, pd.Series, pd.Series]] = []
    for _, row in params_df.iterrows():
        label = str(row["window"])
        train_end = pd.Timestamp(row["train_end"])
        test_start = pd.Timestamp(row["test_start"]) if "test_start" in row else pd.Timestamp(f"{label}-01-01")
        test_end = pd.Timestamp(row["test_end"]) if "test_end" in row else pd.Timestamp(f"{label}-12-31")
        test_end = min(test_end, prices.index.max())
        ga_cfg = _p2cfg_from_param_row(row)
        for a, b in _parse_selected_pairs(row["selected_pairs"]):
            pair_px = prices[[a, b]].dropna()
            pair_px = pair_px[pair_px.index <= test_end]
            ga_result = run_pairs_with_logs(pair_px[a], pair_px[b], train_end=train_end, cfg=ga_cfg)
            fixed_result = run_pairs_with_logs(pair_px[a], pair_px[b], train_end=train_end, cfg=fixed_cfg)
            ga_eq = _rebased_test_equity(ga_result["equity"], test_start, test_end)
            fixed_eq = _rebased_test_equity(fixed_result["equity"], test_start, test_end)
            bh_pair, dca_pair = _pair_benchmarks_for_test(pair_px, a, b, test_start, test_end)
            panels.append((f"{label} {a}-{b}", ga_eq, fixed_eq, bh_pair, dca_pair))

    if not panels:
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(12, max(4, 3.2 * len(panels))), sharex=False)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, ga_eq, fixed_eq, bh_pair, dca_pair) in zip(axes, panels):
        if not ga_eq.empty:
            ax.plot(ga_eq.index, ga_eq.values, label="GA pairs", linewidth=2.2)
        ax.plot(fixed_eq.index, fixed_eq.values, label="Fixed pairs", linewidth=1.7, linestyle="--")
        ax.plot(bh_pair.index, bh_pair.values, label="Buy&Hold pair", linewidth=1.4)
        ax.plot(dca_pair.index, dca_pair.values, label="DCA pair", linewidth=1.4)
        ax.set_title(title)
        ax.set_ylabel("Portfolio value")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Date")
    fig.suptitle("Problem 2 GA vs Fixed: Portfolio Value Lines", y=0.995)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "reco_problem2_ga_vs_fixed.png", dpi=170)
    plt.close(fig)


def plot_problem2_strategy_zoom(prices: pd.DataFrame, params_df: pd.DataFrame) -> None:
    rows = params_df[params_df["window"].astype(str) == "2016-2026"]
    if rows.empty:
        return
    row = rows.iloc[0]
    train_end = pd.Timestamp(row["train_end"])
    test_start = pd.Timestamp(row["test_start"])
    test_end = min(pd.Timestamp(row["test_end"]), prices.index.max())
    ga_cfg = _p2cfg_from_param_row(row)
    fixed_cfg = P2Cfg()
    a, b = P2_FIXED_PAIR
    pair_px = prices[[a, b]].dropna()
    pair_px = pair_px[pair_px.index <= test_end]

    ga_result = run_pairs_with_logs(pair_px[a], pair_px[b], train_end=train_end, cfg=ga_cfg)
    fixed_result = run_pairs_with_logs(pair_px[a], pair_px[b], train_end=train_end, cfg=fixed_cfg)
    ga_eq = _rebased_test_equity(ga_result["equity"], test_start, test_end)
    fixed_eq = _rebased_test_equity(fixed_result["equity"], test_start, test_end)
    if ga_eq.empty and fixed_eq.empty:
        return

    plt.figure(figsize=(11, 5))
    if not fixed_eq.empty:
        plt.plot(fixed_eq.index, fixed_eq.values, label="Fixed classic z-score pairs", linewidth=2.2)
    if not ga_eq.empty:
        plt.plot(ga_eq.index, ga_eq.values, label="GA z-score pairs", linewidth=2.0, linestyle="--")
    plt.axhline(10_000, color="#777777", linewidth=1.0, alpha=0.45)
    plt.title("Problem 2 Zoom (V-MA, 2016-2026): Pairs Strategy Only")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value (USD)")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "reco_problem2_2016-2026_v_ma_strategy_zoom.png", dpi=170)
    plt.close()


def plot_problem2_fixed_overview(prices: pd.DataFrame, selection_df: pd.DataFrame) -> None:
    cfg = P2Cfg()
    panels: list[tuple[str, pd.Series, pd.Series, pd.Series]] = []
    selected = selection_df[selection_df["selected"]].sort_values("window")
    for _, row in selected.iterrows():
        label = str(row["window"])
        train_end = pd.Timestamp(row["train_end"])
        test_start = pd.Timestamp(row["test_start"])
        raw_test_end = pd.Timestamp(row["test_end"])
        test_end = min(raw_test_end, prices.index.max())
        a = str(row["a"])
        b = str(row["b"])
        pair_px = prices[[a, b]].dropna()
        pair_px = pair_px[pair_px.index <= test_end]
        result = run_pairs_with_logs(pair_px[a], pair_px[b], train_end=train_end, cfg=cfg)
        fixed_eq = _rebased_test_equity(result["equity"], test_start, test_end)
        bh_pair, dca_pair = _pair_benchmarks_for_test(pair_px, a, b, test_start, test_end)
        panels.append((f"{label} {a}-{b}", fixed_eq, bh_pair, dca_pair))

    if not panels:
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(12, max(4.5, 3.4 * len(panels))), sharex=False)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, fixed_eq, bh_pair, dca_pair) in zip(axes, panels):
        ax.plot(fixed_eq.index, fixed_eq.values, label="Pairs strategy", linewidth=2.0)
        ax.plot(bh_pair.index, bh_pair.values, label="Buy&Hold pair", linewidth=1.4)
        ax.plot(dca_pair.index, dca_pair.values, label="DCA pair", linewidth=1.4)
        ax.set_title(title)
        ax.set_ylabel("Portfolio value")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Problem 2 Fixed Pairs Strategy: Portfolio Value Lines", y=0.995)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "reco_problem2_vs_benchmarks.png", dpi=170)
    plt.close(fig)


def build_problem2_logs(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = P2Cfg()
    windows = [
        ("2016-2026", pd.Timestamp("2015-12-31"), pd.Timestamp("2016-01-01"), pd.Timestamp("2026-12-31")),
        ("2025", pd.Timestamp("2024-12-31"), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
        ("2026", pd.Timestamp("2025-12-31"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31")),
    ]
    trade_logs: list[pd.DataFrame] = []
    reject_logs: list[pd.DataFrame] = []
    equity_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for label, train_end, test_start, test_end in windows:
        train_prices = prices[prices.index <= train_end].dropna(how="any")
        ranked = _rank_pairs_for_selection(train_prices)
        fixed_a, fixed_b = P2_FIXED_PAIR
        fixed = describe_pair(train_prices, fixed_a, fixed_b)
        selected_rank = next(
            (rank for rank, p in enumerate(ranked, start=1) if (p.a, p.b) == P2_FIXED_PAIR),
            None,
        )
        if selected_rank is None:
            raise RuntimeError(
                f"Fixed P2 pair {fixed_a}-{fixed_b} failed training screen for window {label} "
                f"(corr={fixed.corr:.3f}, coint_p={fixed.coint_pvalue:.4f})"
            )
        for rank, pair in enumerate(ranked, start=1):
            is_selected = (pair.a, pair.b) == P2_FIXED_PAIR
            selection_rows.append(
                {
                    "window": label,
                    "rank": rank,
                    "a": pair.a,
                    "b": pair.b,
                    "pair": f"{pair.a}-{pair.b}",
                    "corr_train": pair.corr,
                    "coint_pvalue_train": pair.coint_pvalue,
                    "selected": is_selected,
                    "selection_rule": "fixed_famous_pair_v_ma" if is_selected else "screened_candidate",
                    "train_start": train_prices.index[0].date().isoformat(),
                    "train_end": train_prices.index[-1].date().isoformat(),
                    "test_start": test_start.date().isoformat(),
                    "test_end": min(test_end, prices.index.max()).date().isoformat(),
                }
            )
        best = fixed
        a, b = fixed_a, fixed_b
        pair_px = prices[[a, b]].dropna()
        pair_px = pair_px[pair_px.index <= test_end]
        result = run_pairs_with_logs(pair_px[a], pair_px[b], train_end=train_end, cfg=cfg)
        eq_test = _rebased_test_equity(result["equity"], test_start, min(test_end, prices.index.max()))
        if len(eq_test) >= 2:
            m = summarize(eq_test)
            equity_rows.append(
                {
                    "window": label,
                    "pair": f"{a}-{b}",
                    "selected_rank": selected_rank,
                    "corr_train": best.corr,
                    "coint_pvalue_train": best.coint_pvalue,
                    "start_equity": float(eq_test.iloc[0]),
                    "end_equity": float(eq_test.iloc[-1]),
                    **m,
                }
            )
        tlog = _filter_trade_log_to_test_window(result["trade_log"], test_start, min(test_end, prices.index.max()))
        if not tlog.empty:
            tlog["window"] = label
            tlog["pair"] = f"{a}-{b}"
            trade_logs.append(tlog)
        rlog = _filter_reject_log_to_test_window(result["reject_log"], test_start, min(test_end, prices.index.max()))
        if not rlog.empty:
            rlog["window"] = label
            rlog["pair"] = f"{a}-{b}"
            reject_logs.append(rlog)
        plot_problem2_price_position_chart(
            prices=prices,
            label=label,
            train_end=train_end,
            test_start=test_start,
            test_end=min(test_end, prices.index.max()),
            a=a,
            b=b,
            fixed_cfg=cfg,
            ga_cfg=None,
        )

    trades_df = pd.concat(trade_logs, ignore_index=True) if trade_logs else pd.DataFrame(
        columns=["entry_date", "exit_date", "side", "entry_z", "exit_z", "holding_days", "exit_reason", "entry_equity", "exit_equity", "trade_return", "window", "pair"]
    )
    rejects_df = pd.concat(reject_logs, ignore_index=True) if reject_logs else pd.DataFrame(
        columns=["date", "side", "zscore", "reason", "half_life", "adf_pvalue", "window", "pair"]
    )
    eq_df = pd.DataFrame(equity_rows)
    selection_df = pd.DataFrame(selection_rows)
    plot_problem2_fixed_overview(prices, selection_df)
    return trades_df, rejects_df, eq_df, selection_df


def build_problem2_ga_logs(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ga_cfg = PairsGeneticAlgorithmConfig(
        population_size=18,
        generations=12,
        elite_count=4,
        fixed_pair_a=P2_FIXED_PAIR[0],
        fixed_pair_b=P2_FIXED_PAIR[1],
        use_regime_filter=False,
    )
    fixed_cfg = P2Cfg()
    windows = [
        ("2016-2026", pd.Timestamp("2015-12-31"), pd.Timestamp("2016-01-01"), pd.Timestamp("2026-12-31")),
        ("2025", pd.Timestamp("2024-12-31"), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
        ("2026", pd.Timestamp("2025-12-31"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31")),
    ]
    param_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    trade_logs: list[pd.DataFrame] = []
    reject_logs: list[pd.DataFrame] = []

    for window_idx, (label, train_end, test_start, test_end) in enumerate(windows, start=1):
        train_prices = prices[prices.index <= train_end].dropna(how="any")
        if len(train_prices) < 520:
            continue

        ga_result = optimize_pairs_ga(
            train_prices=train_prices,
            ga_cfg=ga_cfg,
            seed=50_000 + window_idx,
        )
        print(f"P2 GA: window {label} selected {ga_result.config}", flush=True)
        selected_pairs = ga_result.selected_pairs

        param_rows.append(
            {
                "window": label,
                "train_start": train_prices.index[0].date().isoformat(),
                "train_end": train_prices.index[-1].date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": min(test_end, prices.index.max()).date().isoformat(),
                "validation_start_inside_training": ga_result.validation_start,
                "validation_end_inside_training": ga_result.validation_end,
                "selected_pairs": ",".join(f"{p.a}-{p.b}" for p in selected_pairs),
                "lookback": ga_result.config.lookback,
                "entry_z": ga_result.config.entry_z,
                "exit_z": ga_result.config.exit_z,
                "stop_z": ga_result.config.stop_z,
                "max_holding_days": ga_result.config.max_holding_days,
                "use_regime_filter": ga_result.config.use_regime_filter,
                "regime_adf_pvalue_max": ga_result.config.regime_adf_pvalue_max,
                "max_half_life_days": ga_result.config.max_half_life_days,
                "min_abs_corr": ga_result.config.min_abs_corr,
                "max_coint_pvalue": ga_result.config.max_coint_pvalue,
                "train_fitness": ga_result.fitness,
                "train_annualized_return": ga_result.train_metrics["annualized_return"],
                "train_cumulative_return": ga_result.train_metrics["cumulative_return"],
                "train_max_drawdown": ga_result.train_metrics["max_drawdown"],
                "train_volatility": ga_result.train_metrics["volatility"],
                "validation_trades": ga_result.validation_trades,
                "ga_population_size": ga_result.population_size,
                "ga_generations": ga_result.generations,
            }
        )

        ga_log_cfg = _p2cfg_from_pairs_config(ga_result.config)
        for pair in selected_pairs:
            pair_px = prices[[pair.a, pair.b]].dropna()
            pair_px = pair_px[pair_px.index <= test_end]
            if len(pair_px) < 300:
                continue
            strategy_runs = [
                ("ga_pairs_zscore", ga_log_cfg),
                ("fixed_pairs_zscore", fixed_cfg),
            ]
            for strategy_name, cfg in strategy_runs:
                result = run_pairs_with_logs(pair_px[pair.a], pair_px[pair.b], train_end=train_end, cfg=cfg)
                eq_test = _rebased_test_equity(result["equity"], test_start, test_end, initial_capital=10_000.0)
                if len(eq_test) >= 2:
                    m = summarize(eq_test)
                    metric_rows.append(
                        {
                            "window": label,
                            "pair": f"{pair.a}-{pair.b}",
                            "strategy": strategy_name,
                            "corr_train": pair.corr,
                            "coint_pvalue_train": pair.coint_pvalue,
                            "start_equity": float(eq_test.iloc[0]),
                            "end_equity": float(eq_test.iloc[-1]),
                            **m,
                        }
                    )

                tlog = _filter_trade_log_to_test_window(result["trade_log"], test_start, test_end)
                if not tlog.empty:
                    tlog["window"] = label
                    tlog["pair"] = f"{pair.a}-{pair.b}"
                    tlog["strategy"] = strategy_name
                    trade_logs.append(tlog)
                rlog = _filter_reject_log_to_test_window(result["reject_log"], test_start, test_end)
                if not rlog.empty:
                    rlog["window"] = label
                    rlog["pair"] = f"{pair.a}-{pair.b}"
                    rlog["strategy"] = strategy_name
                    reject_logs.append(rlog)

            pair_test = pair_px[(pair_px.index >= test_start) & (pair_px.index <= test_end)]
            bh_a = buy_and_hold(pair_test[pair.a], initial_capital=10_000.0)
            bh_b = buy_and_hold(pair_test[pair.b], initial_capital=10_000.0)
            bh_pair = (bh_a.reindex(pair_test.index).ffill() + bh_b.reindex(pair_test.index).ffill()) / 2.0
            dca_a = dca_equal_budget(pair_test[pair.a], initial_capital=10_000.0, frequency="M")
            dca_b = dca_equal_budget(pair_test[pair.b], initial_capital=10_000.0, frequency="M")
            dca_pair = (dca_a.reindex(pair_test.index).ffill() + dca_b.reindex(pair_test.index).ffill()) / 2.0
            for strategy_name, eq in [
                ("buy_and_hold_pair", bh_pair),
                ("dca_pair", dca_pair),
            ]:
                eq_test = _rebased_test_equity(eq, test_start, test_end, initial_capital=10_000.0)
                if len(eq_test) >= 2:
                    metric_rows.append(
                        {
                            "window": label,
                            "pair": f"{pair.a}-{pair.b}",
                            "strategy": strategy_name,
                            "corr_train": pair.corr,
                            "coint_pvalue_train": pair.coint_pvalue,
                            "start_equity": float(eq_test.iloc[0]),
                            "end_equity": float(eq_test.iloc[-1]),
                            **summarize(eq_test),
                        }
                    )

    metrics_df = pd.DataFrame(metric_rows)
    params_df = pd.DataFrame(param_rows)
    if not params_df.empty:
        for _, row in params_df.iterrows():
            label = str(row["window"])
            train_end = pd.Timestamp(row["train_end"])
            test_start = pd.Timestamp(row["test_start"])
            test_end = min(pd.Timestamp(row["test_end"]), prices.index.max())
            ga_log_cfg = _p2cfg_from_param_row(row)
            for a, b in _parse_selected_pairs(row["selected_pairs"]):
                plot_problem2_price_position_chart(
                    prices=prices,
                    label=label,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    a=a,
                    b=b,
                    fixed_cfg=fixed_cfg,
                    ga_cfg=ga_log_cfg,
                )
        plot_problem2_ga_equity_overview(prices, params_df)
        plot_problem2_strategy_zoom(prices, params_df)
    trades_df = pd.concat(trade_logs, ignore_index=True) if trade_logs else pd.DataFrame(
        columns=["entry_date", "exit_date", "side", "entry_z", "exit_z", "holding_days", "exit_reason", "entry_equity", "exit_equity", "trade_return", "window", "pair", "strategy"]
    )
    rejects_df = pd.concat(reject_logs, ignore_index=True) if reject_logs else pd.DataFrame(
        columns=["date", "side", "zscore", "reason", "half_life", "adf_pvalue", "window", "pair", "strategy"]
    )
    return params_df, metrics_df, trades_df, rejects_df


def build_diagnostics_markdown(
    p1_summary: pd.DataFrame,
    p1_signals: pd.DataFrame,
    p2_trade_log: pd.DataFrame,
    p2_reject_log: pd.DataFrame,
    p2_window_metrics: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# Strategy Diagnostics (Problem 1 & 2)")
    lines.append("")
    lines.append("## Problem 1: Five-Stock Single-Asset Strategy")
    lines.append("")
    if p1_summary.empty:
        lines.append("- No P1 summary available.")
    else:
        strat = p1_summary[p1_summary["strategy"] == "ga_trend_stop_momentum"].copy()
        for _, r in strat.iterrows():
            lines.append(
                f"- `{r['ticker']}` strategy annualized return `{r['annualized_return']:.4f}`, "
                f"cumulative `{r['cumulative_return']:.4f}`, MDD `{r['max_drawdown']:.4f}`, vol `{r['volatility']:.4f}`."
            )
    lines.append("")
    lines.append("### Problem 1 unexpected drawdowns: log-based reasons")
    if p1_signals.empty:
        lines.append("- No P1 signal log available.")
    else:
        for ticker, g in p1_signals.groupby("ticker"):
            eq = g.set_index("date")["equity"]
            worst = _worst_drawdown_rows(eq, top_n=1)
            if worst.empty:
                continue
            w = worst.iloc[0]
            dt = pd.to_datetime(w["date"])
            around = g[(g["date"] >= (dt - pd.Timedelta(days=10))) & (g["date"] <= (dt + pd.Timedelta(days=10)))]
            avg_exp = around["exposure"].mean() if not around.empty else float("nan")
            lines.append(
                f"- `{ticker}` worst drawdown at `{dt.date()}` ({w['drawdown']:.2%} from peak `{pd.to_datetime(w['peak_date']).date()}`): "
                f"period avg exposure `{avg_exp:.2f}`. This happened when the stop rule was still invested, or when the strategy re-entered after momentum recovered but price later pulled back again."
            )

    lines.append("")
    lines.append("## Problem 2: Pairs Strategy")
    lines.append("")
    if not p2_window_metrics.empty:
        lines.append("### Window outcome snapshot")
        for _, r in p2_window_metrics.iterrows():
            lines.append(
                f"- `{r['window']}` `{r['pair']}` annualized `{r['annualized_return']:.4f}`, "
                f"cumulative `{r['cumulative_return']:.4f}`, MDD `{r['max_drawdown']:.4f}`."
            )
    else:
        lines.append("- No P2 window metrics available.")

    lines.append("")
    lines.append("### Problem 2 unexpected behavior: log-based reasons")
    if p2_trade_log.empty and p2_reject_log.empty:
        lines.append("- No P2 trade/reject log available.")
    else:
        if not p2_trade_log.empty:
            losing = p2_trade_log[p2_trade_log["trade_return"] < 0].sort_values("trade_return").head(5)
            if not losing.empty:
                for _, r in losing.iterrows():
                    lines.append(
                        f"- Losing trade `{r['pair']}` `{pd.to_datetime(r['entry_date']).date()} -> {pd.to_datetime(r['exit_date']).date()}` "
                        f"({r['side']}): return `{r['trade_return']:.2%}`, exit reason `{r['exit_reason']}`, holding `{int(r['holding_days'])}` days."
                    )
            else:
                weakest = p2_trade_log.sort_values("trade_return").head(5)
                for _, r in weakest.iterrows():
                    lines.append(
                        f"- Weak trade `{r['pair']}` `{pd.to_datetime(r['entry_date']).date()} -> {pd.to_datetime(r['exit_date']).date()}` "
                        f"({r['side']}): return `{r['trade_return']:.2%}`, exit reason `{r['exit_reason']}`, holding `{int(r['holding_days'])}` days."
                    )
        if not p2_reject_log.empty:
            reason_counts = p2_reject_log["reason"].value_counts()
            lines.append("- Entry blocks by regime filter:")
            for k, v in reason_counts.items():
                lines.append(f"  - `{k}`: {int(v)}")
            sample = p2_reject_log.head(5)
            for _, r in sample.iterrows():
                lines.append(
                    f"- Rejected `{r['pair']}` {pd.to_datetime(r['date']).date()} (z={r['zscore']:.3f}) because `{r['reason']}` "
                    f"(half-life={r['half_life']:.2f}, adf_p={r['adf_pvalue']:.4f})."
                )

    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    prices = pd.read_csv(DATA_DIR / "prices_close_wide.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    tickers = ["AAPL", "MSFT", "V", "MA", "JPM"]
    prices = prices[tickers].dropna(how="all")

    p1_summary, p1_signal_log, p1_ga_params = build_problem1_five_stock_outputs(prices)
    p1_wf_detail, p1_wf_summary, p1_wf_params = build_problem1_ga_walkforward(prices)
    p2_trade_log, p2_reject_log, p2_window_metrics, p2_pair_selection = build_problem2_logs(prices)
    p2_ga_params, p2_ga_metrics, p2_ga_trades, p2_ga_rejects = build_problem2_ga_logs(prices)

    p1_summary.to_csv(RESULTS_DIR / "reco_problem1_per_stock_summary.csv", index=False)
    p1_signal_log.to_csv(RESULTS_DIR / "reco_problem1_signal_log.csv", index=False)
    p1_ga_params.to_csv(RESULTS_DIR / "reco_problem1_ga_params.csv", index=False)
    p1_wf_detail.to_csv(RESULTS_DIR / "reco_problem1_ga_walkforward.csv", index=False)
    p1_wf_summary.to_csv(RESULTS_DIR / "reco_problem1_ga_walkforward_summary.csv", index=False)
    p1_wf_params.to_csv(RESULTS_DIR / "reco_problem1_ga_walkforward_params.csv", index=False)
    p2_trade_log.to_csv(RESULTS_DIR / "reco_problem2_trade_log.csv", index=False)
    p2_reject_log.to_csv(RESULTS_DIR / "reco_problem2_reject_log.csv", index=False)
    p2_window_metrics.to_csv(RESULTS_DIR / "reco_problem2_window_metrics.csv", index=False)
    p2_pair_selection.to_csv(RESULTS_DIR / "reco_problem2_pair_selection.csv", index=False)
    p2_ga_params.to_csv(RESULTS_DIR / "reco_problem2_ga_params.csv", index=False)
    p2_ga_metrics.to_csv(RESULTS_DIR / "reco_problem2_ga_window_metrics.csv", index=False)
    p2_ga_trades.to_csv(RESULTS_DIR / "reco_problem2_ga_trade_log.csv", index=False)
    p2_ga_rejects.to_csv(RESULTS_DIR / "reco_problem2_ga_reject_log.csv", index=False)

    diag_md = build_diagnostics_markdown(
        p1_summary=p1_summary,
        p1_signals=p1_signal_log,
        p2_trade_log=p2_trade_log,
        p2_reject_log=p2_reject_log,
        p2_window_metrics=p2_window_metrics,
    )
    (RESULTS_DIR / "reco_event_diagnostics.md").write_text(diag_md, encoding="utf-8")

    print(f"Saved: {RESULTS_DIR / 'reco_problem1_per_stock_summary.csv'}")
    print(f"Saved: {RESULTS_DIR / 'reco_problem1_signal_log.csv'}")
    print(f"Saved: {RESULTS_DIR / 'reco_problem1_ga_params.csv'}")
    print(f"Saved: {RESULTS_DIR / 'reco_problem1_ga_walkforward.csv'}")
    print(f"Saved: {RESULTS_DIR / 'reco_problem1_ga_walkforward_summary.csv'}")
    print(f"Saved: {RESULTS_DIR / 'reco_problem1_ga_walkforward_params.csv'}")
    print(f"Saved: {RESULTS_DIR / 'reco_problem2_trade_log.csv'}")
    print(f"Saved: {RESULTS_DIR / 'reco_problem2_reject_log.csv'}")
    print(f"Saved: {RESULTS_DIR / 'reco_problem2_window_metrics.csv'}")
    print(f"Saved: {RESULTS_DIR / 'reco_problem2_pair_selection.csv'}")
    print(f"Saved: {RESULTS_DIR / 'reco_problem2_ga_params.csv'}")
    print(f"Saved: {RESULTS_DIR / 'reco_problem2_ga_window_metrics.csv'}")
    print(f"Saved: {RESULTS_DIR / 'reco_problem2_ga_trade_log.csv'}")
    print(f"Saved: {RESULTS_DIR / 'reco_problem2_ga_reject_log.csv'}")
    print(f"Saved: {RESULTS_DIR / 'reco_event_diagnostics.md'}")
    for t in tickers:
        print(f"Saved: {FIGURES_DIR / f'reco_problem1_{t}_vs_benchmarks.png'}")


if __name__ == "__main__":
    main()
