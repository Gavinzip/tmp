#!/usr/bin/env python3
"""Recommended end-to-end pipeline for Term Project II.

Problem 1:
- Dual Momentum (absolute + relative)
- Trend filter (long-term MA)
- Volatility target + volatility cap
- Cross-asset rotation on the 5-stock universe

Problem 2:
- Fixed Visa-Mastercard pairs trading with z-score entry/exit

Evaluation:
- Expanding yearly temporal validation
- Compare against Buy&Hold and DCA benchmarks
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.core.metrics import summarize
from scripts.core.temporal_validation import expanding_yearly_windows
from scripts.data.fetch_prices import DEFAULT_UNIVERSE, fetch_price_data
from scripts.recommended.benchmarks import buy_and_hold, dca_equal_budget
from scripts.recommended.problem1_dual_momentum import (
    DualMomentumConfig,
    run_dual_momentum_portfolio_backtest,
)
from scripts.recommended.problem2_pairs import (
    PairsConfig,
    describe_pair,
    rank_candidate_pairs,
    run_pairs_backtest,
)

P2_FIXED_PAIR = ("V", "MA")


def _slice_test(equity: pd.Series, test_start: pd.Timestamp, test_end: pd.Timestamp) -> pd.Series:
    return equity[(equity.index >= test_start) & (equity.index <= test_end)].dropna()


def run_problem1(
    prices: pd.DataFrame,
    windows: list[dict[str, pd.Timestamp]],
    cfg: DualMomentumConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for w_idx, win in enumerate(windows, start=1):
        view = prices[(prices.index >= win["train_start"]) & (prices.index <= win["test_end"])].dropna(how="any")
        if len(view) < 320:
            continue

        strategy_result = run_dual_momentum_portfolio_backtest(
            view,
            cfg=cfg,
            initial_capital=10_000.0,
        )
        strategy_equity = strategy_result["equity"]
        weights = strategy_result["weights"]
        avg_exposure = float(weights.sum(axis=1).mean()) if not weights.empty else None
        n_rebalances = int(strategy_result["n_rebalances"])

        universe_eq = view.mean(axis=1)
        bh_equity = buy_and_hold(universe_eq, initial_capital=10_000.0)
        dca_equity = dca_equal_budget(universe_eq, initial_capital=10_000.0, frequency="M")

        strategy_map = {
            "dual_mom_trend_volcap": strategy_equity,
            "buy_and_hold": bh_equity,
            "dca": dca_equity,
        }

        for strategy_name, equity in strategy_map.items():
            test_equity = _slice_test(equity, win["test_start"], win["test_end"])
            if len(test_equity) < 2:
                continue
            metrics = summarize(test_equity)
            rows.append(
                {
                    "problem": "problem1",
                    "window_id": w_idx,
                    "ticker": "UNIVERSE_5",
                    "strategy": strategy_name,
                    "train_start": win["train_start"].date().isoformat(),
                    "train_end": win["train_end"].date().isoformat(),
                    "test_start": win["test_start"].date().isoformat(),
                    "test_end": win["test_end"].date().isoformat(),
                    "avg_exposure_test": avg_exposure if strategy_name == "dual_mom_trend_volcap" else None,
                    "n_trades_full_window": n_rebalances if strategy_name == "dual_mom_trend_volcap" else None,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def run_problem2(
    prices: pd.DataFrame,
    windows: list[dict[str, pd.Timestamp]],
    cfg: PairsConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for w_idx, win in enumerate(windows, start=1):
        train_prices = prices[(prices.index >= win["train_start"]) & (prices.index <= win["train_end"])].dropna(how="any")
        if len(train_prices) < 260:
            continue

        a, b = P2_FIXED_PAIR
        if a not in train_prices.columns or b not in train_prices.columns:
            continue
        pair = describe_pair(train_prices, a, b)

        # Problem 2 is one fixed pairs-trading design. We use the well-known
        # Visa-Mastercard pair consistently instead of switching pairs by year.
        for pair in [pair]:
            pair_prices = prices[[pair.a, pair.b]].dropna()
            pair_prices = pair_prices[
                (pair_prices.index >= win["train_start"]) & (pair_prices.index <= win["test_end"])
            ].dropna()
            if len(pair_prices) < 300:
                continue

            result = run_pairs_backtest(
                x_price=pair_prices[pair.a],
                y_price=pair_prices[pair.b],
                train_end=win["train_end"],
                cfg=cfg,
            )
            strategy_equity = result["equity"]
            n_trades = int(result["n_trades"])
            beta = float(result["beta"])

            test_pair_prices = pair_prices[
                (pair_prices.index >= win["test_start"]) & (pair_prices.index <= win["test_end"])
            ].dropna()
            if len(test_pair_prices) < 2:
                continue

            bh_a = buy_and_hold(test_pair_prices[pair.a], initial_capital=10_000.0)
            bh_b = buy_and_hold(test_pair_prices[pair.b], initial_capital=10_000.0)
            bh_pair = (bh_a.reindex(test_pair_prices.index).ffill() + bh_b.reindex(test_pair_prices.index).ffill()) / 2.0
            bh_pair.name = "equity"

            dca_a = dca_equal_budget(test_pair_prices[pair.a], initial_capital=10_000.0, frequency="M")
            dca_b = dca_equal_budget(test_pair_prices[pair.b], initial_capital=10_000.0, frequency="M")
            dca_pair = (dca_a.reindex(test_pair_prices.index).ffill() + dca_b.reindex(test_pair_prices.index).ffill()) / 2.0
            dca_pair.name = "equity"

            strategy_map = {
                "pairs_zscore": strategy_equity,
                "buy_and_hold_pair": bh_pair,
                "dca_pair": dca_pair,
            }

            for strategy_name, equity in strategy_map.items():
                test_equity = _slice_test(equity, win["test_start"], win["test_end"])
                if len(test_equity) < 2:
                    continue
                metrics = summarize(test_equity)
                rows.append(
                    {
                        "problem": "problem2",
                        "window_id": w_idx,
                        "pair": f"{pair.a}-{pair.b}",
                        "corr_train": pair.corr,
                        "coint_pvalue_train": pair.coint_pvalue,
                        "beta_train": beta,
                        "strategy": strategy_name,
                        "train_start": win["train_start"].date().isoformat(),
                        "train_end": win["train_end"].date().isoformat(),
                        "test_start": win["test_start"].date().isoformat(),
                        "test_end": win["test_end"].date().isoformat(),
                        "n_trades_full_window": n_trades if strategy_name == "pairs_zscore" else None,
                        **metrics,
                    }
                )

    return pd.DataFrame(rows)


def _plot_problem1_example(prices: pd.DataFrame, cfg: DualMomentumConfig) -> None:
    start = pd.Timestamp("2016-01-01")
    view = prices[prices.index >= start].dropna(how="any")
    strategy_result = run_dual_momentum_portfolio_backtest(
        view,
        cfg=cfg,
        initial_capital=10_000.0,
    )
    strat = strategy_result["equity"]
    universe_eq = view.mean(axis=1)
    bh = buy_and_hold(universe_eq, initial_capital=10_000.0)
    dca = dca_equal_budget(universe_eq, initial_capital=10_000.0, frequency="M")

    plt.figure(figsize=(11, 5))
    plt.plot(strat.index, strat.values, label="P1 DualMom+Trend+VolCap", linewidth=1.8)
    plt.plot(bh.index, bh.values, label="P1 Buy&Hold (Equal-Weight Universe)", linewidth=1.5)
    plt.plot(dca.index, dca.values, label="P1 DCA (Equal-Weight Universe)", linewidth=1.5)
    plt.title("Problem 1 Example (5-stock Universe, since 2016): Strategy vs Benchmarks")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (USD)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "reco_problem1_vs_benchmarks.png", dpi=160)
    plt.close()


def _plot_problem2_example(prices: pd.DataFrame, cfg: PairsConfig) -> None:
    start = pd.Timestamp("2016-01-01")
    view = prices[prices.index >= start].dropna(how="any")
    a, b = P2_FIXED_PAIR
    if a not in view.columns or b not in view.columns:
        return
    pair = describe_pair(view, a, b)
    pair_prices = view[[pair.a, pair.b]].dropna()

    result = run_pairs_backtest(
        x_price=pair_prices[pair.a],
        y_price=pair_prices[pair.b],
        train_end=pair_prices.index[int(len(pair_prices) * 0.6)],
        cfg=cfg,
    )
    strat = result["equity"]

    bh_a = buy_and_hold(pair_prices[pair.a], initial_capital=10_000.0)
    bh_b = buy_and_hold(pair_prices[pair.b], initial_capital=10_000.0)
    bh_pair = (bh_a.reindex(pair_prices.index).ffill() + bh_b.reindex(pair_prices.index).ffill()) / 2.0

    dca_a = dca_equal_budget(pair_prices[pair.a], initial_capital=10_000.0, frequency="M")
    dca_b = dca_equal_budget(pair_prices[pair.b], initial_capital=10_000.0, frequency="M")
    dca_pair = (dca_a.reindex(pair_prices.index).ffill() + dca_b.reindex(pair_prices.index).ffill()) / 2.0

    plt.figure(figsize=(11, 5))
    plt.plot(strat.index, strat.values, label=f"P2 Pairs ({pair.a}-{pair.b})", linewidth=1.8)
    plt.plot(bh_pair.index, bh_pair.values, label="P2 Buy&Hold Pair 50/50", linewidth=1.5)
    plt.plot(dca_pair.index, dca_pair.values, label="P2 DCA Pair 50/50", linewidth=1.5)
    plt.title("Problem 2 Example: Pairs Strategy vs Pair Benchmarks")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (USD)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "reco_problem2_vs_benchmarks.png", dpi=160)
    plt.close()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    tickers = [u.ticker for u in DEFAULT_UNIVERSE]
    prices, _ = fetch_price_data(DATA_DIR, tickers=tickers, start="2006-01-01")
    prices.index = pd.to_datetime(prices.index)
    windows = expanding_yearly_windows(prices.index, min_train_years=3, test_years=1)

    p1_cfg = DualMomentumConfig()
    p2_cfg = PairsConfig()

    p1_df = run_problem1(prices, windows, cfg=p1_cfg)
    p2_df = run_problem2(prices, windows, cfg=p2_cfg)

    p1_path = RESULTS_DIR / "reco_problem1_temporal_validation.csv"
    p2_path = RESULTS_DIR / "reco_problem2_temporal_validation.csv"
    summary_path = RESULTS_DIR / "reco_strategy_summary.csv"

    p1_df.to_csv(p1_path, index=False)
    p2_df.to_csv(p2_path, index=False)

    summary_frames = []
    if not p1_df.empty:
        summary_frames.append(
            p1_df.groupby(["problem", "strategy"], as_index=False)[
                [
                    "annualized_return",
                    "cumulative_return",
                    "max_drawdown",
                    "volatility",
                ]
            ].mean()
        )
    if not p2_df.empty:
        summary_frames.append(
            p2_df.groupby(["problem", "strategy"], as_index=False)[
                [
                    "annualized_return",
                    "cumulative_return",
                    "max_drawdown",
                    "volatility",
                ]
            ].mean()
        )

    if summary_frames:
        summary = pd.concat(summary_frames, ignore_index=True)
    else:
        summary = pd.DataFrame(
            columns=[
                "problem",
                "strategy",
                "annualized_return",
                "cumulative_return",
                "max_drawdown",
                "volatility",
            ]
        )
    summary.to_csv(summary_path, index=False)

    _plot_problem1_example(prices, cfg=p1_cfg)
    _plot_problem2_example(prices, cfg=p2_cfg)

    print(f"Saved: {p1_path}")
    print(f"Saved: {p2_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {FIGURES_DIR / 'reco_problem1_vs_benchmarks.png'}")
    print(f"Saved: {FIGURES_DIR / 'reco_problem2_vs_benchmarks.png'}")


if __name__ == "__main__":
    main()
