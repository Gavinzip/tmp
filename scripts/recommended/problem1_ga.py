from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import pandas as pd

from scripts.core.metrics import summarize


@dataclass(frozen=True)
class SingleStockRuleConfig:
    momentum_lookback: int = 126
    reentry_ma: int = 100
    trailing_stop_pct: float = 0.20
    commission_bps_one_way: float = 10.0


@dataclass(frozen=True)
class GeneticAlgorithmConfig:
    population_size: int = 28
    generations: int = 22
    elite_count: int = 4
    mutation_rate: float = 0.22
    stop_min: float = 0.18
    stop_max: float = 0.35
    ma_min: int = 50
    ma_max: int = 200
    momentum_min: int = 63
    momentum_max: int = 252


@dataclass(frozen=True)
class GeneticAlgorithmResult:
    config: SingleStockRuleConfig
    fitness: float
    train_metrics: dict[str, float]
    turnover: float
    avg_exposure: float
    generations: int
    population_size: int


def run_trend_stop_strategy(
    price: pd.Series,
    cfg: SingleStockRuleConfig,
    initial_capital: float = 10_000.0,
) -> dict[str, Any]:
    s = price.dropna().astype(float)
    if s.empty:
        return {
            "equity": pd.Series(dtype=float, name="equity"),
            "signal_df": pd.DataFrame(),
        }

    ret = s.pct_change().fillna(0.0)
    momentum = s.pct_change(cfg.momentum_lookback)
    reentry_ma = s.rolling(cfg.reentry_ma).mean()

    state = 1.0
    running_peak = float(s.iloc[0])
    exposure_values: list[float] = []
    stop_values: list[float] = []
    for dt, px in s.items():
        px_f = float(px)
        if np.isfinite(px_f):
            running_peak = max(running_peak, px_f)

        drawdown_from_peak = px_f / running_peak - 1.0 if running_peak > 0 else float("nan")
        stop_values.append(drawdown_from_peak)

        ma_ok = np.isfinite(reentry_ma.loc[dt])
        mom_ok = np.isfinite(momentum.loc[dt])
        if state > 0 and ma_ok and drawdown_from_peak <= -cfg.trailing_stop_pct and px_f < float(reentry_ma.loc[dt]):
            state = 0.0
        elif state == 0.0 and ma_ok and mom_ok and px_f > float(reentry_ma.loc[dt]) and float(momentum.loc[dt]) > 0.0:
            state = 1.0
            running_peak = px_f

        exposure_values.append(state)

    exposure = pd.Series(exposure_values, index=s.index, name="exposure").astype(float)
    drawdown_from_peak = pd.Series(stop_values, index=s.index, name="drawdown_from_peak").astype(float)

    eff_exp = exposure.shift(1).fillna(0.0)
    turnover = (exposure - exposure.shift(1).fillna(0.0)).abs()
    fee = turnover * (cfg.commission_bps_one_way / 10_000.0)
    strat_ret = eff_exp * ret - fee
    equity = (1.0 + strat_ret).cumprod() * initial_capital
    equity.name = "equity"

    signal_df = pd.DataFrame(
        {
            "price": s,
            "ret": ret,
            "momentum": momentum,
            "reentry_ma": reentry_ma,
            "drawdown_from_peak": drawdown_from_peak,
            "trend_ok": (s > reentry_ma).astype(float),
            "exposure": exposure,
            "turnover": turnover,
            "strategy_ret": strat_ret,
            "equity": equity,
        }
    )
    return {"equity": equity, "signal_df": signal_df}


def optimize_single_stock_ga(
    train_price: pd.Series,
    ga_cfg: GeneticAlgorithmConfig,
    seed: int,
) -> GeneticAlgorithmResult:
    rng = random.Random(seed)
    cache: dict[tuple[int, int, int], tuple[float, dict[str, float], float, float]] = {}

    def normalize(chromosome: tuple[float, int, int]) -> tuple[int, int, int]:
        stop_pct, ma_window, mom_window = chromosome
        stop_int = int(round(max(ga_cfg.stop_min, min(ga_cfg.stop_max, stop_pct)) * 1000))
        ma = int(round(max(ga_cfg.ma_min, min(ga_cfg.ma_max, ma_window))))
        mom = int(round(max(ga_cfg.momentum_min, min(ga_cfg.momentum_max, mom_window))))
        return stop_int, ma, mom

    def decode(key: tuple[int, int, int]) -> SingleStockRuleConfig:
        stop_int, ma, mom = key
        return SingleStockRuleConfig(
            trailing_stop_pct=stop_int / 1000.0,
            reentry_ma=ma,
            momentum_lookback=mom,
        )

    def evaluate(key: tuple[int, int, int]) -> tuple[float, dict[str, float], float, float]:
        if key in cache:
            return cache[key]
        cfg = decode(key)
        result = run_trend_stop_strategy(train_price, cfg)
        equity = result["equity"]
        signals = result["signal_df"]
        metrics = summarize(equity)
        turnover = float(signals["turnover"].sum()) if not signals.empty else 0.0
        avg_exposure = float(signals["exposure"].mean()) if not signals.empty else 0.0
        split_idx = max(int(len(equity) * 0.65), 2)
        early_metrics = summarize(equity.iloc[:split_idx])
        recent_metrics = summarize(equity.iloc[split_idx - 1 :])

        def score(m: dict[str, float]) -> float:
            return (
                m["annualized_return"]
                + 0.05 * m["cumulative_return"]
                - 0.45 * abs(m["max_drawdown"])
                - 0.08 * m["volatility"]
            )

        full_score = score(metrics)
        early_score = score(early_metrics)
        recent_score = score(recent_metrics)

        # Training-only objective: reward robust behavior across the training
        # period, not just a high full-sample result.
        exposure_penalty = max(0.0, 0.45 - avg_exposure) * 0.20
        stability_penalty = abs(early_score - recent_score) * 0.12
        fitness = (
            0.45 * full_score
            + 0.30 * recent_score
            + 0.25 * min(early_score, recent_score)
            - stability_penalty
            - 0.0025 * turnover
            - exposure_penalty
        )
        if not np.isfinite(fitness):
            fitness = -1e9
        cache[key] = (float(fitness), metrics, turnover, avg_exposure)
        return cache[key]

    def random_key() -> tuple[int, int, int]:
        return normalize(
            (
                rng.uniform(ga_cfg.stop_min, ga_cfg.stop_max),
                rng.randint(ga_cfg.ma_min, ga_cfg.ma_max),
                rng.randint(ga_cfg.momentum_min, ga_cfg.momentum_max),
            )
        )

    def crossover(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
        return tuple(a[i] if rng.random() < 0.5 else b[i] for i in range(3))  # type: ignore[return-value]

    def mutate(key: tuple[int, int, int]) -> tuple[int, int, int]:
        stop, ma, mom = key
        if rng.random() < ga_cfg.mutation_rate:
            stop += int(round(rng.gauss(0, 25)))
        if rng.random() < ga_cfg.mutation_rate:
            ma += int(round(rng.gauss(0, 22)))
        if rng.random() < ga_cfg.mutation_rate:
            mom += int(round(rng.gauss(0, 32)))
        return normalize((stop / 1000.0, ma, mom))

    fixed_key = normalize((0.20, 100, 126))
    population = [fixed_key]
    while len(population) < ga_cfg.population_size:
        population.append(random_key())

    for _ in range(ga_cfg.generations):
        ranked = sorted(population, key=lambda k: evaluate(k)[0], reverse=True)
        next_population = ranked[: ga_cfg.elite_count]
        parent_pool = ranked[: max(ga_cfg.elite_count * 3, 8)]
        while len(next_population) < ga_cfg.population_size:
            p1 = rng.choice(parent_pool)
            p2 = rng.choice(parent_pool)
            child = mutate(crossover(p1, p2))
            next_population.append(child)
        population = next_population

    best = max(population, key=lambda k: evaluate(k)[0])
    fitness, train_metrics, turnover, avg_exposure = evaluate(best)
    return GeneticAlgorithmResult(
        config=decode(best),
        fitness=fitness,
        train_metrics=train_metrics,
        turnover=turnover,
        avg_exposure=avg_exposure,
        generations=ga_cfg.generations,
        population_size=ga_cfg.population_size,
    )
