from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import pandas as pd

from scripts.core.metrics import summarize
from scripts.recommended.problem2_pairs import PairsConfig, PairInfo, describe_pair, rank_candidate_pairs, run_pairs_backtest


@dataclass(frozen=True)
class PairsGeneticAlgorithmConfig:
    population_size: int = 18
    generations: int = 12
    elite_count: int = 4
    mutation_rate: float = 0.24
    lookback_min: int = 30
    lookback_max: int = 90
    entry_min: float = 1.00
    entry_max: float = 2.50
    exit_min: float = 0.00
    exit_max: float = 0.60
    stop_min: float = 2.50
    stop_max: float = 4.20
    hold_min: int = 20
    hold_max: int = 80
    adf_min: float = 0.10
    adf_max: float = 0.35
    half_life_min: float = 35.0
    half_life_max: float = 140.0
    min_abs_corr: float = 0.55
    max_coint_pvalue: float = 0.35
    top_pairs: int = 1
    fixed_pair_a: str | None = None
    fixed_pair_b: str | None = None
    use_regime_filter: bool = False


@dataclass(frozen=True)
class PairsGeneticAlgorithmResult:
    config: PairsConfig
    fitness: float
    train_metrics: dict[str, float]
    selected_pairs: list[PairInfo]
    validation_trades: int
    generations: int
    population_size: int
    validation_start: str
    validation_end: str


ChromosomeKey = tuple[int, int, int, int, int, int, int]


def optimize_pairs_ga(
    train_prices: pd.DataFrame,
    ga_cfg: PairsGeneticAlgorithmConfig,
    seed: int,
) -> PairsGeneticAlgorithmResult:
    """Optimize pairs-trading parameters using only data inside train_prices.

    The formation segment chooses candidate pairs. The later validation segment
    scores trading parameters. The final test period is not visible here.
    """

    px = train_prices.dropna(how="any").sort_index()
    if len(px) < 520:
        raise ValueError("not enough training data for pairs GA")

    split_i = int(len(px) * 0.70)
    split_i = min(max(split_i, 260), len(px) - 126)
    formation_end = px.index[split_i]
    validation = px[px.index > formation_end]
    formation = px[px.index <= formation_end]

    pair_screen_cfg = PairsConfig(
        min_abs_corr=ga_cfg.min_abs_corr,
        max_coint_pvalue=ga_cfg.max_coint_pvalue,
    )
    # The pair itself is selected from the complete training period available
    # before the test year. If a fixed pair is supplied, GA only tunes trading
    # parameters for that pair; it does not switch pairs after seeing results.
    if ga_cfg.fixed_pair_a and ga_cfg.fixed_pair_b:
        fixed_pair = describe_pair(px, ga_cfg.fixed_pair_a, ga_cfg.fixed_pair_b)
        if abs(fixed_pair.corr) < ga_cfg.min_abs_corr or fixed_pair.coint_pvalue > ga_cfg.max_coint_pvalue:
            raise ValueError(
                f"fixed pair {fixed_pair.a}-{fixed_pair.b} failed training screen "
                f"(corr={fixed_pair.corr:.3f}, coint_p={fixed_pair.coint_pvalue:.4f})"
            )
        candidate_pairs = [fixed_pair]
    else:
        candidate_pairs = rank_candidate_pairs(px, cfg=pair_screen_cfg)[: ga_cfg.top_pairs]
    if not candidate_pairs:
        raise ValueError("no cointegrated candidate pairs found inside the pairs GA formation window")

    rng = random.Random(seed)
    cache: dict[ChromosomeKey, tuple[float, dict[str, float], int]] = {}

    def normalize(raw: tuple[float, float, float, float, float, float, float]) -> ChromosomeKey:
        lookback, entry_z, exit_z, stop_z, hold_days, adf_p, max_half_life = raw
        lookback_i = int(round(max(ga_cfg.lookback_min, min(ga_cfg.lookback_max, lookback))))
        entry_i = int(round(max(ga_cfg.entry_min, min(ga_cfg.entry_max, entry_z)) * 100))
        exit_i = int(round(max(ga_cfg.exit_min, min(ga_cfg.exit_max, exit_z)) * 100))
        stop_i = int(round(max(ga_cfg.stop_min, min(ga_cfg.stop_max, stop_z)) * 100))
        hold_i = int(round(max(ga_cfg.hold_min, min(ga_cfg.hold_max, hold_days))))
        adf_i = int(round(max(ga_cfg.adf_min, min(ga_cfg.adf_max, adf_p)) * 100))
        half_life_i = int(round(max(ga_cfg.half_life_min, min(ga_cfg.half_life_max, max_half_life))))

        # Keep the trading thresholds logically ordered.
        if exit_i >= entry_i:
            exit_i = max(int(ga_cfg.exit_min * 100), entry_i - 20)
        if stop_i <= entry_i:
            stop_i = min(int(ga_cfg.stop_max * 100), entry_i + 70)
        return lookback_i, entry_i, exit_i, stop_i, hold_i, adf_i, half_life_i

    def decode(key: ChromosomeKey) -> PairsConfig:
        lookback, entry_i, exit_i, stop_i, hold_i, adf_i, half_life_i = key
        return PairsConfig(
            lookback=lookback,
            entry_z=entry_i / 100.0,
            exit_z=exit_i / 100.0,
            stop_z=stop_i / 100.0,
            max_holding_days=hold_i,
            min_abs_corr=ga_cfg.min_abs_corr,
            max_coint_pvalue=ga_cfg.max_coint_pvalue,
            use_regime_filter=ga_cfg.use_regime_filter,
            regime_adf_pvalue_max=adf_i / 100.0,
            min_half_life_days=0.0 if not ga_cfg.use_regime_filter else 2.0,
            max_half_life_days=float(half_life_i),
        )

    def evaluate(key: ChromosomeKey) -> tuple[float, dict[str, float], int]:
        if key in cache:
            return cache[key]

        cfg = decode(key)
        pair_scores: list[float] = []
        metric_rows: list[dict[str, float]] = []
        total_trades = 0
        for pair in candidate_pairs:
            pair_px = px[[pair.a, pair.b]].dropna()
            result = run_pairs_backtest(
                x_price=pair_px[pair.a],
                y_price=pair_px[pair.b],
                train_end=formation_end,
                cfg=cfg,
            )
            eq = result["equity"]
            eq_val = eq[(eq.index >= validation.index[0]) & (eq.index <= validation.index[-1])].dropna()
            if len(eq_val) < 2:
                pair_scores.append(-0.35)
                continue
            eq_val = eq_val / eq_val.iloc[0] * cfg.initial_capital
            metrics = summarize(eq_val)
            n_trades = int(result["n_trades"])
            total_trades += n_trades

            trade_penalty = 0.25 if n_trades == 0 else 0.0
            sparse_penalty = max(0, 3 - n_trades) * 0.04
            score = (
                metrics["annualized_return"]
                + 0.15 * metrics["cumulative_return"]
                - 0.45 * abs(metrics["max_drawdown"])
                - 0.05 * metrics["volatility"]
                - trade_penalty
                - sparse_penalty
            )
            if not np.isfinite(score):
                score = -1e9
            pair_scores.append(float(score))
            metric_rows.append(metrics)

        if metric_rows:
            avg_metrics = {
                k: float(np.mean([m[k] for m in metric_rows]))
                for k in ["annualized_return", "cumulative_return", "max_drawdown", "volatility"]
            }
        else:
            avg_metrics = summarize(pd.Series(dtype=float))

        fitness = float(np.mean(pair_scores)) if pair_scores else -1e9
        cache[key] = (fitness, avg_metrics, total_trades)
        return cache[key]

    def random_key() -> ChromosomeKey:
        return normalize(
            (
                rng.randint(ga_cfg.lookback_min, ga_cfg.lookback_max),
                rng.uniform(ga_cfg.entry_min, ga_cfg.entry_max),
                rng.uniform(ga_cfg.exit_min, ga_cfg.exit_max),
                rng.uniform(ga_cfg.stop_min, ga_cfg.stop_max),
                rng.randint(ga_cfg.hold_min, ga_cfg.hold_max),
                rng.uniform(ga_cfg.adf_min, ga_cfg.adf_max),
                rng.uniform(ga_cfg.half_life_min, ga_cfg.half_life_max),
            )
        )

    def crossover(a: ChromosomeKey, b: ChromosomeKey) -> ChromosomeKey:
        return tuple(a[i] if rng.random() < 0.5 else b[i] for i in range(len(a)))  # type: ignore[return-value]

    def mutate(key: ChromosomeKey) -> ChromosomeKey:
        lookback, entry_i, exit_i, stop_i, hold_i, adf_i, half_life_i = key
        if rng.random() < ga_cfg.mutation_rate:
            lookback += int(round(rng.gauss(0, 9)))
        if rng.random() < ga_cfg.mutation_rate:
            entry_i += int(round(rng.gauss(0, 18)))
        if rng.random() < ga_cfg.mutation_rate:
            exit_i += int(round(rng.gauss(0, 10)))
        if rng.random() < ga_cfg.mutation_rate:
            stop_i += int(round(rng.gauss(0, 25)))
        if rng.random() < ga_cfg.mutation_rate:
            hold_i += int(round(rng.gauss(0, 10)))
        if rng.random() < ga_cfg.mutation_rate:
            adf_i += int(round(rng.gauss(0, 4)))
        if rng.random() < ga_cfg.mutation_rate:
            half_life_i += int(round(rng.gauss(0, 18)))
        return normalize((lookback, entry_i / 100.0, exit_i / 100.0, stop_i / 100.0, hold_i, adf_i / 100.0, half_life_i))

    # Seed the GA with the clean a-priori classic baseline, not a test-period tuned value.
    fixed_key = normalize((60, 2.0, 0.0, 3.0, 60, 0.20, 90.0))
    population = [fixed_key]
    while len(population) < ga_cfg.population_size:
        population.append(random_key())

    for _ in range(ga_cfg.generations):
        ranked = sorted(population, key=lambda k: evaluate(k)[0], reverse=True)
        next_population = ranked[: ga_cfg.elite_count]
        parent_pool = ranked[: max(ga_cfg.elite_count * 3, 8)]
        while len(next_population) < ga_cfg.population_size:
            next_population.append(mutate(crossover(rng.choice(parent_pool), rng.choice(parent_pool))))
        population = next_population

    best = max(population, key=lambda k: evaluate(k)[0])
    fitness, metrics, n_trades = evaluate(best)
    return PairsGeneticAlgorithmResult(
        config=decode(best),
        fitness=fitness,
        train_metrics=metrics,
        selected_pairs=candidate_pairs,
        validation_trades=n_trades,
        generations=ga_cfg.generations,
        population_size=ga_cfg.population_size,
        validation_start=validation.index[0].date().isoformat(),
        validation_end=validation.index[-1].date().isoformat(),
    )
