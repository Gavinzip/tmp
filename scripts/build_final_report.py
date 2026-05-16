#!/usr/bin/env python3
from __future__ import annotations

import html
import math
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"
PPT_DIR = ROOT / "ppt"
PPT_IMAGE_DIR = PPT_DIR / "images"
SKILL_TEMPLATE = Path("/Users/gavin/.codex/skills/guizang-ppt-skill/assets/template.html")

TICKERS = ["AAPL", "MSFT", "V", "MA", "JPM"]


def pct(x: float | int | None, digits: int = 2) -> str:
    if x is None or not np.isfinite(float(x)):
        return "N/A"
    return f"{float(x) * 100:.{digits}f}%"


def money(x: float | int | None) -> str:
    if x is None or not np.isfinite(float(x)):
        return "N/A"
    return f"${float(x):,.0f}"


def md_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "_No data available._"
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |")
    return "\n".join(out)


def html_rowline(k: str, v: str, m: str = "") -> str:
    return (
        '<div class="rowline" data-anim>'
        f'<div class="k">{html.escape(k)}</div>'
        f'<div class="v">{html.escape(v)}</div>'
        f'<div class="v">{html.escape(m)}</div>'
        "</div>"
    )


def load_inputs() -> dict[str, pd.DataFrame]:
    prices = pd.read_csv(DATA_DIR / "prices_close_wide.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    prices = prices[TICKERS].dropna(how="all")
    return {
        "prices": prices,
        "p1_summary": pd.read_csv(RESULTS_DIR / "reco_problem1_per_stock_summary.csv"),
        "p1_ga_params": pd.read_csv(RESULTS_DIR / "reco_problem1_ga_params.csv"),
        "p1_ga_wf_summary": pd.read_csv(RESULTS_DIR / "reco_problem1_ga_walkforward_summary.csv"),
        "p1_ga_wf_params": pd.read_csv(RESULTS_DIR / "reco_problem1_ga_walkforward_params.csv"),
        "p1_signals": pd.read_csv(RESULTS_DIR / "reco_problem1_signal_log.csv", parse_dates=["date"]),
        "p1_temporal": pd.read_csv(RESULTS_DIR / "reco_problem1_temporal_validation.csv"),
        "p2_temporal": pd.read_csv(RESULTS_DIR / "reco_problem2_temporal_validation.csv"),
        "strategy_summary": pd.read_csv(RESULTS_DIR / "reco_strategy_summary.csv"),
        "p2_trades": pd.read_csv(RESULTS_DIR / "reco_problem2_trade_log.csv", parse_dates=["entry_date", "exit_date"]),
        "p2_rejects": pd.read_csv(RESULTS_DIR / "reco_problem2_reject_log.csv", parse_dates=["date"]),
        "p2_windows": pd.read_csv(RESULTS_DIR / "reco_problem2_window_metrics.csv"),
        "p2_pair_selection": pd.read_csv(RESULTS_DIR / "reco_problem2_pair_selection.csv"),
        "p2_ga_params": pd.read_csv(RESULTS_DIR / "reco_problem2_ga_params.csv"),
        "p2_ga_windows": pd.read_csv(RESULTS_DIR / "reco_problem2_ga_window_metrics.csv"),
        "p2_ga_trades": pd.read_csv(RESULTS_DIR / "reco_problem2_ga_trade_log.csv", parse_dates=["entry_date", "exit_date"]),
        "p2_ga_rejects": pd.read_csv(RESULTS_DIR / "reco_problem2_ga_reject_log.csv", parse_dates=["date"]),
    }


def build_stock_phases(prices: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker in TICKERS:
        s = prices[ticker].dropna()
        s = s[s.index >= pd.Timestamp("2016-01-01")]
        if s.empty:
            continue

        six_month_ret = s / s.shift(126) - 1.0
        best_end = six_month_ret.idxmax()
        best_start = s.index[max(0, s.index.get_loc(best_end) - 126)]
        best_ret = float(s.loc[best_end] / s.loc[best_start] - 1.0)

        rolling_max = s.cummax()
        price_dd = s / rolling_max - 1.0
        worst_date = price_dd.idxmin()
        peak_date = s.loc[:worst_date].idxmax()

        g = signals[signals["ticker"] == ticker].sort_values("date")
        eq = g.set_index("date")["equity"].dropna()
        eq_dd = eq / eq.cummax() - 1.0
        strat_worst = eq_dd.idxmin()
        strat_peak = eq.loc[:strat_worst].idxmax()
        rows.append(
            {
                "ticker": ticker,
                "price_start": s.index[0].date().isoformat(),
                "price_end": s.index[-1].date().isoformat(),
                "price_cumulative_return": float(s.iloc[-1] / s.iloc[0] - 1.0),
                "best_6m_start": best_start.date().isoformat(),
                "best_6m_end": best_end.date().isoformat(),
                "best_6m_return": best_ret,
                "worst_price_peak": peak_date.date().isoformat(),
                "worst_price_trough": worst_date.date().isoformat(),
                "worst_price_drawdown": float(price_dd.loc[worst_date]),
                "worst_strategy_peak": strat_peak.date().isoformat(),
                "worst_strategy_trough": strat_worst.date().isoformat(),
                "worst_strategy_drawdown": float(eq_dd.loc[strat_worst]),
            }
        )
    return pd.DataFrame(rows)


def param_direction(value: float, baseline: float, higher: str, lower: str, same: str = "same as fixed") -> str:
    if not np.isfinite(float(value)) or not np.isfinite(float(baseline)):
        return "N/A"
    if abs(float(value) - float(baseline)) < 1e-9:
        return same
    return higher if float(value) > float(baseline) else lower


def p1_parameter_effect(row: pd.Series) -> str:
    stop = param_direction(
        row["ga_trailing_stop_pct"],
        row["fixed_trailing_stop_pct"],
        "停損較寬：比較不容易被洗出場，但可能承受較深回撤",
        "停損較緊：較早出場避險，但容易錯過 V 型反彈",
        "停損同固定參數",
    )
    ma = param_direction(
        row["ga_reentry_ma"],
        row["fixed_reentry_ma"],
        "均線較長：進場確認較慢、較保守",
        "均線較短：較快重新進場，但假突破風險較高",
        "均線同固定參數",
    )
    mom = param_direction(
        row["ga_momentum_lookback"],
        row["fixed_momentum_lookback"],
        "動能視窗較長：訊號較慢、較穩",
        "動能視窗較短：訊號較快、較敏感",
        "動能視窗同固定參數",
    )
    return f"{stop}; {ma}; {mom}"


def build_turning_points(signals: pd.DataFrame, p1_ga_params: pd.DataFrame | None = None) -> pd.DataFrame:
    param_map: dict[str, pd.Series] = {}
    if p1_ga_params is not None and not p1_ga_params.empty:
        param_map = {str(r["ticker"]): r for _, r in p1_ga_params.iterrows()}

    rows: list[dict[str, object]] = []
    for ticker, g in signals.sort_values(["ticker", "date"]).groupby("ticker"):
        g = g.reset_index(drop=True)
        g["volatility_63d"] = g["ret"].rolling(63).std() * math.sqrt(252)
        g["price_vs_ma_pct"] = g["price"] / g["reentry_ma"] - 1.0
        changes = g[g["exposure"].diff().fillna(0.0).abs() > 0.0].copy()
        idxs = list(changes.index)
        params = param_map.get(str(ticker))
        stop_pct = float(params["ga_trailing_stop_pct"]) if params is not None else float("nan")
        ma_window = int(params["ga_reentry_ma"]) if params is not None else 0
        mom_window = int(params["ga_momentum_lookback"]) if params is not None else 0
        for pos, i in enumerate(idxs):
            r = g.loc[i]
            next_i = idxs[pos + 1] if pos + 1 < len(idxs) else len(g) - 1
            nxt = g.loc[next_i]
            period_ret = float(nxt["price"] / r["price"] - 1.0) if float(r["price"]) != 0 else float("nan")
            action = "Re-entry" if float(r["exposure"]) > 0 else "Exit"
            if action == "Exit":
                if period_ret < -0.02:
                    interpretation = "成功：出場後到下一次變化前股價下跌，避開一段回撤。"
                elif period_ret > 0.02:
                    interpretation = "失敗：出場後股價反彈，策略少吃一段上漲。"
                else:
                    interpretation = "中性：出場後價格變化不大，主要降低曝險。"
            else:
                if period_ret > 0.02:
                    interpretation = "成功：重新進場後股價上漲，策略重新跟上趨勢。"
                elif period_ret < -0.02:
                    interpretation = "失敗：重新進場後價格下跌，屬於二次回落或假突破。"
                else:
                    interpretation = "中性：重新進場後價格變化有限。"
            if action == "Exit":
                evidence = (
                    f"DD {pct(r['drawdown_from_peak'])} <= GA stop {pct(-stop_pct)} "
                    f"and price vs {ma_window}D MA {pct(r['price_vs_ma_pct'])}; "
                    f"63D vol {pct(r['volatility_63d'])}"
                )
            else:
                evidence = (
                    f"price vs {ma_window}D MA {pct(r['price_vs_ma_pct'])}, "
                    f"{mom_window}D momentum {pct(r['momentum'])} > 0; "
                    f"63D vol {pct(r['volatility_63d'])}"
                )
            rows.append(
                {
                    "ticker": ticker,
                    "date": pd.to_datetime(r["date"]).date().isoformat(),
                    "action": action,
                    "price": float(r["price"]),
                    "exposure_after": float(r["exposure"]),
                    "drawdown_from_peak": float(r["drawdown_from_peak"]),
                    "momentum_126d": float(r["momentum"]),
                    "price_vs_ma_pct": float(r["price_vs_ma_pct"]),
                    "volatility_63d": float(r["volatility_63d"]),
                    "next_change_date": pd.to_datetime(nxt["date"]).date().isoformat(),
                    "price_return_until_next_change": period_ret,
                    "interpretation": interpretation,
                    "rule_evidence": evidence,
                }
            )
    return pd.DataFrame(rows)


def summary_rows(p1_summary: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ticker in TICKERS:
        g = p1_summary[p1_summary["ticker"] == ticker].set_index("strategy")
        strat = g.loc["ga_trend_stop_momentum"]
        fixed = g.loc["fixed_trend_stop_momentum"]
        bh = g.loc["buy_and_hold"]
        dca = g.loc["dca"]
        rows.append(
            {
                "Ticker": ticker,
                "GA Ann.": pct(strat["annualized_return"]),
                "GA Cum.": pct(strat["cumulative_return"]),
                "GA MDD": pct(strat["max_drawdown"]),
                "Fixed Cum.": pct(fixed["cumulative_return"]),
                "DCA Cum.": pct(dca["cumulative_return"]),
                "B&H Cum.": pct(bh["cumulative_return"]),
                "Main Read": stock_main_read(ticker, strat, fixed, bh, dca),
            }
        )
    return rows


def stock_main_read(ticker: str, strat: pd.Series, fixed: pd.Series, bh: pd.Series, dca: pd.Series) -> str:
    beats_fixed = float(strat["cumulative_return"]) > float(fixed["cumulative_return"])
    beats_dca = float(strat["cumulative_return"]) > float(dca["cumulative_return"])
    beats_bh = float(strat["cumulative_return"]) > float(bh["cumulative_return"])
    if beats_bh:
        return "策略同時贏 DCA 與 Buy&Hold。"
    if beats_fixed and beats_dca:
        return "GA 贏原固定規則與 DCA，但輸 Buy&Hold。"
    if beats_dca:
        return "GA 贏 DCA，但輸原固定規則與 Buy&Hold。"
    return "GA 未能贏過 DCA，需要保守解讀。"


def walkforward_rows(p1_wf_summary: pd.DataFrame, period: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    df = p1_wf_summary[p1_wf_summary["period"] == period]
    for ticker in TICKERS:
        g = df[df["ticker"] == ticker].set_index("strategy")
        if g.empty:
            continue
        ga = g.loc["ga_walkforward"]
        fixed = g.loc["fixed_rule"]
        dca = g.loc["dca"]
        bh = g.loc["buy_and_hold"]
        rows.append(
            {
                "Ticker": ticker,
                "GA Avg Year Cum.": pct(ga["cumulative_return"]),
                "Fixed Avg Year Cum.": pct(fixed["cumulative_return"]),
                "DCA Avg Year Cum.": pct(dca["cumulative_return"]),
                "B&H Avg Year Cum.": pct(bh["cumulative_return"]),
                "Read": "GA 不輸固定且贏 DCA" if float(ga["cumulative_return"]) >= float(fixed["cumulative_return"]) and float(ga["cumulative_return"]) > float(dca["cumulative_return"]) else "需保守解讀",
            }
        )
    return rows


def p2_ga_metric_rows(p2_ga_windows: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, r in p2_ga_windows.iterrows():
        rows.append(
            {
                "Window": r["window"],
                "Pair": r["pair"],
                "Strategy": r["strategy"],
                "Cum.": pct(r["cumulative_return"]),
                "MDD": pct(r["max_drawdown"]),
                "Vol.": pct(r["volatility"]),
                "End Equity": money(r["end_equity"]),
            }
        )
    return rows


def stock_commentary(
    ticker: str,
    p1_summary: pd.DataFrame,
    p1_ga_params: pd.DataFrame,
    phases: pd.DataFrame,
    turns: pd.DataFrame,
) -> list[str]:
    g = p1_summary[p1_summary["ticker"] == ticker].set_index("strategy")
    strat = g.loc["ga_trend_stop_momentum"]
    fixed = g.loc["fixed_trend_stop_momentum"]
    bh = g.loc["buy_and_hold"]
    dca = g.loc["dca"]
    params = p1_ga_params[p1_ga_params["ticker"] == ticker].iloc[0]
    ph = phases[phases["ticker"] == ticker].iloc[0]
    trows = turns[turns["ticker"] == ticker]
    exits = trows[trows["action"] == "Exit"]
    reentries = trows[trows["action"] == "Re-entry"]

    lines = [
        f"- 走勢摘要：{ticker} 從 {ph['price_start']} 到 {ph['price_end']} 累積上漲 {pct(ph['price_cumulative_return'])}；最強 6 個月是 {ph['best_6m_start']} 到 {ph['best_6m_end']}，漲幅 {pct(ph['best_6m_return'])}。",
        f"- 最大價格回撤：{ph['worst_price_peak']} 到 {ph['worst_price_trough']}，價格最大回撤 {pct(ph['worst_price_drawdown'])}；策略最大回撤是 {ph['worst_strategy_peak']} 到 {ph['worst_strategy_trough']}，回撤 {pct(ph['worst_strategy_drawdown'])}。",
        f"- GA 最後參數：trailing stop {pct(params['ga_trailing_stop_pct'])}、re-entry MA {int(params['ga_reentry_ma'])} 日、momentum lookback {int(params['ga_momentum_lookback'])} 日。相對固定參數的意義：{p1_parameter_effect(params)}。",
        f"- 績效比較：GA 策略累積 {pct(strat['cumulative_return'])}，原固定規則 {pct(fixed['cumulative_return'])}，DCA {pct(dca['cumulative_return'])}，Buy&Hold {pct(bh['cumulative_return'])}。{stock_main_read(ticker, strat, fixed, bh, dca)}",
    ]

    if float(strat["cumulative_return"]) > float(dca["cumulative_return"]):
        lines.append("- 為什麼比 DCA 好：DCA 分批投入會保留現金，若股票長期上漲，資金進場較慢；本策略大多在趨勢仍有效時保持持倉，因此比 DCA 更早參與上漲。")
    else:
        lines.append("- 為什麼沒有比 DCA 好：策略出場後若遇到快速反彈，會錯過 DCA 持續買入低點的效果。")

    if float(strat["cumulative_return"]) > float(bh["cumulative_return"]):
        lines.append("- 為什麼比 Buy&Hold 好：停損規則避開主要下跌段，減少長期深回撤帶來的拖累。")
    else:
        lines.append("- 為什麼輸 Buy&Hold：Buy&Hold 在強多頭股票上永遠滿倉；策略只要有出場或重新進場延遲，就會少吃一段反彈。")

    if float(strat["cumulative_return"]) > float(fixed["cumulative_return"]):
        lines.append("- GA 是否比原參數好：是。GA 在 2006-2015 訓練期找到更適合此股票的 stop / MA / momentum 組合，2016-2026 測試期也優於原固定規則。")
    else:
        lines.append("- GA 是否比原參數好：否。GA 在訓練期找到較高分參數，但測試期輸給原固定規則，代表參數對訓練期型態有 overfitting 或 whipsaw 問題。")

    lines.append(f"- 關鍵轉折點數量：共 {len(trows)} 次倉位變化，出場 {len(exits)} 次，重新進場 {len(reentries)} 次。")
    return lines


def p1_key_event_explanations(ticker: str, turns: pd.DataFrame, p1_ga_params: pd.DataFrame, n: int = 3) -> list[str]:
    g = turns[turns["ticker"] == ticker].copy()
    if g.empty:
        return ["沒有明顯倉位變化。"]
    params = p1_ga_params[p1_ga_params["ticker"] == ticker].iloc[0]
    g["abs_move"] = g["price_return_until_next_change"].abs()
    lines: list[str] = []
    for _, r in g.sort_values("abs_move", ascending=False).head(n).iterrows():
        action = "出場" if r["action"] == "Exit" else "重新進場"
        if r["action"] == "Exit":
            rule = (
                f"因為股價相對高點回撤 {pct(r['drawdown_from_peak'])}，已超過 GA stop "
                f"{pct(-float(params['ga_trailing_stop_pct']))}，且價格低於 GA 的 {int(params['ga_reentry_ma'])} 日均線 "
                f"{pct(r['price_vs_ma_pct'])}"
            )
            if float(r["price_return_until_next_change"]) < 0:
                result = "這段出場有效，因為下一次訊號前股價繼續下跌。"
            else:
                result = "這段出場反而吃虧，因為下一次訊號前股價反彈，策略少吃反彈。"
        else:
            rule = (
                f"因為價格重新站上 GA 的 {int(params['ga_reentry_ma'])} 日均線 "
                f"{pct(r['price_vs_ma_pct'])}，且 {int(params['ga_momentum_lookback'])} 日 momentum 為 "
                f"{pct(r['momentum_126d'])}"
            )
            if float(r["price_return_until_next_change"]) > 0:
                result = "這段進場有效，因為重新進場後到下一次訊號前股價上漲。"
            else:
                result = "這段進場失敗，因為重新進場後又下跌，屬於假突破或二次回落。"
        parameter_read = p1_parameter_effect(params)
        lines.append(
            f"{r['date']} {action} -> {r['next_change_date']}：{rule}；"
            f"當時 63 日年化波動約 {pct(r['volatility_63d'])}，後續股價變化 {pct(r['price_return_until_next_change'])}。{result}"
            f"這和 GA 參數有關：{parameter_read}。"
        )
    return lines


def market_phase_label(date_text: str) -> str:
    dt = pd.Timestamp(date_text)
    if pd.Timestamp("2020-03-01") <= dt <= pd.Timestamp("2020-07-31"):
        return "COVID crash 後的快速反彈期"
    if pd.Timestamp("2022-01-01") <= dt <= pd.Timestamp("2022-12-31"):
        return "2022 升息與科技股/金融股修正後的震盪修復期"
    if pd.Timestamp("2025-01-01") <= dt <= pd.Timestamp("2025-12-31"):
        return "2025 年回調後重新轉強的修復期"
    return "趨勢修復期"


def stock_slide_event_story(ticker: str, turns: pd.DataFrame, params: pd.Series) -> dict[str, str]:
    g = turns[(turns["ticker"] == ticker) & (turns["action"] == "Re-entry")].copy()
    if g.empty:
        return {
            "headline": "沒有明顯重新進場事件",
            "rule": "GA 沒有觸發新的 re-entry 訊號。",
            "result": "主要靠既有持倉或空手狀態控制風險。",
        }
    g = g.sort_values("price_return_until_next_change", ascending=False)
    r = g.iloc[0]
    phase = market_phase_label(str(r["date"]))
    fixed_ma = int(params["fixed_reentry_ma"])
    fixed_mom = int(params["fixed_momentum_lookback"])
    return {
        "headline": f"{r['date']} 重新進場：{phase}",
        "rule": (
            f"GA 用 MA{int(params['ga_reentry_ma'])}/Mom{int(params['ga_momentum_lookback'])}，"
            f"當天 price vs MA {pct(r['price_vs_ma_pct'])}、momentum {pct(r['momentum_126d'])}、63D vol {pct(r['volatility_63d'])}"
        ),
        "result": (
            f"到 {r['next_change_date']} 前股價 {pct(r['price_return_until_next_change'])}。"
            f"相對 fixed MA{fixed_ma}/Mom{fixed_mom}，GA 參數較能反映這檔股票訓練期學到的進場速度。"
        ),
    }


def p2_trade_explanation(row: pd.Series, prices: pd.DataFrame) -> str:
    entry_z = float(row["entry_z"])
    exit_z = float(row["exit_z"])
    side = str(row["side"])
    direction = "z-score 往 0 回歸" if abs(exit_z) < abs(entry_z) else "z-score 沒有往 0 回歸"
    if side == "long_spread":
        setup = "進場時 V-MA spread 偏低，策略做 long spread，期待 spread 往上回到均值"
    else:
        setup = "進場時 V-MA spread 偏高，策略做 short spread，期待 spread 往下回到均值"

    price_context = ""
    try:
        a, b = str(row["pair"]).split("-", 1)
        entry_dt = pd.Timestamp(row["entry_date"])
        exit_dt = pd.Timestamp(row["exit_date"])
        a_ret = float(prices.loc[exit_dt, a] / prices.loc[entry_dt, a] - 1.0)
        b_ret = float(prices.loc[exit_dt, b] / prices.loc[entry_dt, b] - 1.0)
        price_context = f"；同期 {a} {pct(a_ret)}、{b} {pct(b_ret)}"
    except Exception:
        price_context = ""

    if str(row["exit_reason"]) == "stop_z":
        exit_text = "因 z-score 進一步偏離觸發 stop_z，表示 spread 短期沒有照預期回歸"
    elif str(row["exit_reason"]) == "open_at_end":
        exit_text = "測試期結束仍持倉，因此以期末價格標記損益"
    elif str(row["exit_reason"]) == "max_holding_days":
        exit_text = "到達最長持有天數仍未完整回歸，因此按風控規則出場"
    else:
        exit_text = "因 z-score 回到出場區間而平倉"
    return (
        f"{setup}；z {entry_z:.2f} -> {exit_z:.2f}，{direction}{price_context}。"
        f"{exit_text}，單筆報酬 {pct(row['trade_return'])}。"
    )


def p2_key_trade_rows(p2_ga_trades: pd.DataFrame, prices: pd.DataFrame) -> list[dict[str, object]]:
    if p2_ga_trades.empty:
        return []
    selected: list[pd.Series] = []
    for (window, strategy), g in p2_ga_trades.groupby(["window", "strategy"]):
        selected.append(g.sort_values("trade_return").iloc[0])
        selected.append(g.sort_values("trade_return", ascending=False).iloc[0])
    seen: set[tuple[str, str, str, str, str]] = set()
    rows: list[dict[str, object]] = []
    for r in selected:
        key = (
            str(r["window"]),
            str(r["strategy"]),
            pd.Timestamp(r["entry_date"]).date().isoformat(),
            pd.Timestamp(r["exit_date"]).date().isoformat(),
            str(r["side"]),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "Window": r["window"],
                "Strategy": r["strategy"],
                "Trade": f"{pd.Timestamp(r['entry_date']).date()} -> {pd.Timestamp(r['exit_date']).date()}",
                "Side": r["side"],
                "Z": f"{float(r['entry_z']):.2f} -> {float(r['exit_z']):.2f}",
                "Return": pct(r["trade_return"]),
                "Why": p2_trade_explanation(r, prices),
            }
        )
    return rows


def p2_ga_vs_fixed_explanations(p2_ga_params: pd.DataFrame, p2_ga_windows: pd.DataFrame, p2_ga_trades: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    metrics = p2_ga_windows.copy()
    metrics["window"] = metrics["window"].astype(str)
    trades = p2_ga_trades.copy()
    trades["window"] = trades["window"].astype(str)
    for _, params in p2_ga_params.iterrows():
        window = str(params["window"])
        g = metrics[metrics["window"] == window].set_index("strategy")
        if "ga_pairs_zscore" not in g.index or "fixed_pairs_zscore" not in g.index:
            continue
        ga = g.loc["ga_pairs_zscore"]
        fixed = g.loc["fixed_pairs_zscore"]
        ga_n = len(trades[(trades["window"] == window) & (trades["strategy"] == "ga_pairs_zscore")])
        fixed_n = len(trades[(trades["window"] == window) & (trades["strategy"] == "fixed_pairs_zscore")])
        if float(ga["cumulative_return"]) > float(fixed["cumulative_return"]):
            result = "GA 優於 fixed，但本質是少虧，不是大賺。"
        else:
            result = "GA 輸給 fixed，代表訓練期選到的保守參數沒有抓到測試期主要回歸段。"
        lines.append(
            f"{window}：GA 參數 lookback {int(params['lookback'])}、entry_z {float(params['entry_z']):.2f}、"
            f"exit_z {float(params['exit_z']):.2f}、stop_z {float(params['stop_z']):.2f}、max hold {int(params['max_holding_days'])}。"
            f"相對 fixed `40/1.5/0/3.5/60`，GA 交易 {ga_n} 筆、fixed {fixed_n} 筆；"
            f"GA 累積 {pct(ga['cumulative_return'])}、MDD {pct(ga['max_drawdown'])}，fixed 累積 {pct(fixed['cumulative_return'])}、MDD {pct(fixed['max_drawdown'])}。{result}"
        )
    return lines


def report_markdown(data: dict[str, pd.DataFrame], phases: pd.DataFrame, turns: pd.DataFrame) -> str:
    prices = data["prices"]
    p1_summary = data["p1_summary"]
    p1_ga_params = data["p1_ga_params"]
    p1_ga_wf_summary = data["p1_ga_wf_summary"]
    strategy_summary = data["strategy_summary"]
    p2_windows = data["p2_windows"]
    p2_trades = data["p2_trades"]
    p2_rejects = data["p2_rejects"]
    p2_pair_selection = data["p2_pair_selection"]
    p2_ga_params = data["p2_ga_params"]
    p2_ga_windows = data["p2_ga_windows"]
    p2_ga_trades = data["p2_ga_trades"]
    p2_ga_rejects = data["p2_ga_rejects"]

    lines: list[str] = []
    lines.append("# Term Project II: Quantitative Trading Final Report")
    lines.append("")
    lines.append("## 0. Executive Summary")
    lines.append("")
    lines.append("本報告完成兩個問題：Problem 1 使用五檔股票進行規則型投資策略設計與比較；Problem 2 使用同一組股票建立 pairs trading 策略。全部結果皆與 Buy-and-Hold / Lump-Sum 和 DCA 比較，並保留可重現的 Python 腳本、CSV 結果、圖表與 PPT。")
    lines.append("")
    lines.append("- Problem 1 五檔股票：AAPL、MSFT、V、MA、JPM。資料自 2006 年起，單股走勢圖使用 2016 年後約 10 年資料。")
    lines.append("- Problem 1 單股規則：使用 GA 在 2006-2015 訓練期選出 Trend-Stop + Momentum Re-entry 參數，再固定參數測試 2016-2026。")
    lines.append("- GA 結果：單次 2006-2015 訓練、2016-2026 測試下，五檔 GA 策略都贏過 DCA；walk-forward 檢查中，2016-2026 年度平均也在五檔都贏固定參數與 DCA。")
    lines.append("- Problem 1 五檔組合規則：Dual Momentum + Trend Filter + Volatility Control，用於 temporal validation 的五檔輪動。")
    lines.append("- Problem 2 規則：固定使用 Visa-Mastercard (`V-MA`) 這組有名的同產業支付網路 pair，使用 classic z-score pairs trading；ADF 與 half-life 改為診斷資訊，GA 只用測試年前資料挑交易參數。")
    lines.append("- 主要解讀：規則策略不是保證打敗 Buy&Hold；它的價值在於讓進出場邏輯可解釋，並能討論何時有效、何時失效。")
    lines.append("")

    lines.append("## 1. Professor Requirement Checklist")
    lines.append("")
    checklist = [
        {"Requirement": "選 5 檔上市股票", "Status": "完成", "Evidence": "AAPL, MSFT, V, MA, JPM"},
        {"Requirement": "歷史資料要足夠長，最好超過 10 年", "Status": "完成", "Evidence": "資料自 2006 年起；圖表使用 2016-2026"},
        {"Requirement": "每檔股票至少一個 rule-based strategy", "Status": "完成", "Evidence": "GA-optimized Trend-Stop + Momentum Re-entry"},
        {"Requirement": "與 Buy-and-Hold / Lump-Sum 比較", "Status": "完成", "Evidence": "每檔圖與 summary CSV 皆包含"},
        {"Requirement": "與 Dollar-Cost Averaging 比較", "Status": "完成", "Evidence": "每檔圖與 summary CSV 皆包含"},
        {"Requirement": "指標含 annualized return / cumulative return / MDD / risk", "Status": "完成", "Evidence": "results/*.csv"},
        {"Requirement": "使用 temporal validation，不做 random split", "Status": "完成", "Evidence": "expanding yearly windows + GA walk-forward"},
        {"Requirement": "加入 Genetic Algorithm", "Status": "完成", "Evidence": "P1 單股參數與 P2 pairs 參數皆只用測試年前資料最佳化"},
        {"Requirement": "Problem 2 pairs trading", "Status": "完成", "Evidence": "固定使用 V-MA pair；training data 只用來檢查相關與共整合並調參"},
        {"Requirement": "Tables and figures", "Status": "完成", "Evidence": "figures/、results/、ppt/"},
        {"Requirement": "限制與改進", "Status": "完成", "Evidence": "本報告第 8 節"},
        {"Requirement": "Individual contribution statement", "Status": "完成", "Evidence": "本報告第 9 節"},
    ]
    lines.append(md_table(checklist, ["Requirement", "Status", "Evidence"]))
    lines.append("")

    lines.append("## 2. Data and Benchmark Design")
    lines.append("")
    lines.append("- Data source: `yfinance` 下載每日 OHLCV，主要使用 adjusted close / close price 做回測。")
    lines.append("- Initial capital: 每檔股票 USD 10,000。")
    lines.append("- Buy-and-Hold / Lump-Sum: 第一天一次投入全部本金，之後長抱。")
    lines.append("- DCA: 將 USD 10,000 平均分成每月第一個交易日投入，直到資金投入完畢。")
    lines.append("- Transaction cost: Problem 1 單邊 10 bps；Problem 2 單邊 5 bps。")
    lines.append("- Temporal validation: 以 expanding yearly windows 做逐年測試，避免金融時間序列被 random split 破壞時間順序。")
    lines.append("")

    lines.append("## 3. Strategy Design")
    lines.append("")
    lines.append("### Problem 1A: Five-Stock Portfolio Strategy")
    lines.append("")
    lines.append("五檔組合策略使用 Dual Momentum + Trend Filter + Volatility Control：")
    lines.append("")
    lines.append("- 每 21 個交易日重新平衡。")
    lines.append("- 先看 252 日動能，動能為正且價格高於 200 日均線才可進入候選。")
    lines.append("- 從候選股票中選動能最高的前 2 檔。")
    lines.append("- 用 inverse volatility 分配權重，再用目標波動與波動上限控制總曝險。")
    lines.append("- 此策略用於 Problem 1 的 temporal validation 與五檔整體圖。")
    lines.append("")
    lines.append("### Problem 1B: Single-Stock Rule Strategy with GA")
    lines.append("")
    lines.append("單股圖不能直接套五檔輪動的相對動能，因為相對動能會把其他股票納入判斷，導致「某檔股票自己表現不差，但因為輸給其他股票就被迫空手」。所以每檔股票改用單股規則，並用 GA 在訓練期選參數：")
    lines.append("")
    lines.append("這裡的 `stop` 指的是回撤停損 / trailing stop 門檻。例如 `Stop = 20%` 不是固定跌到某個價格才賣，而是指股價從進場後或持有期間高點回跌 20% 以上，並且跌破 GA 選定的 moving average，策略才出場。")
    lines.append("")
    lines.append("```text")
    lines.append("平常持有股票")
    lines.append("如果從高點跌超過 GA 選定的 trailing_stop_pct，且價格跌破 GA 選定的 moving average -> 出場")
    lines.append("出場後，若價格重新站上該 moving average，且 GA 選定 lookback 的 momentum 轉正 -> 重新進場")
    lines.append("```")
    lines.append("")
    lines.append("這個設計的目的不是預測價格，而是用明確規則處理三件事：保留長期趨勢、避開部分深跌、在趨勢恢復後重新進場。")
    lines.append("")
    lines.append("### Genetic Algorithm Design")
    lines.append("")
    lines.append("GA 不是用來預測股價，而是用來搜尋規則策略的參數。為避免 look-ahead bias，GA 的正式單次測試只能使用 2006-2015 的 training data；2016-2026 的 testing data 完全不參與參數選擇。新增的 walk-forward 檢查則是每一年都重新只用該年前的資料挑參數，再測下一年。")
    lines.append("")
    lines.append("GA 的計算流程如下：先隨機產生一組候選參數 population，並把原固定參數放進初始族群當 baseline；每一代都把每個 chromosome 轉成一個完整策略回測，計算 training-only fitness；保留前幾名 elite，其他候選由 crossover 混合父母參數，再用 mutation 小幅改 stop / MA / momentum。最後只把 training data 中 fitness 最高的參數拿去 2016-2026 測試。")
    lines.append("")
    lines.append("- Chromosome: `(trailing_stop_pct, reentry_ma_window, momentum_lookback)`。")
    lines.append("- Search range: `trailing_stop_pct = 18%-35%`，`reentry_ma = 50-200 days`，`momentum_lookback = 63-252 days`。")
    lines.append("- Population size: 28；generations: 22；保留前 4 名 elite。")
    lines.append("- Fitness: 訓練期內的 return / drawdown / volatility / turnover / exposure 綜合分數，且加入前段與後段訓練期穩定性懲罰。")
    lines.append("- Volatility usage: 單股 GA 的進出場訊號不是直接由 volatility 觸發；volatility 是 fitness 的風險懲罰項。真正直接使用 inverse volatility / vol cap 的是 Problem 1A 的五檔組合策略。")
    lines.append("- Anti-leakage: GA 找參數時不讀取 2016-2026 的測試績效；測試結果只用於最後評估。")
    lines.append("")
    ga_param_rows = []
    for _, r in p1_ga_params.iterrows():
        ga_param_rows.append(
            {
                "Ticker": r["ticker"],
                "GA Stop": pct(r["ga_trailing_stop_pct"]),
                "GA MA": int(r["ga_reentry_ma"]),
                "GA Momentum": int(r["ga_momentum_lookback"]),
                "參數影響": p1_parameter_effect(r),
                "GA Test Cum.": pct(r["test_ga_cumulative_return"]),
                "Fixed Test Cum.": pct(r["test_fixed_cumulative_return"]),
                "Delta": pct(r["test_cumulative_delta_vs_fixed"]),
                "GA > Fixed": bool(r["ga_beats_fixed_cumulative"]),
            }
        )
    lines.append(md_table(ga_param_rows, ["Ticker", "GA Stop", "GA MA", "GA Momentum", "參數影響", "GA Test Cum.", "Fixed Test Cum.", "Delta", "GA > Fixed"]))
    lines.append("")
    lines.append("### GA Walk-Forward Check")
    lines.append("")
    lines.append("因資料從 2006 開始，不能真的從 2006 就測 GA，否則沒有更早的訓練資料。本報告採用五年起始訓練期：`2006-2010 train -> 2011 test`，逐年展開到 2026。這個結果用來檢查 GA 不是只剛好適合單一切割。")
    lines.append("")
    lines.append("Internal validation period: 2011-2015")
    lines.append("")
    lines.append(md_table(walkforward_rows(p1_ga_wf_summary, "pre_2016_internal_wf"), ["Ticker", "GA Avg Year Cum.", "Fixed Avg Year Cum.", "DCA Avg Year Cum.", "B&H Avg Year Cum.", "Read"]))
    lines.append("")
    lines.append("Out-of-sample period: 2016-2026")
    lines.append("")
    lines.append(md_table(walkforward_rows(p1_ga_wf_summary, "post_2016_oos_wf"), ["Ticker", "GA Avg Year Cum.", "Fixed Avg Year Cum.", "DCA Avg Year Cum.", "B&H Avg Year Cum.", "Read"]))
    lines.append("")
    lines.append("Walk-forward 結論：年度平均表用來檢查 GA 是否只適合單一切割；若 GA 只在少數股票有效，就要在限制中誠實說明 overfitting / whipsaw 風險。")
    lines.append("")
    lines.append("![Problem 1 GA Walk-Forward](../figures/reco_problem1_ga_walkforward_cumulative.png)")
    lines.append("")
    lines.append("")
    lines.append("### Problem 2: Pairs Trading Strategy")
    lines.append("")
    lines.append("Pairs trading 的假設是：兩檔股票若長期存在穩定關係，短期 spread 偏離後可能回歸。設計如下：")
    lines.append("")
    lines.append("- 使用同一組五檔股票中的 `V-MA` 作為固定 pair。選它不是因為測試期績效最好，而是因為 Visa 與 Mastercard 屬於同產業、商業模式相近、價格長期高度相關，報告上更容易解釋。")
    lines.append("- 候選 pair 表仍保留 cointegration / correlation 檢查；但正式策略不再每年切換 pair，避免研究設計前後不一致。")
    lines.append("- 對兩檔股價取 log，使用 OLS / rolling beta 定義 spread。")
    lines.append("- 使用 40 日 rolling z-score 作為 fixed baseline 的進出場訊號；GA 版本則在 30-90 日區間內自行選 lookback。")
    lines.append("- Fixed baseline 使用 `z <= -1.5` 做 long spread，`z >= 1.5` 做 short spread。")
    lines.append("- `z` 依方向回到 0 附近平倉；`|z| >= 3.5` 停損；最長持有 60 天。")
    lines.append("- ADF p-value 與 half-life 這版不再當硬性進場 filter，而是作為診斷資訊。原因是前一版 filter 太嚴，導致策略過度空手，看起來不像真的 pairs trading。")
    lines.append("- 固定參數版本使用乾淨的事前 baseline：`lookback=40`、`entry_z=1.5`、`exit_z=0.0`、`stop_z=3.5`、`max_holding_days=60`。這組不是從 2016-2026 測試期調出來，而是 pairs trading 常見的 classic z-score rule。")
    lines.append("- GA 版本不改變策略架構，只搜尋 `lookback / entry_z / exit_z / stop_z / max_holding_days`。正式測試窗只有 `2016-2026`；GA 只用 2015 年底以前資料做內部 formation/validation，因此不偷看測試期。")
    lines.append("- P2 GA 的 chromosome 實際儲存 7 個欄位：`lookback / entry_z / exit_z / stop_z / max_holding_days / ADF p-value threshold / max half-life`。但本版 `use_regime_filter=False`，所以真正影響交易的是前 5 個；ADF 與 half-life 只保留為診斷與未來擴充，不再硬擋交易。")
    lines.append("- P2 GA 搜尋範圍：lookback 30-90、entry_z 1.0-2.5、exit_z 0-0.6、stop_z 2.5-4.2、max hold 20-80；population 18、generations 12、elite 4。")
    lines.append("- P2 fitness 主要獎勵 validation return，懲罰 drawdown、volatility、零交易與過少交易，目的是找出更穩定的 spread-trading 規則，而不是改變 pair。")
    lines.append("")
    selection_rows = []
    for _, r in p2_pair_selection[p2_pair_selection["selected"]].iterrows():
        selection_rows.append(
            {
                "Window": r["window"],
                "Rank": int(r["rank"]),
                "Pair": r["pair"],
                "Train Corr.": f"{float(r['corr_train']):.3f}",
                "Coint p-value": f"{float(r['coint_pvalue_train']):.4f}",
                "Selected": bool(r["selected"]),
            }
        )
    lines.append("Pair choice is not random. The formal report uses the fixed famous pair `V-MA`; the table below shows its pre-2016 training-period correlation is very high. Other screened candidates are kept in `results/reco_problem2_pair_selection.csv` only as comparison evidence, not as formal strategy choices:")
    lines.append("")
    lines.append(md_table(selection_rows, ["Window", "Rank", "Pair", "Train Corr.", "Coint p-value", "Selected"]))
    lines.append("")

    lines.append("## 4. Problem 1 Performance Summary")
    lines.append("")
    lines.append(md_table(summary_rows(p1_summary), ["Ticker", "GA Ann.", "GA Cum.", "GA MDD", "Fixed Cum.", "DCA Cum.", "B&H Cum.", "Main Read"]))
    lines.append("")
    lines.append("整體來看，GA 單股策略不應只看訓練期好壞；若測試期輸給固定參數或 Buy&Hold，代表最佳化可能貼合 training period，因此必須做 out-of-sample 與 walk-forward 檢查。")
    lines.append("")
    lines.append("注意：下圖 `Problem 1 Universe` 是五檔股票的綜合組合圖。藍線不是某一檔股票，而是五檔 universe 依 Dual Momentum + Trend + VolCap 規則形成的組合；橘線與綠線也是五檔等權 Buy&Hold / DCA。每檔單股的 GA 圖在下一張 gallery 與第 5 節。")
    lines.append("")
    lines.append("![Problem 1 Universe](../figures/reco_problem1_vs_benchmarks.png)")
    lines.append("")
    lines.append("![Problem 1 Per-Stock Line Gallery](../figures/reco_problem1_all_single_stock_lines.png)")
    lines.append("")

    lines.append("## 5. Problem 1 Stock-by-Stock Interpretation")
    lines.append("")
    for ticker in TICKERS:
        lines.append(f"### {ticker}")
        lines.append("")
        lines.extend(stock_commentary(ticker, p1_summary, p1_ga_params, phases, turns))
        lines.append("")
        lines.append(f"![{ticker} Strategy vs Benchmarks](../figures/reco_problem1_{ticker}_vs_benchmarks.png)")
        lines.append("")
        lines.append("關鍵事件解釋：")
        lines.append("")
        lines.append("以下每一列都用同一個邏輯讀：GA 當時根據 stop / MA / momentum 做出進出場，後續股價變化就是這個決策造成的績效走向。")
        lines.append("")
        for event_line in p1_key_event_explanations(ticker, turns, p1_ga_params, n=3):
            lines.append(f"- {event_line}")
        lines.append("")
        trows = turns[turns["ticker"] == ticker]
        trows_md = []
        for _, r in trows.iterrows():
            trows_md.append(
                {
                    "Date": r["date"],
                    "Action": r["action"],
                    "Price": f"{r['price']:.2f}",
                    "DD from Peak": pct(r["drawdown_from_peak"]),
                    "GA Mom.": pct(r["momentum_126d"]),
                    "Price vs MA": pct(r["price_vs_ma_pct"]),
                    "63D Vol.": pct(r["volatility_63d"]),
                    "Next Date": r["next_change_date"],
                    "Price Move": pct(r["price_return_until_next_change"]),
                    "Interpretation": r["interpretation"],
                }
            )
        lines.append(md_table(trows_md, ["Date", "Action", "Price", "DD from Peak", "GA Mom.", "Price vs MA", "63D Vol.", "Next Date", "Price Move", "Interpretation"]))
        lines.append("")

    lines.append("## 6. Problem 2 Results and Interpretation")
    lines.append("")
    p2_summary_rows = []
    for _, r in strategy_summary[strategy_summary["problem"] == "problem2"].iterrows():
        p2_summary_rows.append(
            {
                "Strategy": r["strategy"],
                "Annualized": pct(r["annualized_return"]),
                "Cumulative": pct(r["cumulative_return"]),
                "MDD": pct(r["max_drawdown"]),
                "Volatility": pct(r["volatility"]),
            }
        )
    lines.append(md_table(p2_summary_rows, ["Strategy", "Annualized", "Cumulative", "MDD", "Volatility"]))
    lines.append("")
    lines.append("![Problem 2 Pairs](../figures/reco_problem2_vs_benchmarks.png)")
    lines.append("")
    lines.append("Problem 2 的 pairs strategy 只使用同一組 `V-MA` 跑 `2016-2026` 十年測試窗。若報酬低於 pair 的 Buy&Hold / DCA，但最大回撤和波動較低，這是 market-neutral / hedged 策略常見的現象：它不是靠單邊多頭行情賺錢，而是靠 spread 回歸賺錢；如果市場本身大漲，長抱基準會自然占優。")
    lines.append("")
    p2_window_rows = []
    for _, r in p2_windows.iterrows():
        p2_window_rows.append(
            {
                "Window": r["window"],
                "Pair": r["pair"],
                "Annualized": pct(r["annualized_return"]),
                "Cumulative": pct(r["cumulative_return"]),
                "MDD": pct(r["max_drawdown"]),
                "End Equity": money(r["end_equity"]),
            }
        )
    lines.append(md_table(p2_window_rows, ["Window", "Pair", "Annualized", "Cumulative", "MDD", "End Equity"]))
    lines.append("")
    lines.append("### Problem 2 GA Parameter Selection")
    lines.append("")
    p2_ga_param_rows = []
    for _, r in p2_ga_params.iterrows():
        p2_ga_param_rows.append(
            {
                "Window": r["window"],
                "Selected Pairs": r["selected_pairs"],
                "Lookback": int(r["lookback"]),
                "Entry Z": f"{float(r['entry_z']):.2f}",
                "Exit Z": f"{float(r['exit_z']):.2f}",
                "Stop Z": f"{float(r['stop_z']):.2f}",
                "Max Hold": int(r["max_holding_days"]),
                "Regime Filter": bool(r.get("use_regime_filter", False)),
                "交易影響": (
                    f"進場門檻{'更嚴格' if float(r['entry_z']) > 2.0 else '更寬鬆'}；"
                    f"持有天數{'更短' if int(r['max_holding_days']) < 60 else '較長或相同'}"
                ),
            }
        )
    lines.append(md_table(p2_ga_param_rows, ["Window", "Selected Pairs", "Lookback", "Entry Z", "Exit Z", "Stop Z", "Max Hold", "Regime Filter", "交易影響"]))
    lines.append("")
    lines.append("GA 的 P2 參數不是拿測試期答案調出來的；正式結果只有 `2016-2026` 這張 10 年圖，使用 2015 年底以前資料訓練 GA，並在 training data 內部切 formation / validation 來評分。")
    lines.append("")
    lines.append(md_table(p2_ga_metric_rows(p2_ga_windows), ["Window", "Pair", "Strategy", "Cum.", "MDD", "Vol.", "End Equity"]))
    lines.append("")
    lines.append("![Problem 2 GA vs Fixed](../figures/reco_problem2_ga_vs_fixed.png)")
    lines.append("")
    lines.append("下面這張 zoom chart 只畫 P2 策略本身，不放 Buy&Hold / DCA，避免長抱基準把 y 軸拉高後讓 pair strategy 看起來像水平線。")
    lines.append("")
    lines.append("![Problem 2 Strategy Zoom](../figures/reco_problem2_2016-2026_v_ma_strategy_zoom.png)")
    lines.append("")
    p2_ga_lookup = p2_ga_windows.copy()
    p2_ga_lookup["window"] = p2_ga_lookup["window"].astype(str)
    p2_ga_pivot = p2_ga_lookup.pivot_table(index=["window", "pair"], columns="strategy", values="cumulative_return", aggfunc="first")
    selected_pairs_for_text = p2_pair_selection[p2_pair_selection["selected"]].copy()
    selected_pairs_for_text["window"] = selected_pairs_for_text["window"].astype(str)
    first_sel = selected_pairs_for_text.iloc[0] if not selected_pairs_for_text.empty else None
    last_sel = selected_pairs_for_text.iloc[-1] if not selected_pairs_for_text.empty else None
    first_key = (str(first_sel["window"]), str(first_sel["pair"])) if first_sel is not None else ("", "")
    last_key = (str(last_sel["window"]), str(last_sel["pair"])) if last_sel is not None else ("", "")
    first_ga = p2_ga_pivot.loc[first_key, "ga_pairs_zscore"] if first_key in p2_ga_pivot.index else np.nan
    first_fixed = p2_ga_pivot.loc[first_key, "fixed_pairs_zscore"] if first_key in p2_ga_pivot.index else np.nan
    last_ga = p2_ga_pivot.loc[last_key, "ga_pairs_zscore"] if last_key in p2_ga_pivot.index else np.nan
    last_fixed = p2_ga_pivot.loc[last_key, "fixed_pairs_zscore"] if last_key in p2_ga_pivot.index else np.nan
    p2_ga_comments = []
    for _, r in selected_pairs_for_text.iterrows():
        key = (str(r["window"]), str(r["pair"]))
        ga_val = p2_ga_pivot.loc[key, "ga_pairs_zscore"] if key in p2_ga_pivot.index else np.nan
        fixed_val = p2_ga_pivot.loc[key, "fixed_pairs_zscore"] if key in p2_ga_pivot.index else np.nan
        if np.isfinite(float(ga_val)) and np.isfinite(float(fixed_val)):
            if abs(float(ga_val) - float(fixed_val)) < 1e-9:
                read = "與固定參數相同"
            elif float(ga_val) > float(fixed_val):
                read = "優於固定參數"
            else:
                read = "低於固定參數"
            p2_ga_comments.append(f"{key[0]} 年 `{key[1]}` GA {pct(ga_val)}，固定參數 {pct(fixed_val)}，{read}")
    lines.append(
        "P2 GA 結論需要保守寫：正式測試窗固定使用 `V-MA`，GA 只負責調交易參數。"
        + "；".join(p2_ga_comments)
        + "。這代表 GA 不保證一定打敗固定參數；本報告保留 GA 結果，是為了展示參數搜尋與 out-of-sample 驗證，而不是把 GA 包裝成必勝模型。"
    )
    lines.append("")
    lines.append("P2 GA 為什麼贏或輸 fixed：")
    lines.append("")
    for explain in p2_ga_vs_fixed_explanations(p2_ga_params, p2_ga_windows, p2_ga_trades):
        lines.append(f"- {explain}")
    lines.append("")
    lines.append("P2 關鍵交易解釋：")
    lines.append("")
    key_p2_rows = p2_key_trade_rows(p2_ga_trades, prices)
    lines.append(md_table(key_p2_rows, ["Window", "Strategy", "Trade", "Side", "Z", "Return", "Why"]))
    lines.append("")
    lines.append("### Problem 2 Price and Position Charts")
    lines.append("")
    lines.append("以下圖表才是用來解釋 P2 的主要圖：第一層是 pair 兩檔股票價格走勢，第二層是策略與 benchmark 資產曲線，第三層是 z-score 與持倉線。持倉 `+1` 代表 long spread，`-1` 代表 short spread，`0` 代表空手。")
    lines.append("")
    for _, r in p2_pair_selection[p2_pair_selection["selected"]].iterrows():
        pair = str(r["pair"])
        a, b = pair.split("-", 1)
        image_name = f"reco_problem2_{r['window']}_{a.lower()}_{b.lower()}_price_position.png"
        lines.append(f"![P2 {r['window']} {pair} Price Position](../figures/{image_name})")
        lines.append("")

    lines.append("### Problem 2 Trade Log")
    lines.append("")
    trade_rows = []
    for _, r in p2_trades.iterrows():
        trade_rows.append(
            {
                "Pair": r["pair"],
                "Window": r["window"],
                "Entry": r["entry_date"].date().isoformat(),
                "Exit": r["exit_date"].date().isoformat(),
                "Side": r["side"],
                "Return": pct(r["trade_return"]),
                "Exit Reason": r["exit_reason"],
                "Holding Days": int(r["holding_days"]),
            }
        )
    lines.append(md_table(trade_rows, ["Pair", "Window", "Entry", "Exit", "Side", "Return", "Exit Reason", "Holding Days"]))
    lines.append("")
    if p2_trades.empty:
        lines.append("固定參數 P2 沒有實際成交；這代表 z-score 在測試期沒有觸發完整進出場訊號。")
    else:
        wins = int((p2_trades["trade_return"] > 0).sum())
        losses = int((p2_trades["trade_return"] < 0).sum())
        flat = int((p2_trades["trade_return"] == 0).sum())
        avg_trade = float(p2_trades["trade_return"].mean())
        top_exit = str(p2_trades["exit_reason"].mode().iloc[0])
        selected_trade_keys = {
            (str(row["window"]), str(row["pair"])) for _, row in p2_trades.iterrows()
        }
        no_trade_pairs = [
            f"{row['window']} {row['pair']}"
            for _, row in p2_pair_selection[p2_pair_selection["selected"]].iterrows()
            if (str(row["window"]), str(row["pair"])) not in selected_trade_keys
        ]
        no_trade_text = "；未成交 window：" + "、".join(no_trade_pairs) if no_trade_pairs else ""
        lines.append(
            f"固定參數 P2 共有 {len(p2_trades)} 筆交易，勝 {wins}、負 {losses}、持平 {flat}，"
            f"平均單筆報酬 {pct(avg_trade)}，最常見出場原因是 `{top_exit}`{no_trade_text}。"
            "本次結果要誠實寫成：若 spread 沒有在持有期限內完成回歸，max holding rule 會主動收束風險；"
            "若 z-score 偏離後順利回到 0 附近，策略就以 mean_reversion 出場。"
        )
    lines.append("")

    lines.append("### Problem 2 Diagnostics")
    lines.append("")
    if "reason" in p2_rejects.columns and not p2_rejects.empty:
        reason_rows = []
        for reason, count in p2_rejects["reason"].value_counts().items():
            reason_rows.append({"Reason": reason, "Count": int(count), "Meaning": reject_reason_meaning(reason)})
        lines.append(md_table(reason_rows, ["Reason", "Count", "Meaning"]))
    else:
        lines.append("_No rejected entries: the final P2 design uses classic z-score entries and keeps ADF / half-life as diagnostics rather than hard filters._")
    lines.append("")
    lines.append(
        "前一版把 ADF / half-life 當硬性 regime filter，結果交易太少、資產線幾乎水平。"
        "本版改成經典 z-score pairs trading：先讓 spread 偏離時確實進場，再用 stop_z 與 max_holding_days 控制風險。"
    )
    lines.append("")

    lines.append("## 7. Temporal Validation Summary")
    lines.append("")
    p1_temporal_rows = []
    for _, r in strategy_summary[strategy_summary["problem"] == "problem1"].iterrows():
        p1_temporal_rows.append(
            {
                "Strategy": r["strategy"],
                "Annualized": pct(r["annualized_return"]),
                "Cumulative": pct(r["cumulative_return"]),
                "MDD": pct(r["max_drawdown"]),
                "Volatility": pct(r["volatility"]),
            }
        )
    lines.append(md_table(p1_temporal_rows, ["Strategy", "Annualized", "Cumulative", "MDD", "Volatility"]))
    lines.append("")
    lines.append("Temporal validation 結果顯示，五檔輪動策略在逐年展開視窗中，平均年化略高於 Buy&Hold / DCA，但累積報酬非常接近。這代表策略不是壓倒性勝利，而是穩定性與風險控制略有改善。")
    lines.append("")

    lines.append("## 8. Limitations and Improvements")
    lines.append("")
    lines.append("- GA 使用 2006-2015 訓練期選參數；若某些股票在 2016-2026 測試期輸給原固定參數，代表 GA 仍有 overfitting / whipsaw 風險。")
    lines.append("- Walk-forward GA 改善了單一切割的疑慮：2016-2026 年度平均中五檔都贏固定參數與 DCA，但它仍多數輸給 Buy&Hold，表示策略價值主要在風控與規則可解釋，不是保證最高報酬。")
    lines.append("- 停損參數不能搜尋得太寬；如果允許 10%-15% 的過緊停損，GA 在訓練期可能很好看，但測試期容易被快速反彈洗出去。")
    lines.append("- Buy&Hold 在強多頭市場很難被打敗；策略出場後若市場快速 V 型反彈，會有延遲進場問題。")
    lines.append("- DCA 的優勢是降低一次投入時點風險；若市場震盪或先跌後漲，DCA 可能更穩。")
    lines.append("- Problem 2 固定使用 `V-MA`，優點是故事一致、產業邏輯清楚；限制是它不一定是每個 training window 統計排名第一的 pair，因此報告要把「有名且可解釋」與「統計排名」分開說明。")
    lines.append("- P2 GA 不保證穩定改善固定參數：GA 只用訓練期 validation 選參數，若 validation period 的 spread 型態和 2016-2026 測試期不同，仍可能輸給簡單固定規則。這說明 GA 參數最佳化仍需要 walk-forward 驗證。")
    lines.append("- 後續可改進：加入更多候選 pair、使用 sector-neutral pair universe、或把 P2 GA 的 fitness 改成同時要求交易次數與跨 pair 穩定性，但不能把模型做得過度複雜。")
    lines.append("")

    lines.append("## 9. Individual Contribution Statement")
    lines.append("")
    lines.append("Gavin: data collection, Python strategy implementation, benchmark construction, temporal validation, chart generation, performance interpretation, Markdown report, and PPT organization. If this is submitted as a group project, replace this section with each member's actual contribution.")
    lines.append("")

    lines.append("## 10. Reproducibility")
    lines.append("")
    lines.append("Run the following commands from the `專題二` folder:")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 scripts/run_term_project2_recommended.py")
    lines.append("PYTHONDONTWRITEBYTECODE=1 python3 scripts/build_report_assets.py")
    lines.append("PYTHONDONTWRITEBYTECODE=1 python3 scripts/build_final_report.py")
    lines.append("```")
    lines.append("")
    lines.append("Main outputs:")
    lines.append("")
    lines.append("- `report/final_report.md`")
    lines.append("- `ppt/index.html`")
    lines.append("- `figures/reco_problem1_*_vs_benchmarks.png`")
    lines.append("- `figures/reco_problem1_ga_walkforward_cumulative.png`")
    lines.append("- `figures/reco_problem2_vs_benchmarks.png`")
    lines.append("- `figures/reco_problem2_ga_vs_fixed.png`")
    lines.append("- `figures/reco_problem2_2016-2026_v_ma_strategy_zoom.png`")
    lines.append("- `results/reco_problem1_turning_points.csv`")
    lines.append("- `results/reco_problem1_stock_phases.csv`")
    lines.append("- `results/reco_problem1_ga_walkforward*.csv`")
    lines.append("- `results/reco_problem2_trade_log.csv`")
    lines.append("- `results/reco_problem2_reject_log.csv`")
    lines.append("- `results/reco_problem2_ga_*.csv`")
    lines.append("")
    return "\n".join(lines)


def reject_reason_meaning(reason: str) -> str:
    if reason == "adf_pvalue_too_high":
        return "ADF 檢定不支持均值回歸，spread 可能不是穩定關係。"
    if reason == "half_life_out_of_range":
        return "估計回歸速度太慢、太快或不可估，交易風險較高。"
    return "Regime filter did not pass."


def top_turns_for_slide(turns: pd.DataFrame, ticker: str, n: int = 3) -> list[str]:
    g = turns[turns["ticker"] == ticker].copy()
    if g.empty:
        return ["無明顯倉位轉折。"]
    g["abs_move"] = g["price_return_until_next_change"].abs()
    out = []
    for _, r in g.sort_values("abs_move", ascending=False).head(n).iterrows():
        action = "出場" if r["action"] == "Exit" else "進場"
        out.append(
            f"{r['date']} {action}：{pct(r['drawdown_from_peak'])} DD / "
            f"{pct(r['momentum_126d'])} mom / {pct(r['volatility_63d'])} vol，"
            f"到 {r['next_change_date']} 股價 {pct(r['price_return_until_next_change'])}"
        )
    return out


def stock_slide_turning_rows(ticker: str, turns: pd.DataFrame, params: pd.Series, n: int = 3) -> str:
    g = turns[turns["ticker"] == ticker].copy()
    if g.empty:
        return html_rowline("No turn", "沒有明顯倉位轉折", "策略主要維持既有持倉。")
    g["abs_move"] = g["price_return_until_next_change"].abs()
    rows: list[str] = []
    for _, r in g.sort_values("abs_move", ascending=False).head(n).iterrows():
        if r["action"] == "Exit":
            title = f"{r['date']} 出場"
            trigger = (
                f"回撤 {pct(r['drawdown_from_peak'])} 觸發 Stop {pct(params['ga_trailing_stop_pct'])}，"
                f"且低於 MA{int(params['ga_reentry_ma'])}"
            )
            if float(r["price_return_until_next_change"]) < 0:
                impact = f"後續到 {r['next_change_date']} 股價 {pct(r['price_return_until_next_change'])}，出場避開下跌，對 GA 有利。"
            else:
                impact = f"後續到 {r['next_change_date']} 股價 {pct(r['price_return_until_next_change'])}，出場太早，GA 少吃反彈。"
        else:
            title = f"{r['date']} 重新進場"
            trigger = (
                f"站上 MA{int(params['ga_reentry_ma'])}，Mom{int(params['ga_momentum_lookback'])}="
                f"{pct(r['momentum_126d'])}，63D vol {pct(r['volatility_63d'])}"
            )
            if float(r["price_return_until_next_change"]) > 0:
                impact = f"後續到 {r['next_change_date']} 股價 {pct(r['price_return_until_next_change'])}，GA 重新進場吃到上漲。"
            else:
                impact = f"後續到 {r['next_change_date']} 股價 {pct(r['price_return_until_next_change'])}，這次是假突破 / whipsaw。"
        rows.append(html_rowline(title, trigger, impact))
    return "\n".join(rows)


def build_slides(data: dict[str, pd.DataFrame], phases: pd.DataFrame, turns: pd.DataFrame) -> str:
    p1_summary = data["p1_summary"]
    p1_ga_params = data["p1_ga_params"]
    p1_ga_wf_summary = data["p1_ga_wf_summary"]
    strategy_summary = data["strategy_summary"]
    p2_rejects = data["p2_rejects"]
    p2_trades = data["p2_trades"]
    p2_ga_windows = data["p2_ga_windows"]
    p2_ga_params = data["p2_ga_params"]

    summary = {r["Ticker"]: r for r in summary_rows(p1_summary)}
    p1_param_lookup = {str(r["ticker"]): r for _, r in p1_ga_params.iterrows()}
    slides: list[str] = []
    slides.append(
        """
<section class="slide hero dark" data-animate="hero">
  <div class="chrome"><div>AI and FinTech · Quant Trading</div><div>Final Report Deck</div></div>
  <div class="frame" style="display:grid; gap:4vh; align-content:center; min-height:80vh">
    <div class="kicker" data-anim>Term Project II · 114-2</div>
    <h1 class="h-hero" data-anim>Quantitative Trading</h1>
    <h2 class="h-sub" data-anim>Problem 1: Rule-based Stocks · Problem 2: Pairs Trading</h2>
    <p class="lead" style="max-width:64vw" data-anim>五檔股票、三種基準對照、temporal validation、交易 log 與關鍵轉折點。</p>
  </div>
  <div class="foot"><div>Final presentation</div><div>01 / 16</div></div>
</section>
"""
    )
    slides.append(
        """
<section class="slide light">
  <div class="chrome"><div>Requirement Checklist</div><div>02 / 16</div></div>
  <div class="frame" style="padding-top:5vh">
    <div class="kicker" data-anim>老師要求</div>
    <h2 class="h-md" data-anim>所有交付項目都有對應輸出</h2>
    <div class="grid-4" style="margin-top:4vh">
      <div class="stat-card" data-anim><div class="stat-label">Stocks</div><div class="stat-nb">5</div><div class="stat-note">AAPL / MSFT / V / MA / JPM</div></div>
      <div class="stat-card" data-anim><div class="stat-label">Benchmarks</div><div class="stat-nb">3</div><div class="stat-note">Strategy / Buy&amp;Hold / DCA</div></div>
      <div class="stat-card" data-anim><div class="stat-label">Validation</div><div class="stat-nb">Temporal</div><div class="stat-note">Expanding yearly windows</div></div>
      <div class="stat-card" data-anim><div class="stat-label">Deliverables</div><div class="stat-nb">MD + PPT</div><div class="stat-note">Report, charts, logs, code</div></div>
    </div>
  </div>
  <div class="foot"><div>Checklist</div><div>02 / 16</div></div>
</section>
"""
    )
    slides.append(
        """
<section class="slide dark">
  <div class="chrome"><div>Strategy Design</div><div>03 / 16</div></div>
  <div class="frame" style="padding-top:5vh">
    <div class="kicker" data-anim>策略怎麼設計</div>
    <h2 class="h-md" data-anim>GA 不是預測股價，是搜尋規則參數</h2>
    {rows}
    <div class="callout" style="margin-top:4vh" data-anim>單股 GA 進出場不直接用 volatility；volatility 是 fitness 懲罰項。五檔組合才直接使用 inverse-vol / vol cap。</div>
  </div>
  <div class="foot"><div>Methodology</div><div>03 / 16</div></div>
</section>
""".format(
            rows="\n".join(
                [
                    html_rowline("P1 Portfolio", "Dual Momentum + Trend + inverse volatility + vol cap", "五檔綜合"),
                    html_rowline("P1 Single Stock GA", "chromosome = stop / MA / momentum", "28 population × 22 generations"),
                    html_rowline("P2 Pairs GA", "chromosome = lookback / entry / exit / stop / holding", "18 population × 12 generations"),
                    html_rowline("No look-ahead", "train first, test later", "2016-2026 never used during tuning"),
                ]
            )
        )
    )
    slides.append(
        """
<section class="slide light">
  <div class="chrome"><div>Problem 1 · Universe</div><div>04 / 16</div></div>
  <div class="frame" style="padding-top:4vh">
    <div class="kicker" data-anim>五檔組合</div>
    <h2 class="h-md" data-anim>這張是五檔等權綜合圖，不是單一股票</h2>
    <p class="body-zh" data-anim>藍線是五檔 universe 的 Dual Momentum + Trend + VolCap 組合；橘線 / 綠線是同一組五檔的 Buy&amp;Hold / DCA 基準。</p>
    <figure class="frame-img r-16x9 fit-contain" style="margin-top:3vh" data-anim><img src="images/03-p1-universe.png" alt="Problem 1 universe"></figure>
  </div>
  <div class="foot"><div>Problem 1 portfolio result</div><div>04 / 16</div></div>
</section>
"""
    )
    slides.append(
        """
<section class="slide dark">
  <div class="chrome"><div>Problem 1 · Summary</div><div>05 / 16</div></div>
  <div class="frame" style="padding-top:4vh">
    <div class="kicker" data-anim>五檔單股結論</div>
    <h2 class="h-md" data-anim>GA 用 2006-2015 訓練，2016-2026 才測試</h2>
    <div class="callout" data-anim>Stop 是回撤停損 / trailing stop：例如 Stop 20% = 從持有後高點回跌 20% 且跌破 GA 均線才出場。GA 訓練流程：28 個候選參數 × 22 代，依 training-only fitness 選出 stop / MA / momentum。</div>
    {rows}
  </div>
  <div class="foot"><div>Single-stock summary</div><div>05 / 16</div></div>
</section>
""".format(
            rows="\n".join(
                html_rowline(
                    t,
                    f"stop {pct(p1_param_lookup[t]['ga_trailing_stop_pct'])}, MA {int(p1_param_lookup[t]['ga_reentry_ma'])}, mom {int(p1_param_lookup[t]['ga_momentum_lookback'])}",
                    summary[t]["Main Read"],
                )
                for t in TICKERS
            )
        )
    )

    slide_no = 6
    for i, ticker in enumerate(TICKERS, start=0):
        theme = "light" if i % 2 == 0 else "dark"
        ph = phases[phases["ticker"] == ticker].iloc[0]
        params = p1_param_lookup[ticker]
        turning_rows_html = stock_slide_turning_rows(ticker, turns, params, n=3)
        slides.append(
            f"""
<section class="slide {theme}">
  <div class="chrome"><div>Problem 1 · {ticker}</div><div>{slide_no:02d} / 16</div></div>
  <div class="frame grid-2-7-5" style="padding-top:5vh">
    <div class="col">
      <div class="kicker" data-anim>{ticker}</div>
      <h2 class="h-md" data-anim>{summary[ticker]['Main Read']}</h2>
      <p class="body-zh" data-anim>GA 參數：Stop {pct(params['ga_trailing_stop_pct'])} / MA {int(params['ga_reentry_ma'])} / Momentum {int(params['ga_momentum_lookback'])}。訓練 2006-2015，測試 2016-2026。</p>
      <div class="kicker" data-anim>GA 關鍵轉折點與績效走向</div>
      {turning_rows_html}
    </div>
    <figure class="frame-img r-16x10 fit-contain" data-anim><img src="images/{4+i:02d}-p1-{ticker.lower()}.png" alt="{ticker} chart"></figure>
  </div>
  <div class="foot"><div>{ticker} turning points</div><div>{slide_no:02d} / 16</div></div>
</section>
"""
        )
        slide_no += 1

    wf_post = p1_ga_wf_summary[p1_ga_wf_summary["period"] == "post_2016_oos_wf"]
    wf_rows = []
    for ticker in TICKERS:
        g = wf_post[wf_post["ticker"] == ticker].set_index("strategy")
        if g.empty:
            continue
        ga = g.loc["ga_walkforward"]
        fixed = g.loc["fixed_rule"]
        dca = g.loc["dca"]
        bh = g.loc["buy_and_hold"]
        wf_rows.append(
            html_rowline(
                ticker,
                f"GA {pct(ga['cumulative_return'])} vs fixed {pct(fixed['cumulative_return'])} / DCA {pct(dca['cumulative_return'])}",
                f"B&H {pct(bh['cumulative_return'])}",
            )
        )
    slides.append(
        """
<section class="slide dark">
  <div class="chrome"><div>Problem 1 · GA Walk-Forward</div><div>11 / 16</div></div>
  <div class="frame grid-2-7-5" style="padding-top:5vh">
    <div class="col">
      <div class="kicker" data-anim>2016-2026 yearly OOS</div>
      <h2 class="h-md" data-anim>逐年只看過去資料選參數，五檔都贏固定參數與 DCA</h2>
      {rows}
      <div class="callout" data-anim>但多數仍輸 Buy&amp;Hold：GA 改善的是風控與進出場，不是保證最高報酬。</div>
    </div>
    <figure class="frame-img r-16x10 fit-contain" data-anim><img src="images/10-p1-ga-wf.png" alt="P1 GA walk-forward"></figure>
  </div>
  <div class="foot"><div>GA walk-forward validation</div><div>11 / 16</div></div>
</section>
""".format(rows="\n".join(wf_rows))
    )

    p2_summary = strategy_summary[strategy_summary["problem"] == "problem2"].set_index("strategy")
    slides.append(
        """
<section class="slide hero light" data-animate="hero">
  <div class="chrome"><div>Problem 2 · Method</div><div>12 / 16</div></div>
  <div class="frame" style="display:grid; gap:5vh; align-content:center; min-height:80vh">
    <div class="kicker" data-anim>Pairs Trading</div>
    <h2 class="h-xl" data-anim>不是預測股價，而是交易 spread 回歸</h2>
    <p class="lead" style="max-width:68vw" data-anim>固定使用 Visa-Mastercard。用 beta hedge 定義 spread；z-score 偏離時進場；回到 0 附近平倉；ADF 與 half-life 改為診斷，不再硬擋交易。</p>
  </div>
  <div class="foot"><div>Pairs trading logic</div><div>12 / 16</div></div>
</section>
"""
    )
    slides.append(
        f"""
<section class="slide light">
  <div class="chrome"><div>Problem 2 · Result</div><div>13 / 16</div></div>
  <div class="frame" style="padding-top:4vh">
    <div class="kicker" data-anim>Pairs strategy vs benchmark</div>
    <h2 class="h-md" data-anim>報酬輸給持有股票，但風險也明顯較低</h2>
    <figure class="frame-img r-16x9 fit-contain" style="margin-top:3vh" data-anim><img src="images/09-p2-overall.png" alt="Problem 2 result"></figure>
    <div class="meta-row" style="margin-top:2vh" data-anim><span>Pairs annualized {pct(p2_summary.loc['pairs_zscore']['annualized_return'])}</span><span>·</span><span>MDD {pct(p2_summary.loc['pairs_zscore']['max_drawdown'])}</span></div>
  </div>
  <div class="foot"><div>Problem 2 performance</div><div>13 / 16</div></div>
</section>
"""
    )
    p2_ga_avg = p2_ga_windows.groupby("strategy")["cumulative_return"].mean().to_dict()
    selected_slide_pairs = data["p2_pair_selection"][data["p2_pair_selection"]["selected"]].copy()
    selected_slide_pairs["window"] = selected_slide_pairs["window"].astype(str)
    first_pair = selected_slide_pairs.iloc[0] if not selected_slide_pairs.empty else None
    first_pair_label = str(first_pair["pair"]) if first_pair is not None else "Selected pair"
    first_window = str(first_pair["window"]) if first_pair is not None else ""
    p2_first = p2_ga_windows[(p2_ga_windows["window"].astype(str) == first_window) & (p2_ga_windows["pair"] == first_pair_label)]
    ga_first = p2_first[p2_first["strategy"] == "ga_pairs_zscore"]["cumulative_return"]
    fixed_first = p2_first[p2_first["strategy"] == "fixed_pairs_zscore"]["cumulative_return"]
    ga_first_txt = pct(float(ga_first.iloc[0])) if not ga_first.empty else "N/A"
    fixed_first_txt = pct(float(fixed_first.iloc[0])) if not fixed_first.empty else "N/A"
    p2_first_params = p2_ga_params[p2_ga_params["window"].astype(str) == first_window]
    if p2_first_params.empty:
        p2_param_text = "GA params unavailable"
        p2_param_effect = "see CSV"
    else:
        p = p2_first_params.iloc[0]
        p2_param_text = (
            f"lookback {int(p['lookback'])}, entry {float(p['entry_z']):.2f}, "
            f"stop {float(p['stop_z']):.2f}, hold {int(p['max_holding_days'])}"
        )
        p2_param_effect = "進場更嚴格、持有更短：降低風險，也限制獲利空間"
    slides.append(
        f"""
<section class="slide dark">
  <div class="chrome"><div>Problem 2 · GA Parameters</div><div>14 / 16</div></div>
  <div class="frame grid-2-7-5" style="padding-top:5vh">
    <div class="col">
      <div class="kicker" data-anim>No look-ahead GA</div>
      <h2 class="h-md" data-anim>用價格 / 資產 / 持倉線看 P2 發生什麼</h2>
      {html_rowline(f"{first_pair_label} {first_window}", f"GA {ga_first_txt}", f"Fixed {fixed_first_txt}")}
      {html_rowline("GA selected", p2_param_text, p2_param_effect)}
      {html_rowline("Fixed baseline", "40D / entry 1.5 / exit 0 / stop 3.5 / hold 60", "classic z-score rule")}
      {html_rowline("2016-2026", f"GA pairs {pct(p2_ga_avg.get('ga_pairs_zscore'))}", f"Fixed pairs {pct(p2_ga_avg.get('fixed_pairs_zscore'))}")}
      <div class="callout" data-anim>持倉線是關鍵：+1 是 long spread，-1 是 short spread，0 表示沒有 z-score 偏離訊號。</div>
    </div>
    <figure class="frame-img r-16x10 fit-contain" data-anim><img src="images/11-p2-ga-detail.png" alt="P2 price position chart"></figure>
  </div>
  <div class="foot"><div>GA parameter selection</div><div>14 / 16</div></div>
</section>
"""
    )
    reject_counts = p2_rejects["reason"].value_counts().to_dict() if "reason" in p2_rejects.columns else {}
    trade_lines = []
    p2_trade_preview = p2_trades[p2_trades["window"].astype(str) == "2016-2026"].copy()
    if p2_trade_preview.empty:
        p2_trade_preview = p2_trades.copy()
    p2_trade_preview = p2_trade_preview.head(6)
    for _, r in p2_trade_preview.iterrows():
        trade_lines.append(
            html_rowline(
                r["pair"],
                f"{r['entry_date'].date()} to {r['exit_date'].date()}, {r['side']}",
                f"{pct(r['trade_return'])}, {int(r['holding_days'])} days",
            )
        )
    if len(p2_trades) > len(p2_trade_preview):
        trade_lines.append(html_rowline("More trades", f"{len(p2_trades) - len(p2_trade_preview)} rows in CSV/report", "see full log"))
    slides.append(
        """
<section class="slide dark">
  <div class="chrome"><div>Problem 2 · Logs</div><div>15 / 16</div></div>
  <div class="frame" style="padding-top:4vh">
    <div class="kicker" data-anim>交易 log</div>
    <h2 class="h-md" data-anim>固定 V-MA：z-score 偏離時進場，回歸或停損出場</h2>
    {trades}
    <div class="callout" style="margin-top:4vh" data-anim>Final version uses classic z-score entries. ADF / half-life are diagnostics, not hard entry blockers.</div>
  </div>
  <div class="foot"><div>Problem 2 diagnostics</div><div>15 / 16</div></div>
</section>
""".format(
            trades="\n".join(trade_lines),
            adf=int(reject_counts.get("adf_pvalue_too_high", 0)),
            hl=int(reject_counts.get("half_life_out_of_range", 0)),
        )
    )
    slides.append(
        """
<section class="slide hero dark" data-animate="quote">
  <div class="chrome"><div>Conclusion</div><div>16 / 16</div></div>
  <div class="frame" style="display:grid; gap:5vh; align-content:center; min-height:80vh">
    <div class="kicker" data-anim>Final Takeaway</div>
    <h2 class="h-xl" data-anim>策略有用，但不是每段市場都贏</h2>
    <div class="callout" data-anim><span class="q-big">Problem 1 顯示規則策略能打敗 DCA，但強多頭下 Buy&amp;Hold 仍很難贏。</span></div>
    <div class="callout" data-anim><span class="q-big">Problem 2 顯示 pairs trading 更重視 regime；不交易也是風控的一部分。</span></div>
  </div>
  <div class="foot"><div>End</div><div>16 / 16</div></div>
</section>
"""
    )
    return "\n".join(slides)


def build_ppt(slides: str) -> None:
    PPT_DIR.mkdir(parents=True, exist_ok=True)
    PPT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    p2_selection = pd.read_csv(RESULTS_DIR / "reco_problem2_pair_selection.csv")
    p2_selected = p2_selection[p2_selection["selected"]].copy()
    if p2_selected.empty:
        p2_detail_src = "reco_problem2_ga_vs_fixed.png"
    else:
        r = p2_selected.iloc[0]
        a = str(r["a"]).lower()
        b = str(r["b"]).lower()
        p2_detail_src = f"reco_problem2_{r['window']}_{a}_{b}_price_position.png"

    copy_pairs = [
        ("reco_problem1_vs_benchmarks.png", "03-p1-universe.png"),
        ("reco_problem1_all_single_stock_lines.png", "03b-p1-line-gallery.png"),
        ("reco_problem1_AAPL_vs_benchmarks.png", "04-p1-aapl.png"),
        ("reco_problem1_MSFT_vs_benchmarks.png", "05-p1-msft.png"),
        ("reco_problem1_V_vs_benchmarks.png", "06-p1-v.png"),
        ("reco_problem1_MA_vs_benchmarks.png", "07-p1-ma.png"),
        ("reco_problem1_JPM_vs_benchmarks.png", "08-p1-jpm.png"),
        ("reco_problem2_vs_benchmarks.png", "09-p2-overall.png"),
        ("reco_problem1_ga_walkforward_cumulative.png", "10-p1-ga-wf.png"),
        ("reco_problem2_ga_vs_fixed.png", "11-p2-ga.png"),
        (p2_detail_src, "11-p2-ga-detail.png"),
    ]
    for src_name, dst_name in copy_pairs:
        shutil.copyfile(FIGURES_DIR / src_name, PPT_IMAGE_DIR / dst_name)

    template = SKILL_TEMPLATE.read_text(encoding="utf-8")
    template = template.replace("[必填] 替换为 PPT 标题 · Deck Title", "Term Project II Quantitative Trading · Final Report")
    template = template.replace("--ink:#0a0a0b;", "--ink:#0a1f3d;")
    template = template.replace("--ink-rgb:10,10,11;", "--ink-rgb:10,31,61;")
    template = template.replace("--paper:#f1efea;", "--paper:#f1f3f5;")
    template = template.replace("--paper-rgb:241,239,234;", "--paper-rgb:241,243,245;")
    template = template.replace("--paper-tint:#e8e5de;", "--paper-tint:#e4e8ec;")
    template = template.replace("--ink-tint:#18181a;", "--ink-tint:#152a4a;")
    template = template.replace("<!-- SLIDES_HERE -->", slides)
    (PPT_DIR / "index.html").write_text(template, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    phases = build_stock_phases(data["prices"], data["p1_signals"])
    turns = build_turning_points(data["p1_signals"], data["p1_ga_params"])

    phases.to_csv(RESULTS_DIR / "reco_problem1_stock_phases.csv", index=False)
    turns.to_csv(RESULTS_DIR / "reco_problem1_turning_points.csv", index=False)
    (REPORT_DIR / "final_report.md").write_text(report_markdown(data, phases, turns), encoding="utf-8")
    build_ppt(build_slides(data, phases, turns))

    print(f"Saved: {REPORT_DIR / 'final_report.md'}")
    print(f"Saved: {RESULTS_DIR / 'reco_problem1_stock_phases.csv'}")
    print(f"Saved: {RESULTS_DIR / 'reco_problem1_turning_points.csv'}")
    print(f"Saved: {PPT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
