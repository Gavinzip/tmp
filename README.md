# 專題二：Quantitative Trading

此資料夾是最終繳交版，只保留目前實際使用的流程與輸出。

1. Problem 1: Dual Momentum + Trend Filter + Volatility Control（5 檔輪動）與 5 檔單股 GA Trend-Stop 規則圖
2. Problem 2: 固定 Visa-Mastercard (`V-MA`) Pairs Trading（Classic Z-score + Risk Controls）與 GA 參數篩選

兩題皆使用 `Expanding Yearly Temporal Validation`：

- 以前段年份做 training 視窗
- 下一年做 testing
- 逐年展開

## 結構

- `scripts/data/fetch_prices.py`: 抓取 5 檔股票日資料
- `scripts/recommended/problem1_dual_momentum.py`: Problem 1 策略
- `scripts/recommended/problem1_ga.py`: Problem 1 單股 GA 參數最佳化與 Trend-Stop 策略
- `scripts/recommended/problem2_pairs.py`: Problem 2 策略
- `scripts/recommended/problem2_ga.py`: Problem 2 pairs trading GA 參數最佳化
- `scripts/recommended/benchmarks.py`: Buy&Hold / DCA benchmark
- `scripts/core/temporal_validation.py`: 時間序列驗證視窗
- `scripts/core/metrics.py`: 指標計算
- `scripts/run_term_project2_recommended.py`: 最終一鍵主流程
- `scripts/build_report_assets.py`: 產生 5 檔單股比較圖、P2 交易 log 與事件診斷
- `scripts/build_final_report.py`: 產生完整 Markdown 報告、轉折點表與 PPT
- `scripts/build_word_report.py`: 產生 Word 報告 `report/term_project2_report_formatted.docx`

## 安裝

```bash
pip install -r requirements.txt
```

## 執行

```bash
python3 scripts/run_term_project2_recommended.py
python3 scripts/build_report_assets.py
python3 scripts/build_final_report.py
python3 scripts/build_word_report.py
```

## 輸出

- `results/reco_problem1_temporal_validation.csv`
- `results/reco_problem2_temporal_validation.csv`
- `results/reco_strategy_summary.csv`
- `results/reco_problem1_ga_params.csv`
- `results/reco_problem1_ga_walkforward.csv`
- `results/reco_problem1_ga_walkforward_summary.csv`
- `results/reco_problem1_ga_walkforward_params.csv`
- `results/reco_problem2_ga_params.csv`
- `results/reco_problem2_ga_window_metrics.csv`
- `results/reco_problem2_ga_trade_log.csv`
- `results/reco_problem2_ga_reject_log.csv`
- `results/reco_problem2_pair_selection.csv`
- `results/reco_problem1_stock_phases.csv`
- `results/reco_problem1_turning_points.csv`
- `figures/reco_problem1_vs_benchmarks.png`
- `figures/reco_problem1_{AAPL,MSFT,V,MA,JPM}_vs_benchmarks.png`
- `figures/reco_problem1_all_single_stock_lines.png`
- `figures/reco_problem1_ga_walkforward_cumulative.png`
- `figures/reco_problem2_vs_benchmarks.png`
- `figures/reco_problem2_ga_vs_fixed.png`
- `figures/reco_problem2_2016-2026_v_ma_price_position.png`
- `figures/reco_problem2_2016-2026_v_ma_strategy_zoom.png`
- `results/reco_event_diagnostics.md`
- `report/final_report.md`
- `report/term_project2_report_formatted.docx`
- `ppt/index.html`

## 預設股票池

- AAPL
- MSFT
- V
- MA
- JPM

可在 `scripts/data/fetch_prices.py` 的 `DEFAULT_UNIVERSE` 調整。

## 作業要求對照

- 5 檔股票 + 長期歷史資料：已完成（AAPL, MSFT, V, MA, JPM，資料自 2006 起；V/MA 為常見支付網路 pair）
- Problem 1 規則策略：已完成（五檔輪動用 Dual Momentum + Trend + Volatility；單股圖用 GA-optimized Trend-Stop + Momentum Re-entry；另有 GA walk-forward 檢查）
- Problem 2 Pairs Trading：已完成（固定 V-MA；classic z-score pairs trading；ADF/half-life 作為診斷；另有 GA 參數篩選）
- 與 Buy-and-Hold / DCA 比較：已完成（兩題皆有）
- Temporal Validation：已完成（expanding yearly windows）
- 指標包含年化、累積、MDD、風險：已完成（`core/metrics.py` + 結果 CSV）
- 輸出圖表與表格：已完成（`figures/` + `results/`）
