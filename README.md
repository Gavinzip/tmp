# Term Project II: Quantitative Trading

本 repo 是目前整理後的「專題二」最終版本，只保留這次採用的正確流程、程式碼、結果圖、報告與 PPT。舊版錯誤圖、舊實驗碼、過程中被淘汰的策略輸出沒有放進來。

## 交付檔案在哪裡

- 完整 Markdown 報告：`report/final_report.md`
- HTML PPT：`ppt/index.html`
- PPT 使用圖片：`ppt/images/`
- 程式碼產生的圖表：`figures/`
- 回測結果、GA 參數、交易紀錄、轉折點表：`results/`
- 價格資料：`data/prices_close_wide.csv` 與 `data/raw/*.csv`

補充：`results/reco_problem2_pair_selection.csv` 是 pair 選擇診斷表，會列出其他 screened candidates 作為比較證據；正式 Problem 2 策略只使用 `V-MA`。

## 專題內容

本專題依照 Term Project II 的要求，使用時間序列驗證而不是 random train/test split，並和 Buy-and-Hold / DCA 做比較。

- Problem 1：五檔股票 `AAPL, MSFT, V, MA, JPM`。
- Problem 1A：五檔組合的 `Dual Momentum + Trend Filter + Volatility Control`。
- Problem 1B：每一檔股票各自跑 `GA Trend-Stop + Momentum Re-entry`，並和固定參數、DCA、Buy-and-Hold 比較。
- Problem 2：使用知名支付網路 pair `V-MA` 做 Pairs Trading。
- Problem 2 GA：對 pairs trading 的參數做 training-only GA 搜尋，再拿到 out-of-sample 測試期間驗證。

## 完整流程

一鍵重跑：

```bash
pip install -r requirements.txt
./run_full_pipeline.sh
```

等同於依序執行：

```bash
python3 scripts/run_term_project2_recommended.py
python3 scripts/build_report_assets.py
python3 scripts/build_final_report.py
```

流程說明：

1. `scripts/data/fetch_prices.py` 從 Yahoo Finance 抓取價格資料。
2. `scripts/run_term_project2_recommended.py` 呼叫抓價程式，產生 `data/prices_close_wide.csv`，並先跑 Problem 1 / Problem 2 的主要 temporal validation。
3. `scripts/recommended/problem1_ga.py` 定義 Problem 1 單股 GA、Trend-Stop 策略、染色體與 fitness。
4. `scripts/recommended/problem2_ga.py` 定義 Problem 2 pairs trading GA、染色體與 training-only validation。
5. `scripts/build_report_assets.py` 產生 5 檔單股圖、GA walk-forward 表、Problem 2 交易紀錄與事件診斷。
6. `scripts/build_final_report.py` 產生 `report/final_report.md`、`ppt/index.html`、轉折點表與 PPT 圖片。

## GA 怎麼訓練

### Problem 1 GA

位置：`scripts/recommended/problem1_ga.py`

GA 搜尋的參數：

- `trailing_stop_pct`：回撤停損門檻，例如 `Stop 20%` 代表持倉期間從高點回落 20%，且價格低於 GA 選出的均線時出場。
- `reentry_ma`：重新進場用的移動平均天數。
- `momentum_lookback`：重新進場用的動能觀察天數。

驗證方式：

- 訓練期只用測試期以前的資料，不用未來資料。
- 報告主軸使用 `2006-2015` 訓練，`2016-2026` 測試。
- 另外保留 walk-forward 檢查：先用 `2006-2010` 訓練測 `2011`，再逐年擴張到 2026。

主要輸出：

- `results/reco_problem1_ga_params.csv`
- `results/reco_problem1_ga_walkforward.csv`
- `results/reco_problem1_ga_walkforward_params.csv`
- `results/reco_problem1_ga_walkforward_summary.csv`
- `figures/reco_problem1_ga_walkforward_cumulative.png`

### Problem 2 GA

位置：`scripts/recommended/problem2_ga.py`

GA 搜尋的參數：

- `lookback`：z-score rolling window。
- `entry_z`：進場門檻。
- `exit_z`：出場門檻。
- `stop_z`：spread 擴大時的風險停損門檻。
- `max_holding_days`：最大持倉天數。
- `regime_adf_pvalue_max`：均值回歸 regime 的 ADF 檢查門檻。
- `max_half_life_days`：均值回歸半衰期上限。

驗證方式：

- 固定使用 `V-MA`，GA 不在測試期後換 pair。
- GA 只使用測試期間以前的資料選參數。
- 目前報告保守呈現：P2 GA 有改善部分期間，但不是每段都穩定勝過 fixed baseline。

主要輸出：

- `results/reco_problem2_ga_params.csv`
- `results/reco_problem2_ga_window_metrics.csv`
- `results/reco_problem2_ga_trade_log.csv`
- `results/reco_problem2_ga_reject_log.csv`
- `results/reco_problem2_pair_selection.csv`，只作為 `V-MA` pair choice 的訓練期診斷與候選比較
- `figures/reco_problem2_ga_vs_fixed.png`

## 主要圖表

Problem 1：

- 五檔組合總覽：`figures/reco_problem1_vs_benchmarks.png`
- 五檔單股總覽：`figures/reco_problem1_all_single_stock_lines.png`
- AAPL：`figures/reco_problem1_AAPL_vs_benchmarks.png`
- MSFT：`figures/reco_problem1_MSFT_vs_benchmarks.png`
- V：`figures/reco_problem1_V_vs_benchmarks.png`
- MA：`figures/reco_problem1_MA_vs_benchmarks.png`
- JPM：`figures/reco_problem1_JPM_vs_benchmarks.png`
- GA walk-forward：`figures/reco_problem1_ga_walkforward_cumulative.png`

Problem 2：

- V-MA pairs trading vs benchmarks：`figures/reco_problem2_vs_benchmarks.png`
- GA vs fixed pairs：`figures/reco_problem2_ga_vs_fixed.png`
- 2016-2026 價格與持倉：`figures/reco_problem2_2016-2026_v_ma_price_position.png`
- 2016-2026 策略局部放大：`figures/reco_problem2_2016-2026_v_ma_strategy_zoom.png`
- 2025 價格與持倉：`figures/reco_problem2_2025_v_ma_price_position.png`
- 2026 價格與持倉：`figures/reco_problem2_2026_v_ma_price_position.png`

## 重要程式位置

- 抓價格：`scripts/data/fetch_prices.py`
- Benchmark：`scripts/recommended/benchmarks.py`
- Problem 1 組合策略：`scripts/recommended/problem1_dual_momentum.py`
- Problem 1 GA 單股策略：`scripts/recommended/problem1_ga.py`
- Problem 2 fixed pairs strategy：`scripts/recommended/problem2_pairs.py`
- Problem 2 GA：`scripts/recommended/problem2_ga.py`
- 時間序列驗證：`scripts/core/temporal_validation.py`
- 績效指標：`scripts/core/metrics.py`
- 主流程：`scripts/run_term_project2_recommended.py`
- 圖表與 log：`scripts/build_report_assets.py`
- 報告與 PPT：`scripts/build_final_report.py`

## 作業要求對照

- 五檔股票：已完成，`AAPL, MSFT, V, MA, JPM`。
- 每檔至少一個規則型策略：已完成，單股 `GA Trend-Stop + Momentum Re-entry`。
- Problem 1 策略：已完成，組合策略與五檔單股策略都有圖與表。
- Problem 2 pair trading：已完成，固定 `V-MA` pairs trading。
- Temporal validation：已完成，使用 expanding yearly windows 與 training-only GA。
- Buy-and-Hold / DCA 比較：已完成。
- 交易紀錄、拒絕進場原因、轉折點：已完成，放在 `results/` 與報告中。
- 報告與 PPT：已完成，分別在 `report/` 與 `ppt/`。

## 注意

目前 repo 內的 `data/`、`results/`、`figures/`、`report/`、`ppt/` 都是已產生好的版本。若重新執行完整流程，Yahoo Finance 的最新資料日期可能會因當天資料更新而有些微差異。
