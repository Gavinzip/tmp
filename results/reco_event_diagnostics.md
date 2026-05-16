# Strategy Diagnostics (Problem 1 & 2)

## Problem 1: Five-Stock Single-Asset Strategy

- `AAPL` strategy annualized return `0.2079`, cumulative `6.0536`, MDD `-0.4644`, vol `0.2504`.
- `MSFT` strategy annualized return `0.1839`, cumulative `4.7312`, MDD `-0.3713`, vol `0.2426`.
- `V` strategy annualized return `0.0761`, cumulative `1.1340`, MDD `-0.4394`, vol `0.2053`.
- `MA` strategy annualized return `0.1570`, cumulative `3.5188`, MDD `-0.3208`, vol `0.2413`.
- `JPM` strategy annualized return `0.1788`, cumulative `4.4819`, MDD `-0.4171`, vol `0.2326`.

### Problem 1 unexpected drawdowns: log-based reasons
- `AAPL` worst drawdown at `2023-01-05` (-46.44% from peak `2022-01-03`): period avg exposure `1.00`. This happened when the stop rule was still invested, or when the strategy re-entered after momentum recovered but price later pulled back again.
- `JPM` worst drawdown at `2022-10-11` (-41.71% from peak `2021-10-22`): period avg exposure `1.00`. This happened when the stop rule was still invested, or when the strategy re-entered after momentum recovered but price later pulled back again.
- `MA` worst drawdown at `2022-10-12` (-32.08% from peak `2020-02-19`): period avg exposure `1.00`. This happened when the stop rule was still invested, or when the strategy re-entered after momentum recovered but price later pulled back again.
- `MSFT` worst drawdown at `2020-03-16` (-37.13% from peak `2020-02-10`): period avg exposure `0.93`. This happened when the stop rule was still invested, or when the strategy re-entered after momentum recovered but price later pulled back again.
- `V` worst drawdown at `2022-12-19` (-43.94% from peak `2021-07-27`): period avg exposure `1.00`. This happened when the stop rule was still invested, or when the strategy re-entered after momentum recovered but price later pulled back again.

## Problem 2: Pairs Strategy

### Window outcome snapshot
- `2016-2026` `V-MA` annualized `-0.0028`, cumulative `-0.0287`, MDD `-0.1884`.

### Problem 2 unexpected behavior: log-based reasons
- Losing trade `V-MA` `2016-11-17 -> 2017-02-15` (short_spread): return `-3.35%`, exit reason `max_holding_days`, holding `60` days.
- Losing trade `V-MA` `2018-11-26 -> 2019-02-25` (long_spread): return `-3.27%`, exit reason `max_holding_days`, holding `60` days.
- Losing trade `V-MA` `2020-08-17 -> 2020-11-10` (short_spread): return `-2.94%`, exit reason `max_holding_days`, holding `60` days.
- Losing trade `V-MA` `2020-06-24 -> 2020-08-10` (long_spread): return `-2.93%`, exit reason `mean_reversion`, holding `32` days.
- Losing trade `V-MA` `2024-01-23 -> 2024-04-04` (long_spread): return `-2.71%`, exit reason `mean_reversion`, holding `50` days.
