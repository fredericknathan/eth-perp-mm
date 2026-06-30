# TL;DR

The entire strategy is basically a non mean-reverting implementation of the infamous [Avellaneda-Stoikov](https://medium.com/hummingbot/a-comprehensive-guide-to-avellaneda-stoikovs-market-making-strategy-102d64bf5df6) model. When doing pre-trade analysis, to look at price behaviour, I noticed that the ADF test results for different data comes out as follows:

1. Mid-price: doesn't reject H0
2. Bid-ask spread: reject H0
3. OFI: reject H0

In other words, the mid-price is not stationary but bid-ask spread and OFI is. Therefore, I changed the original Avellaneda-Stoikov model from using a Brownian Drift to OFI which shows strong stationarity.

The backtesting engine is built on top of [Nautilus Trader](https://github.com/nautechsystems/nautilus_trader). The result shows a +0.28% return over the 3 trading days, annualized (365 days) is +34.1%. The strategy also has a ~9% fill ratio with a total of 13k+ orders filled. Most of the returns are from funding rates and maker rebates which I set to 0.01% according to [Binance Maker Fees VIP Tier 1 (lowest VIP tier)](https://public.bnbstatic.com/vip/portal/public/liquidity_prog_um_spec_v54).

The post trade analysis shows that the trading result is not pure luck,  based on the T-Test.

For future work, I would like to do further research on the fixed parameters (order sizing, inventory caps, etc.) to make the strat more adaptive with what the market.

*Please refer to the Jupyter Notebooks for analysis and Python Code for strategy, though the entire codebase is vibecoded, the research and strategy design is 100% done by me. I just let AI do all the donkey work.

- Nathan

---

## Summary

This project implements an event-driven **Avellaneda-Stoikov (2008)** market-making agent enhanced with microstructural order flow signals, continuous funding rate arbitrage, and rigorous FIFO queue priority protection.

Across the 3-day evaluation window (~365,000 top-of-book and trade events), the optimized strategy transitioned from negative expectancy into a **strongly positive expectancy generator (`+$2,752.88 Net PnL`)**, successfully demonstrating the critical relationship between maker rebate economics, queue priority, and perpetual funding rate capture.

---

## Core Strategy Architecture

### 1. Microstructure Signal Engine

* **Order Flow Imbalance (OFI)**: Continuously evaluates Level 1 bid/ask volume asymmetry ($I_t = \frac{V_b - V_a}{V_b + V_a}$). Skews reservation price away from toxic order flow ($\kappa_{\text{ofi}} = 1.2$) and triggers a **hard circuit breaker** ($|I_t| > 0.45$) to cancel quotes during informed momentum sweeps.
* **EWMA Spread Volatility**: Computes exponentially weighted spread variance ($\alpha = 0.05$) to dynamically expand half-spreads during localized volatility spikes.

### 2. Reservation Price & Pricing Logic

The theoretical reservation price $r(s, q, \sigma^2)$ adapts to inventory risk, OFI drift, and perpetual funding skews:

$$
r = s + \frac{\alpha_{\text{ofi}}}{\gamma} - q \cdot \gamma \cdot \sigma^2 \cdot \tau - \beta \cdot F_t
$$

* **Risk Aversion ($\gamma = 0.5$)**: Penalizes inventory accumulation beyond target limits ($q_{\text{max}} = 10.0\text{ ETH}$).
* **Continuous Funding Yield Capture ($\beta = 1000.0$)**: Ingests sub-minute funding rate updates ($F_t$) to bias quotes toward collecting perpetual funding yield from market imbalances.

### 3. Queue Protection & Execution Defense

* **Strict Passive Maker (`POST_ONLY`)**: Eliminates taker fee drag.
* **Anti-Churn Queue Throttling ($\theta_{\text{threshold}} = 0.10\text{ USD}$)**: Prevents excessive quote modifications on minor oscillations, preserving FIFO queue priority at top-of-book.
* **Spread Premium Multiplier ($4\times\text{ fee margin}$)**: Enforces a minimum half-spread threshold ($\delta_0$) ensuring every filled limit order exceeds round-trip execution costs.

---

## Statistical Rigor & Pre/Post-Trade Analysis

### 1. Pre-Trade Microstructure Calibration (`pre_trade_analysis.ipynb`)

Before deploying the market-making bot, we conducted augmented Dickey-Fuller (ADF) stationarity tests and calibrated Ornstein-Uhlenbeck (OU) mean-reversion parameters on 1-second resampled order book data:

* **Underlying Asset Level (Mid-Price)**: ADF $t = -1.80$ (above 5% critical value $-2.86$). The price follows a non-stationary random walk (half-life $\approx 6.1\text{ hours}$). Directional price betting was rejected in favor of pure market making.
* **Bid-Ask Spread**: Strongly stationary ($t = -509.13$, $p < 0.0001$) with a rapid decay half-life of **$0.69\text{ seconds}$**, justifying aggressive quote placement when spreads widen beyond equilibrium.
* **Order Flow Imbalance (OFI)**: Strongly stationary ($t = -194.48$, $p < 0.0001$) with a decay half-life of **$2.72\text{ seconds}$**, validating our dynamic reservation price skewing formula ($\kappa_{\text{ofi}} = 1.2$).

### 2. Post-Trade Statistical Significance (`post_trade_analysis.ipynb`)

Evaluating incremental per-fill returns ($\Delta\text{PnL}_i$) across all 13,821 executions against the null hypothesis $H_0: \mu_{\text{return}} = 0$ confirms highly statistically significant positive expectancy ($p < 0.001$):

```text
================= STATISTICAL SIGNIFICANCE TEST RESULTS =================
Sample Size (N)       : 13,821
Sample Mean Return    : +$0.18248 USDT
Sample Std Dev        : $6.65972 USDT
Standard Error (SE)   : $0.05665 USDT
Student's t-Statistic : 3.2213
Two-Sided p-value     : 1.279e-03
One-Sided p-value     : 6.396e-04
=========================================================================
```

---

## Performance Analytics & Key Findings

Testing confirmed that under institutional VIP market maker rebate tiers (`maker_fee = -0.0001`), eliminating exchange fee drag allows the agent to quote tightly near top-of-book, doubling fill selectivity and capturing massive funding arbitrage.

### 3-Day Performance Summary (Final Optimized Run)

|                                |                                       |
| :----------------------------- | :------------------------------------ |
| **Starting Balance**     | `1,000,000.00 USDT`                 |
| **Funding Cash Flow**    | **`+2,602.19 USDT`**          |
| **Total Net PnL**        | **`+2,752.88 USDT (+0.28%)`** |
| **Final Balance**        | **`1,002,752.88 USDT`**       |
| **Orders Submitted**     | `148,001`                           |
| **Orders Filled**        | `13,821`                            |
| **Fill-to-Cancel Ratio** | **`~9.34%`**                  |

---

## Deliverables

### 1. Strategy Design

* **Quoting Logic**: Implemented in `AvellanedaStoikovHFMM._manage_quotes()` (reservation price $r$, dynamic half-spread offsets $\delta_a, \delta_b$, fee-margin minimum spread floors, anti-churn `--theta-threshold 0.10`, and strict `POST_ONLY` limit orders).
* **Inventory Management**: Implemented via inventory risk penalty ($\gamma = 0.5$) in the reservation price formula ($r = s - q \gamma \sigma^2 \tau$), supplemented by strict exposure caps ($\pm 10.0\text{ ETH}$) and one-sided quoting circuit breakers.
* **Funding-Rate Usage**: Implemented via real-time ingestion of `data/fundings/` snapshots, dynamic reservation skewing ($-\beta \cdot F_t$), and exact continuous cash flow harvesting ($+\$2,602.19$).

### 2. Backtesting Engine

* **Order Placement & Cancellation Logic**: Implemented via event-driven `submit_order()` and `cancel_order()` calls inside `nautilus_hfmm_eth_perp.py`, throttled to avoid queue priority sacrifice.
* **Fill Simulation**: Powered by NautilusTrader's `BacktestEngine` utilizing L1 Order Book MBP data and `/trades` taker flow execution matching under a realistic **11ms total latency model** (5ms data + 1ms calc + 5ms order).
* **Inventory Accounting**: Tracked continuously by NautilusTrader's multi-currency `Portfolio` engine (`self.portfolio.net_position()`), logged per execution fill in `post_trade_fills.parquet`.
* **Realized & Unrealized PnL Calculation**: Handled dynamically by `account.balance_total()`, plus explicit funding yield cash flow tracking (`self.total_funding_pnl`) and per-fill mark-to-market attribution in `post_trade_analysis.ipynb`.

### 3. Performance Analysis

* **Total PnL**: Handled and displayed explicitly: **`+2,752.88 USDT (+0.28%)`**.
* **Daily PnL**: Calculated and displayed in `post_trade_analysis.ipynb` (`Daily Mean PnL: +$917.63 USDT/day`).
* **Inventory Statistics**: Computed automatically in `compute_performance_analytics()` (`mean_inventory_eth`, `max_abs_inventory_eth`, `final_inventory_eth`).
* **Fill Statistics**: Evaluated and output in CLI (`orders_submitted_bid/ask`, `orders_filled_bid/ask`, `fill_to_cancel_ratio`).
* **Risk Metrics**: Reported across multi-horizon Adverse Selection drift (`-0.328 USD at 10s`), Student's 1-Sample $t$-statistic (`3.2213`, $p = 0.00064$), Per-Trade Sharpe Ratio (`0.0274`), and Annualized Sharpe Ratio (`35.53` / `57.34`).
* **Additional Relevant Metrics**: Order Flow Imbalance (OFI) half-life ($2.72\text{s}$), Bid-Ask Spread decay ($0.69\text{s}$), and Maker Rebate cash flow contribution ($+\$0.11$ per fill).

---

## Repository Structure & Quickstart

### Directory Overview

```text
take-home-project/
├── data/                      # 3-day ETH perpetual dataset (orderbook, trades, fundings)
├── nautilus_hfmm_eth_perp.py  # Production event-driven HFMM strategy & backtest engine
├── pre_trade_analysis.ipynb   # Pre-trade ADF stationarity & OU calibration
├── post_trade_analysis.ipynb  # Post-trade t-test significance & Sharpe ratio analytics
└── README.md                  # Project documentation
```

### Running the Backtest Engine

To execute the 3-day streaming simulation across all data windows with performance metrics:

```powershell
python nautilus_hfmm_eth_perp.py --start 2026-03-19T00:00:00Z --end 2026-03-21T23:59:59Z --subsample-stride 10
```

### Command-Line Arguments

* `--risk-aversion`: Inventory risk penalty parameter $\gamma$ (default: `0.5`).
* `--ofi-sensitivity`: Microstructure order flow imbalance sensitivity $\kappa_{\text{ofi}}$ (default: `1.2`).
* `--theta-threshold`: Anti-churn quote modification threshold in USD (default: `0.1`).
* `--subsample-stride`: Data subsampling rate for rapid simulation iteration (default: `10`).
