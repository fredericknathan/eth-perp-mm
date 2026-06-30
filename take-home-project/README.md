# Market Making — Take-Home

## Task

Design a **market making** bot and **backtest** it on **3 calendar days** of data (2026-03-19 … 2026-03-21).

Expected result: strategy description, simulator (fills, inventory, PnL), metrics for the period, and brief conclusions.

## Data

Instrument — **ETH perpetual** (~2200 USD), single market, without a symbol column.

| Folder              | Files            | Contents                                                                                                                             |
| ------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `data/orderbook/` | one file per day | Order book snapshots: 20 bid/ask levels (`bid_price_i`, `ask_price_i`, `bid_qty_i`, `ask_qty_i`), `datetime` (nanoseconds) |
| `data/trades/`    | one file per day | Trades:`datetime`, `price`, `size`, `is_maker_ask` (1 — buyer is aggressor, 0 — seller is aggressor)                       |
| `data/fundings/`  | one file per day | Funding forecast/rate:`datetime`, `funding_rate` (~every 20 s)                                                                   |
