from __future__ import annotations

import argparse
import gc
import json
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import polars as pl
import scipy.stats as stats
from tqdm import tqdm

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.engine import BacktestEngineConfig
from nautilus_trader.backtest.models import LatencyModel
from nautilus_trader.common.config import LoggingConfig
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import BookType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.events import PositionClosed
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import TradeId
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Currency
from nautilus_trader.model.objects import Money
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy


DATA_FOLDER = Path("take-home-project/data")
NANOSECONDS_PER_MILLISECOND = 1_000_000
MICROSECONDS_PER_SECOND = 1_000_000
NANOSECONDS_PER_SECOND = 1_000_000_000

INSTRUMENT_ID = InstrumentId.from_str("ETHUSDT-PERP.EXCHANGE")
VENUE = Venue("EXCHANGE")


class AvellanedaStoikovHFMMConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    risk_aversion: float = 0.5
    max_inventory: Decimal = Decimal("10.0")
    order_qty: Decimal = Decimal("0.5")
    ofi_sensitivity: float = 1.2
    theta_threshold: float = 0.1
    funding_skew_beta: float = 1000.0
    ewma_alpha: float = 0.05
    flatten_on_stop: bool = True


class AvellanedaStoikovHFMM(Strategy):
    def __init__(self, config: AvellanedaStoikovHFMMConfig) -> None:
        super().__init__(config)
        self.instrument = None
        self._ewma_spread_mean = 0.0
        self._ewma_spread_var = 0.0
        self._initialized_ewma = False
        self._last_funding_rate = 0.0

        self._active_bid_order = None
        self._active_ask_order = None

        self.orders_submitted_bid = 0
        self.orders_submitted_ask = 0
        self.orders_filled_bid = 0
        self.orders_filled_ask = 0
        self.fills_history: list[dict] = []
        self.mid_price_history: list[tuple[int, float]] = []
        self.funding_schedule: list[tuple[int, float]] = []
        self.funding_idx: int = 0
        self.total_funding_pnl: float = 0.0

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        self.subscribe_quote_ticks(self.config.instrument_id)
        self.subscribe_trade_ticks(self.config.instrument_id)

        fn_files = sorted((DATA_FOLDER / "fundings").glob("*.parquet"))
        if fn_files:
            df_fn = pl.concat([pl.read_parquet(f) for f in fn_files]).sort("datetime")
            for row in df_fn.iter_rows(named=True):
                ts_ns = int(row["datetime"].timestamp() * 1e9)
                self.funding_schedule.append((ts_ns, float(row["funding_rate"])))

    def on_stop(self) -> None:
        if not self.config.flatten_on_stop:
            return
        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)

    def on_quote_tick(self, tick: QuoteTick) -> None:
        s = float(tick.bid_price + tick.ask_price) / 2.0
        spread = float(tick.ask_price - tick.bid_price)
        self.mid_price_history.append((int(tick.ts_event), s))

        if not self._initialized_ewma:
            self._ewma_spread_mean = spread
            self._ewma_spread_var = spread * spread
            self._initialized_ewma = True
        else:
            diff = spread - self._ewma_spread_mean
            self._ewma_spread_mean += self.config.ewma_alpha * diff
            self._ewma_spread_var = (1.0 - self.config.ewma_alpha) * self._ewma_spread_var + self.config.ewma_alpha * diff * diff

        bid_sz = float(tick.bid_size)
        ask_sz = float(tick.ask_size)
        total_sz = bid_sz + ask_sz
        ofi = (bid_sz - ask_sz) / total_sz if total_sz > 0 else 0.0

        pos = self.portfolio.net_position(self.config.instrument_id)
        q = float(pos) if pos is not None else 0.0

        alpha_ofi = self.config.ofi_sensitivity * ofi
        sigma_sq = max(self._ewma_spread_var, 1e-6)
        tau = 1.0

        while self.funding_idx < len(self.funding_schedule) and tick.ts_event >= self.funding_schedule[self.funding_idx][0]:
            rate = self.funding_schedule[self.funding_idx][1]
            self._last_funding_rate = rate
            self.total_funding_pnl += -q * s * rate
            self.funding_idx += 1

        funding_adj = -self.config.funding_skew_beta * self._last_funding_rate

        r = s + (alpha_ofi / self.config.risk_aversion) - q * self.config.risk_aversion * sigma_sq * tau + funding_adj

        fee_margin = s * float(self.instrument.maker_fee) * 4
        delta_0 = max(spread / 2.0, fee_margin, float(self.instrument.price_increment))
        delta_b = delta_0 * (1.0 - 0.6 * ofi)
        delta_a = delta_0 * (1.0 + 0.6 * ofi)

        tick_size = float(self.instrument.price_increment)
        target_bid_px = math.floor((r - delta_b) / tick_size) * tick_size
        target_ask_px = math.ceil((r + delta_a) / tick_size) * tick_size

        target_bid_px = min(target_bid_px, float(tick.bid_price))
        target_ask_px = max(target_ask_px, float(tick.ask_price))

        self._manage_quotes(q, target_bid_px, target_ask_px, float(tick.bid_price), float(tick.ask_price), ofi)

    def on_trade_tick(self, tick: TradeTick) -> None:
        pass

    def on_event(self, event) -> None:
        if isinstance(event, OrderFilled) and event.instrument_id == self.config.instrument_id:
            if event.order_side == OrderSide.BUY:
                self.orders_filled_bid += 1
            elif event.order_side == OrderSide.SELL:
                self.orders_filled_ask += 1
            pos = self.portfolio.net_position(self.config.instrument_id)
            self.fills_history.append({
                "ts_ns": int(event.ts_event),
                "side": "BUY" if event.order_side == OrderSide.BUY else "SELL",
                "price": float(event.last_px),
                "size": float(event.last_qty),
                "inventory": float(pos) if pos is not None else 0.0,
                "funding_rate": float(self._last_funding_rate),
            })

    def _manage_quotes(self, q: float, target_bid: float, target_ask: float, touch_bid: float, touch_ask: float, ofi: float = 0.0) -> None:
        max_q = float(self.config.max_inventory)
        qty = self.instrument.make_qty(self.config.order_qty)

        if self._active_bid_order and self._active_bid_order.is_closed:
            self._active_bid_order = None
        if self._active_ask_order and self._active_ask_order.is_closed:
            self._active_ask_order = None

        if q >= max_q or ofi < -0.45:
            if self._active_bid_order is not None:
                self.cancel_order(self._active_bid_order)
                self._active_bid_order = None
        else:
            px = target_bid if q > -max_q else touch_bid
            px_obj = self.instrument.make_price(px)
            if self._active_bid_order is None:
                order = self.order_factory.limit(
                    instrument_id=self.config.instrument_id,
                    order_side=OrderSide.BUY,
                    quantity=qty,
                    price=px_obj,
                    time_in_force=TimeInForce.GTC,
                    post_only=True,
                )
                self.submit_order(order)
                self._active_bid_order = order
                self.orders_submitted_bid += 1
            elif abs(float(self._active_bid_order.price) - px) >= self.config.theta_threshold:
                self.cancel_order(self._active_bid_order)
                order = self.order_factory.limit(
                    instrument_id=self.config.instrument_id,
                    order_side=OrderSide.BUY,
                    quantity=qty,
                    price=px_obj,
                    time_in_force=TimeInForce.GTC,
                    post_only=True,
                )
                self.submit_order(order)
                self._active_bid_order = order
                self.orders_submitted_bid += 1

        if q <= -max_q or ofi > 0.45:
            if self._active_ask_order is not None:
                self.cancel_order(self._active_ask_order)
                self._active_ask_order = None
        else:
            px = target_ask if q < max_q else touch_ask
            px_obj = self.instrument.make_price(px)
            if self._active_ask_order is None:
                order = self.order_factory.limit(
                    instrument_id=self.config.instrument_id,
                    order_side=OrderSide.SELL,
                    quantity=qty,
                    price=px_obj,
                    time_in_force=TimeInForce.GTC,
                    post_only=True,
                )
                self.submit_order(order)
                self._active_ask_order = order
                self.orders_submitted_ask += 1
            elif abs(float(self._active_ask_order.price) - px) >= self.config.theta_threshold:
                self.cancel_order(self._active_ask_order)
                order = self.order_factory.limit(
                    instrument_id=self.config.instrument_id,
                    order_side=OrderSide.SELL,
                    quantity=qty,
                    price=px_obj,
                    time_in_force=TimeInForce.GTC,
                    post_only=True,
                )
                self.submit_order(order)
                self._active_ask_order = order
                self.orders_submitted_ask += 1


def make_perpetual(
    instrument_id: InstrumentId,
    raw_symbol: str,
    base_currency: str,
    quote_currency: str,
    settlement_currency: str,
    taker_fee: str,
    maker_fee: str,
) -> CryptoPerpetual:
    return CryptoPerpetual(
        instrument_id=instrument_id,
        raw_symbol=Symbol(raw_symbol),
        base_currency=Currency.from_str(base_currency),
        quote_currency=Currency.from_str(quote_currency),
        settlement_currency=Currency.from_str(settlement_currency),
        is_inverse=False,
        price_precision=2,
        size_precision=4,
        price_increment=Price.from_str("0.01"),
        size_increment=Quantity.from_str("0.0001"),
        min_quantity=Quantity.from_str("0.0001"),
        min_notional=Money(1.0, Currency.from_str(settlement_currency)),
        margin_init=Decimal("1.00"),
        margin_maint=Decimal("0.50"),
        maker_fee=Decimal(maker_fee),
        taker_fee=Decimal(taker_fee),
        ts_event=0,
        ts_init=0,
    )


def build_order_latency_model(order_latency_nanos: int) -> LatencyModel | None:
    if order_latency_nanos == 0:
        return None
    return LatencyModel(
        base_latency_nanos=0,
        insert_latency_nanos=order_latency_nanos,
        update_latency_nanos=order_latency_nanos,
        cancel_latency_nanos=order_latency_nanos,
    )


def build_engine(args: argparse.Namespace, order_latency_nanos: int) -> BacktestEngine:
    engine = BacktestEngine(
        BacktestEngineConfig(
            trader_id="BACKTESTER-001",
            logging=LoggingConfig(
                log_level=args.log_level,
                bypass_logging=args.bypass_logging,
                log_colors=False,
                print_config=False,
            ),
        ),
    )
    order_latency_model = build_order_latency_model(order_latency_nanos)

    engine.add_venue(
        venue=VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1_000_000, Currency.from_str("USDT"))],
        latency_model=order_latency_model,
        book_type=BookType.L1_MBP,
    )

    engine.add_instrument(
        make_perpetual(
            INSTRUMENT_ID,
            raw_symbol="ETHUSDT",
            base_currency="ETH",
            quote_currency="USDT",
            settlement_currency="USDT",
            maker_fee="-0.0001", # BINANCE TIER 1 FEE/REBATE https://www.binance.com/en/fee/umMaker
            taker_fee="0.0005",
        ),
    )

    strategy = AvellanedaStoikovHFMM(
        AvellanedaStoikovHFMMConfig(
            instrument_id=INSTRUMENT_ID,
            risk_aversion=args.risk_aversion,
            max_inventory=Decimal(str(args.max_inventory)),
            order_qty=Decimal(str(args.order_qty)),
            ofi_sensitivity=args.ofi_sensitivity,
            theta_threshold=args.theta_threshold,
            flatten_on_stop=args.flatten_on_stop,
        ),
    )
    engine.add_strategy(strategy)
    return engine


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def day_windows(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    current = datetime.combine(start.date(), datetime.min.time(), tzinfo=timezone.utc)
    last = end.date()
    windows = []
    while current.date() <= last:
        next_day = current + timedelta(days=1)
        window_start = max(start, current)
        window_end = min(end, next_day - timedelta(microseconds=1) if next_day <= end else end)
        if window_start <= window_end:
            windows.append((window_start, window_end))
        current = next_day
    return windows


def latency_ms_to_ns(value: float, name: str) -> int:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return int(round(value * NANOSECONDS_PER_MILLISECOND))


def load_backtest_data_window(
    day_start: datetime,
    day_end: datetime,
    signal_latency_nanos: int = 0,
    subsample_stride: int = 10,
) -> list:
    date_str = day_start.strftime("%Y-%m-%d")
    ob_path = DATA_FOLDER / "orderbook" / f"{date_str}.parquet"
    tr_path = DATA_FOLDER / "trades" / f"{date_str}.parquet"
    data = []

    if ob_path.exists():
        df_ob = pl.read_parquet(ob_path)[::subsample_stride]
        for row in df_ob.iter_rows(named=True):
            ts_event = int(row["datetime"].timestamp() * 1e9)
            ts_init = ts_event + signal_latency_nanos
            q = QuoteTick(
                INSTRUMENT_ID,
                Price.from_str(f"{row['bid_price_1']:.2f}"),
                Price.from_str(f"{row['ask_price_1']:.2f}"),
                Quantity.from_str(f"{row['bid_qty_1']:.4f}"),
                Quantity.from_str(f"{row['ask_qty_1']:.4f}"),
                ts_event,
                ts_init,
            )
            data.append(q)

    if tr_path.exists():
        df_tr = pl.read_parquet(tr_path)[::subsample_stride]
        for idx, row in enumerate(df_tr.iter_rows(named=True)):
            ts_event = int(row["datetime"].timestamp() * 1e9)
            ts_init = ts_event + signal_latency_nanos
            side = AggressorSide.BUYER if row["is_maker_ask"] == 1 else AggressorSide.SELLER
            t = TradeTick(
                INSTRUMENT_ID,
                Price.from_str(f"{row['price']:.2f}"),
                Quantity.from_str(f"{row['size']:.4f}"),
                side,
                TradeId(f"{date_str}-{idx}"),
                ts_event,
                ts_init,
            )
            data.append(t)

    data.sort(key=lambda x: x.ts_init)
    return data


def run_streaming_backtest(
    engine: BacktestEngine,
    start: datetime,
    end: datetime,
    signal_latency_nanos: int = 0,
    subsample_stride: int = 10,
) -> int:
    total_count = 0
    ran = False
    windows = day_windows(start, end)
    progress = tqdm(windows, desc="Running HFMM backtest days", unit="day")
    for day_start, day_end in progress:
        data = load_backtest_data_window(day_start, day_end, signal_latency_nanos, subsample_stride)
        day_count = len(data)
        total_count += day_count
        progress.set_postfix_str(f"{day_start.date()} +{day_count:,} total={total_count:,}")
        if not data:
            continue

        engine.add_data(data, sort=False)
        engine.run(start=start, end=end, streaming=True)
        ran = True
        engine.clear_data()
        del data
        gc.collect()

    if ran:
        engine.end()
    return total_count


def compute_funding_payments(positions, start: datetime, end: datetime) -> pl.DataFrame:
    fn_files = sorted((DATA_FOLDER / "fundings").glob("*.parquet"))
    if not fn_files:
        return pl.DataFrame()
    df_fn = pl.concat([pl.read_parquet(f) for f in fn_files]).sort("datetime")
    return pl.DataFrame()


def compute_performance_analytics(strategy: AvellanedaStoikovHFMM, engine: BacktestEngine) -> dict:
    account = strategy.portfolio.account(VENUE)
    starting_balance = 1_000_000.0
    final_balance = float(account.balance_total(Currency.from_str("USDT")))
    funding_pnl = strategy.total_funding_pnl
    net_pnl = (final_balance - starting_balance) + funding_pnl

    fills = strategy.fills_history
    if fills:
        pl.DataFrame(fills).write_parquet("post_trade_fills.parquet")

    mids = strategy.mid_price_history

    adverse_selection = {"1s": 0.0, "5s": 0.0, "10s": 0.0}
    if fills and mids:
        mids_df = pl.DataFrame(mids, schema=["ts_ns", "mid"], orient="row").sort("ts_ns")
        mids_np = mids_df["mid"].to_numpy()
        ts_np = mids_df["ts_ns"].to_numpy()

        for horizon_sec, key in [(1, "1s"), (5, "5s"), (10, "10s")]:
            drift_list = []
            for fill in fills:
                fill_ts = fill["ts_ns"]
                target_ts = fill_ts + int(horizon_sec * 1e9)
                idx = np.searchsorted(ts_np, target_ts)
                if idx < len(mids_np):
                    future_mid = mids_np[idx]
                    sign = 1.0 if fill["side"] == "BUY" else -1.0
                    drift = sign * (future_mid - fill["price"])
                    drift_list.append(drift)
            if drift_list:
                adverse_selection[key] = float(np.mean(drift_list))

    ftc_bid = (strategy.orders_filled_bid / strategy.orders_submitted_bid) if strategy.orders_submitted_bid > 0 else 0.0
    ftc_ask = (strategy.orders_filled_ask / strategy.orders_submitted_ask) if strategy.orders_submitted_ask > 0 else 0.0

    inv_list = [f.get("inventory", 0.0) for f in fills] if fills else [0.0]
    mean_inv = float(np.mean(inv_list))
    max_abs_inv = float(np.max(np.abs(inv_list)))

    analytics = {
        "financial_returns": {
            "starting_balance_usdt": starting_balance,
            "final_balance_usdt": final_balance,
            "funding_pnl_usdt": round(funding_pnl, 2),
            "total_net_pnl_usdt": round(net_pnl, 2),
        },
        "inventory_statistics": {
            "mean_inventory_eth": round(mean_inv, 4),
            "max_abs_inventory_eth": round(max_abs_inv, 4),
            "final_inventory_eth": round(float(inv_list[-1]), 4) if inv_list else 0.0,
        },
        "fill_statistics": {
            "orders_submitted_bid": strategy.orders_submitted_bid,
            "orders_submitted_ask": strategy.orders_submitted_ask,
            "orders_filled_bid": strategy.orders_filled_bid,
            "orders_filled_ask": strategy.orders_filled_ask,
            "fill_to_cancel_ratio_bid": round(ftc_bid, 4),
            "fill_to_cancel_ratio_ask": round(ftc_ask, 4),
        },
        "risk_metrics": {
            "adverse_selection_drift_1s_usd": round(adverse_selection["1s"], 6),
            "adverse_selection_drift_5s_usd": round(adverse_selection["5s"], 6),
            "adverse_selection_drift_10s_usd": round(adverse_selection["10s"], 6),
        },
    }
    return analytics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Nautilus HFMM Avellaneda-Stoikov strategy.")
    parser.add_argument("--start", default="2026-03-19T00:00:00Z")
    parser.add_argument("--end", default="2026-03-21T23:59:59Z")
    parser.add_argument("--risk-aversion", type=float, default=0.5)
    parser.add_argument("--max-inventory", type=float, default=10.0)
    parser.add_argument("--order-qty", type=float, default=0.5)
    parser.add_argument("--ofi-sensitivity", type=float, default=1.2)
    parser.add_argument("--theta-threshold", type=float, default=0.1)
    parser.add_argument("--subsample-stride", type=int, default=10, help="Subsample stride for fast HFMM simulation")
    parser.add_argument("--flatten-on-stop", action="store_true", default=True)
    parser.add_argument("--log-level", default="ERROR")
    parser.add_argument("--bypass-logging", action="store_true")
    parser.add_argument("--data-latency-ms", type=float, default=5.0)
    parser.add_argument("--calc-latency-ms", type=float, default=1.0)
    parser.add_argument("--order-latency-ms", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    import numpy as np
    global np
    import numpy as np

    args = parse_args()
    start = parse_utc(args.start)
    end = parse_utc(args.end)
    data_latency_nanos = latency_ms_to_ns(args.data_latency_ms, "--data-latency-ms")
    calc_latency_nanos = latency_ms_to_ns(args.calc_latency_ms, "--calc-latency-ms")
    order_latency_nanos = latency_ms_to_ns(args.order_latency_ms, "--order-latency-ms")
    signal_latency_nanos = data_latency_nanos + calc_latency_nanos

    print(
        "Latency model: "
        f"data={args.data_latency_ms:g}ms, "
        f"calc={args.calc_latency_ms:g}ms, "
        f"order={args.order_latency_ms:g}ms, "
        f"total={((signal_latency_nanos + order_latency_nanos) / NANOSECONDS_PER_MILLISECOND):g}ms",
    )
    print(
        "HFMM Strategy: "
        f"risk_aversion={args.risk_aversion}, "
        f"max_inventory={args.max_inventory}, "
        f"order_qty={args.order_qty}, "
        f"ofi_sensitivity={args.ofi_sensitivity}, "
        f"theta_threshold={args.theta_threshold}",
    )

    engine = build_engine(args, order_latency_nanos)
    total_events = run_streaming_backtest(engine, start, end, signal_latency_nanos, args.subsample_stride)

    strategy = engine.trader.strategies()[0]
    analytics = compute_performance_analytics(strategy, engine)

    print("\n======================= HFMM PERFORMANCE ANALYTICS =======================")
    print(json.dumps(analytics, indent=2))
    print("==========================================================================")


if __name__ == "__main__":
    main()
