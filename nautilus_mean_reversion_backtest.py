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
from tqdm import tqdm

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.engine import BacktestEngineConfig
from nautilus_trader.backtest.models import LatencyModel
from nautilus_trader.common.config import LoggingConfig
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import MarkPriceUpdate
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import BookType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.events import PositionClosed
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Currency
from nautilus_trader.model.objects import Money
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.strategy import Strategy


CATALOG_PATH = Path("nautilus_catalog/spcx_xyz-spcx_2026-05-22_2026-06-07")
DATA_FOLDER = Path("data_folder")
NANOSECONDS_PER_MILLISECOND = 1_000_000
MICROSECONDS_PER_SECOND = 1_000_000
NANOSECONDS_PER_SECOND = 1_000_000_000

BINANCE_INSTRUMENT_ID = InstrumentId.from_str("SPCXUSDT-PERP.BINANCE")
HL_INSTRUMENT_ID = InstrumentId.from_str("XYZ-SPCX.HYPERLIQUID")


class CrossVenueMeanReversionConfig(StrategyConfig, frozen=True):
    binance_instrument_id: InstrumentId
    hl_instrument_id: InstrumentId
    fallback_qty: Decimal = Decimal("1")
    target_notional: Decimal = Decimal("1000")
    max_qty: Decimal = Decimal("100")
    lookback_minutes: float = 60.0
    min_samples: int = 100
    min_std_pct: float = 1e-6
    entry_z: float = 2.0
    exit_z: float = 0.25
    flatten_on_stop: bool = False


@dataclass(frozen=True)
class RollingDiffStats:
    sample_count: int
    mean_pct: float
    std_pct: float
    z_score: float


class CrossVenueMeanReversion(Strategy):
    def __init__(self, config: CrossVenueMeanReversionConfig) -> None:
        super().__init__(config)
        self._marks: dict[InstrumentId, float] = {}
        self._diff_window: deque[tuple[int, float]] = deque()
        self._diff_sum = 0.0
        self._diff_sum_sq = 0.0
        self._position_qty: dict[InstrumentId, Decimal] = {}
        self._closing_instruments: set[InstrumentId] = set()
        self._state = "flat"

    def on_start(self) -> None:
        # Book deltas are still loaded into the engine for market-order execution.
        # This strategy only needs mark prices, so avoid unused per-delta callbacks.
        self.subscribe_mark_prices(self.config.binance_instrument_id)
        self.subscribe_mark_prices(self.config.hl_instrument_id)

    def on_stop(self) -> None:
        if not self.config.flatten_on_stop:
            return
        self.cancel_all_orders(self.config.binance_instrument_id)
        self.cancel_all_orders(self.config.hl_instrument_id)
        self.close_all_positions(self.config.binance_instrument_id)
        self.close_all_positions(self.config.hl_instrument_id)

    def on_mark_price(self, mark_price: MarkPriceUpdate) -> None:
        self._marks[mark_price.instrument_id] = float(mark_price.value)
        if self.config.binance_instrument_id not in self._marks:
            return
        if self.config.hl_instrument_id not in self._marks:
            return

        binance_mark = self._marks[self.config.binance_instrument_id]
        hl_mark = self._marks[self.config.hl_instrument_id]
        diff_pct = ((binance_mark - hl_mark) / hl_mark) * 100.0
        stats = self._update_diff_window(mark_price.ts_init, diff_pct)
        if stats is None:
            return

        z_score = stats.z_score

        if self._state == "flat":
            if z_score > self.config.entry_z:
                self._enter_long_hl_short_binance()
            elif z_score < -self.config.entry_z:
                self._enter_long_binance_short_hl()
        elif self._state == "long_hl_short_binance":
            if z_score <= self.config.exit_z:
                self._exit_long_hl_short_binance()
        elif self._state == "long_binance_short_hl":
            if z_score >= -self.config.exit_z:
                self._exit_long_binance_short_hl()

    def on_position_closed(self, event: PositionClosed) -> None:
        self._closing_instruments.discard(event.instrument_id)
        if self._closing_instruments or not self._state.startswith("exiting_"):
            return

        self._state = "flat"
        self._position_qty.clear()

    def _update_diff_window(self, ts_ns: int, diff_pct: float) -> RollingDiffStats | None:
        self._diff_window.append((ts_ns, diff_pct))
        self._diff_sum += diff_pct
        self._diff_sum_sq += diff_pct * diff_pct
        cutoff_ns = ts_ns - int(self.config.lookback_minutes * 60 * NANOSECONDS_PER_SECOND)
        while self._diff_window and self._diff_window[0][0] < cutoff_ns:
            _, old_diff_pct = self._diff_window.popleft()
            self._diff_sum -= old_diff_pct
            self._diff_sum_sq -= old_diff_pct * old_diff_pct

        sample_count = len(self._diff_window)
        if sample_count < self.config.min_samples:
            return None

        rolling_mean = self._diff_sum / sample_count
        variance = (self._diff_sum_sq - (self._diff_sum * self._diff_sum / sample_count)) / (sample_count - 1)
        variance = max(variance, 0.0)
        rolling_std = math.sqrt(variance)
        if rolling_std < self.config.min_std_pct:
            return None

        z_score = (diff_pct - rolling_mean) / rolling_std
        return RollingDiffStats(
            sample_count=sample_count,
            mean_pct=rolling_mean,
            std_pct=rolling_std,
            z_score=z_score,
        )

    def _quantity_for(self, instrument_id: InstrumentId) -> Decimal:
        if self.config.target_notional <= 0:
            return self.config.fallback_qty

        mark = self._marks.get(instrument_id)
        if mark is None or mark <= 0:
            return self.config.fallback_qty

        qty = self.config.target_notional / Decimal(str(mark))
        if self.config.max_qty > 0:
            qty = min(qty, self.config.max_qty)
        return qty

    def _submit_market(
        self,
        instrument_id: InstrumentId,
        side: OrderSide,
        quantity: Decimal | None = None,
        reduce_only: bool = False,
    ) -> None:
        instrument = self.cache.instrument(instrument_id)
        if instrument is None:
            self.log.error(f"Missing instrument {instrument_id}")
            return

        qty = instrument.make_qty(quantity or self._quantity_for(instrument_id))
        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=side,
            quantity=qty,
            time_in_force=TimeInForce.IOC,
            reduce_only=reduce_only,
        )
        self.submit_order(order)

    def _enter_long_hl_short_binance(self) -> None:
        self._state = "long_hl_short_binance"
        hl_qty = self._quantity_for(self.config.hl_instrument_id)
        binance_qty = self._quantity_for(self.config.binance_instrument_id)
        self._position_qty[self.config.hl_instrument_id] = hl_qty
        self._position_qty[self.config.binance_instrument_id] = binance_qty
        self._submit_market(self.config.hl_instrument_id, OrderSide.BUY, hl_qty)
        self._submit_market(self.config.binance_instrument_id, OrderSide.SELL, binance_qty)

    def _enter_long_binance_short_hl(self) -> None:
        self._state = "long_binance_short_hl"
        binance_qty = self._quantity_for(self.config.binance_instrument_id)
        hl_qty = self._quantity_for(self.config.hl_instrument_id)
        self._position_qty[self.config.binance_instrument_id] = binance_qty
        self._position_qty[self.config.hl_instrument_id] = hl_qty
        self._submit_market(self.config.binance_instrument_id, OrderSide.BUY, binance_qty)
        self._submit_market(self.config.hl_instrument_id, OrderSide.SELL, hl_qty)

    def _exit_long_hl_short_binance(self) -> None:
        self._state = "exiting_long_hl_short_binance"
        self._closing_instruments = {self.config.hl_instrument_id, self.config.binance_instrument_id}
        self._submit_market(
            self.config.hl_instrument_id,
            OrderSide.SELL,
            self._position_qty.get(self.config.hl_instrument_id),
            reduce_only=True,
        )
        self._submit_market(
            self.config.binance_instrument_id,
            OrderSide.BUY,
            self._position_qty.get(self.config.binance_instrument_id),
            reduce_only=True,
        )

    def _exit_long_binance_short_hl(self) -> None:
        self._state = "exiting_long_binance_short_hl"
        self._closing_instruments = {self.config.binance_instrument_id, self.config.hl_instrument_id}
        self._submit_market(
            self.config.binance_instrument_id,
            OrderSide.SELL,
            self._position_qty.get(self.config.binance_instrument_id),
            reduce_only=True,
        )
        self._submit_market(
            self.config.hl_instrument_id,
            OrderSide.BUY,
            self._position_qty.get(self.config.hl_instrument_id),
            reduce_only=True,
        )


@dataclass(frozen=True)
class TickerSource:
    exchange: str
    symbol: str
    instrument_id: InstrumentId


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def date_stems(start: datetime, end: datetime) -> list[str]:
    current = start.date()
    last = end.date()
    stems = []
    while current <= last:
        stems.append(current.isoformat())
        current += timedelta(days=1)
    return stems


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


def source_parquet_paths(
    source: TickerSource,
    start: datetime,
    end: datetime,
    allow_missing: bool = False,
) -> list[Path]:
    folder = DATA_FOLDER / source.exchange / "derivative_ticker" / source.symbol
    paths = [folder / f"{stem}.parquet" for stem in date_stems(start, end)]
    missing = [path for path in paths if not path.exists()]
    if missing and not allow_missing:
        raise FileNotFoundError(f"Missing derivative ticker files for {source.exchange}/{source.symbol}: {missing[:3]}")
    return [path for path in paths if path.exists()]


def load_mark_prices(
    source: TickerSource,
    start: datetime,
    end: datetime,
    signal_latency_nanos: int = 0,
) -> list[MarkPriceUpdate]:
    start_ns = int(start.timestamp() * NANOSECONDS_PER_SECOND)
    end_ns = int(end.timestamp() * NANOSECONDS_PER_SECOND)
    raw_start_us = math.ceil((start_ns - signal_latency_nanos) / 1_000)
    raw_end_us = math.floor((end_ns - signal_latency_nanos) / 1_000)
    if raw_start_us > raw_end_us:
        return []

    raw_start = datetime.fromtimestamp(raw_start_us / MICROSECONDS_PER_SECOND, tz=timezone.utc)
    raw_end = datetime.fromtimestamp(raw_end_us / MICROSECONDS_PER_SECOND, tz=timezone.utc)
    source_parquet_paths(source, start, end)
    paths = source_parquet_paths(source, raw_start, raw_end, allow_missing=True)
    if not paths:
        return []

    frame = (
        pl.scan_parquet(paths)
        .select("timestamp", "mark_price")
        .drop_nulls(["mark_price"])
        .filter(pl.col("timestamp").is_between(raw_start_us, raw_end_us, closed="both"))
        .sort("timestamp")
        .unique(subset="timestamp", keep="last", maintain_order=True)
        .collect()
    )

    updates = []
    for ts_us, mark_price in frame.iter_rows():
        ts_event_ns = int(ts_us) * 1_000
        ts_init_ns = ts_event_ns + signal_latency_nanos
        price = Price.from_str(f"{float(mark_price):.8f}")
        updates.append(MarkPriceUpdate(source.instrument_id, price, ts_event_ns, ts_init_ns))
    return updates


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
        price_precision=8,
        size_precision=8,
        price_increment=Price.from_str("0.00000001"),
        size_increment=Quantity.from_str("0.00000001"),
        min_quantity=Quantity.from_str("0.00000001"),
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
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1_000_000, Currency.from_str("USDT"))],
        latency_model=order_latency_model,
        book_type=BookType.L2_MBP,
        trade_execution=False,
        liquidity_consumption=True,
    )
    engine.add_venue(
        venue=Venue("HYPERLIQUID"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1_000_000, Currency.from_str("USDC"))],
        latency_model=order_latency_model,
        book_type=BookType.L2_MBP,
        trade_execution=False,
        liquidity_consumption=True,
    )

    engine.add_instrument(
        make_perpetual(
            BINANCE_INSTRUMENT_ID,
            raw_symbol="SPCXUSDT",
            base_currency="SPCX",
            quote_currency="USDT",
            settlement_currency="USDT",
            maker_fee="0.0002",
            taker_fee="0.0005",
        ),
    )
    engine.add_instrument(
        make_perpetual(
            HL_INSTRUMENT_ID,
            raw_symbol="XYZ-SPCX",
            base_currency="SPCX",
            quote_currency="USDC",
            settlement_currency="USDC",
            maker_fee="0.00003",
            taker_fee="0.00009",
        ),
    )

    strategy = CrossVenueMeanReversion(
        CrossVenueMeanReversionConfig(
            binance_instrument_id=BINANCE_INSTRUMENT_ID,
            hl_instrument_id=HL_INSTRUMENT_ID,
            fallback_qty=Decimal(str(args.qty)),
            target_notional=Decimal(str(args.target_notional)),
            max_qty=Decimal(str(args.max_qty)),
            lookback_minutes=args.lookback_minutes,
            min_samples=args.min_samples,
            min_std_pct=args.min_std_pct,
            entry_z=args.entry_z,
            exit_z=args.exit_z,
            flatten_on_stop=args.flatten_on_stop,
        ),
    )
    engine.add_strategy(strategy)
    return engine


def load_backtest_data(
    start: datetime,
    end: datetime,
    signal_latency_nanos: int = 0,
    show_progress: bool = True,
) -> list:
    catalog = ParquetDataCatalog(str(CATALOG_PATH))
    data = []
    windows = day_windows(start, end)
    progress = tqdm(windows, desc="Loading backtest days", unit="day", disable=not show_progress)
    for day_start, day_end in progress:
        day_count = 0
        for instrument_id in [BINANCE_INSTRUMENT_ID, HL_INSTRUMENT_ID]:
            deltas = catalog.order_book_deltas(
                instrument_ids=[str(instrument_id)],
                start=day_start.isoformat(),
                end=day_end.isoformat(),
            )
            data.extend(deltas)
            day_count += len(deltas)

        for source in [
            TickerSource("binance-futures", "SPCXUSDT", BINANCE_INSTRUMENT_ID),
            TickerSource("hyperliquid", "XYZ-SPCX", HL_INSTRUMENT_ID),
        ]:
            marks = load_mark_prices(source, day_start, day_end, signal_latency_nanos)
            data.extend(marks)
            day_count += len(marks)

        progress.set_postfix_str(f"{day_start.date()} +{day_count:,} total={len(data):,}")
    return data


def load_backtest_data_window(
    catalog: ParquetDataCatalog,
    start: datetime,
    end: datetime,
    signal_latency_nanos: int = 0,
) -> list:
    data = []
    for instrument_id in [BINANCE_INSTRUMENT_ID, HL_INSTRUMENT_ID]:
        data.extend(
            catalog.order_book_deltas(
                instrument_ids=[str(instrument_id)],
                start=start.isoformat(),
                end=end.isoformat(),
            ),
        )

    for source in [
        TickerSource("binance-futures", "SPCXUSDT", BINANCE_INSTRUMENT_ID),
        TickerSource("hyperliquid", "XYZ-SPCX", HL_INSTRUMENT_ID),
    ]:
        data.extend(load_mark_prices(source, start, end, signal_latency_nanos))

    return data


def run_streaming_backtest(
    engine: BacktestEngine,
    start: datetime,
    end: datetime,
    signal_latency_nanos: int = 0,
) -> int:
    catalog = ParquetDataCatalog(str(CATALOG_PATH))
    total_count = 0
    ran = False
    windows = day_windows(start, end)
    progress = tqdm(windows, desc="Running backtest days", unit="day")
    for day_start, day_end in progress:
        data = load_backtest_data_window(catalog, day_start, day_end, signal_latency_nanos)
        day_count = len(data)
        total_count += day_count
        progress.set_postfix_str(f"{day_start.date()} +{day_count:,} total={total_count:,}")
        if not data:
            continue

        engine.add_data(data, sort=True)
        engine.run(start=start, end=end, streaming=True)
        ran = True
        engine.clear_data()
        del data
        gc.collect()

    if ran:
        engine.end()
    return total_count


def hourly_boundaries_us(start: datetime, end: datetime) -> list[int]:
    current = datetime.combine(start.date(), datetime.min.time(), tzinfo=timezone.utc)
    if current < start:
        current += timedelta(hours=1)

    boundaries = []
    while current <= end:
        boundaries.append(int(current.timestamp() * 1_000_000))
        current += timedelta(hours=1)
    return boundaries


def load_funding_events_for_source(source: TickerSource, start: datetime, end: datetime) -> pl.DataFrame:
    start_us = int(start.timestamp() * 1_000_000)
    end_us = int(end.timestamp() * 1_000_000)
    ticker = (
        pl.scan_parquet(source_parquet_paths(source, start, end))
        .select("timestamp", "funding_timestamp", "funding_rate", "mark_price")
        .drop_nulls(["funding_rate", "mark_price"])
    )

    if source.exchange == "binance-futures":
        return (
            ticker.filter(
                pl.col("funding_timestamp").is_not_null()
                & pl.col("funding_timestamp").is_between(start_us, end_us, closed="both")
                & (pl.col("timestamp") <= pl.col("funding_timestamp"))
            )
            .sort(["funding_timestamp", "timestamp"])
            .group_by("funding_timestamp", maintain_order=True)
            .last()
            .select(
                pl.lit(str(source.instrument_id)).alias("instrument_id"),
                pl.col("funding_timestamp").alias("payment_ts_us"),
                pl.col("timestamp").alias("observed_ts_us"),
                pl.col("funding_rate"),
                pl.col("mark_price"),
            )
            .collect()
        )

    boundaries = pl.DataFrame({"payment_ts_us": hourly_boundaries_us(start, end)})
    if boundaries.is_empty():
        return pl.DataFrame(
            schema={
                "instrument_id": pl.String,
                "payment_ts_us": pl.Int64,
                "observed_ts_us": pl.Int64,
                "funding_rate": pl.Float64,
                "mark_price": pl.Float64,
            },
        )

    ticker_frame = (
        ticker.filter(pl.col("timestamp") <= end_us)
        .sort("timestamp")
        .collect()
    )
    return (
        boundaries.sort("payment_ts_us")
        .join_asof(ticker_frame, left_on="payment_ts_us", right_on="timestamp", strategy="backward")
        .drop_nulls(["funding_rate", "mark_price"])
        .select(
            pl.lit(str(source.instrument_id)).alias("instrument_id"),
            "payment_ts_us",
            pl.col("timestamp").alias("observed_ts_us"),
            "funding_rate",
            "mark_price",
        )
    )


def load_funding_events(start: datetime, end: datetime) -> pl.DataFrame:
    frames = [
        load_funding_events_for_source(TickerSource("binance-futures", "SPCXUSDT", BINANCE_INSTRUMENT_ID), start, end),
        load_funding_events_for_source(TickerSource("hyperliquid", "XYZ-SPCX", HL_INSTRUMENT_ID), start, end),
    ]
    return pl.concat(frames, how="vertical").sort(["payment_ts_us", "instrument_id"])


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def compute_funding_payments(positions, start: datetime, end: datetime) -> pl.DataFrame:
    funding_events = load_funding_events(start, end)
    payment_rows = []
    events_by_instrument = {
        instrument_id: frame.sort("payment_ts_us").to_dicts()
        for instrument_id, frame in funding_events.partition_by("instrument_id", as_dict=True).items()
    }
    end_ns = int(end.timestamp() * 1_000_000_000)

    for row in positions.reset_index().to_dict("records"):
        instrument_id = row["instrument_id"]
        if instrument_id not in events_by_instrument:
            continue

        open_ns = int(row["ts_init"])
        close_ns = int(row["ts_last"]) if not _is_missing(row.get("closing_order_id")) else end_ns
        if close_ns <= open_ns:
            continue

        qty = float(row["peak_qty"] if not _is_missing(row.get("peak_qty")) else row["quantity"])
        avg_px_open = float(row["avg_px_open"])
        direction = 1.0 if row["entry"] == "BUY" else -1.0
        settlement_currency = "USDT" if instrument_id.endswith(".BINANCE") else "USDC"

        for event in events_by_instrument[instrument_id]:
            payment_ts_ns = int(event["payment_ts_us"]) * 1_000
            if not (open_ns <= payment_ts_ns < close_ns):
                continue

            rate = float(event["funding_rate"])
            mark_price = float(event["mark_price"])
            notional = qty * mark_price
            funding_payment = -direction * notional * rate
            entry_notional = qty * avg_px_open
            payment_rows.append(
                {
                    "position_id": row["position_id"],
                    "instrument_id": instrument_id,
                    "payment_ts_ns": payment_ts_ns,
                    "payment_time": datetime.fromtimestamp(payment_ts_ns / 1_000_000_000, tz=timezone.utc).isoformat(),
                    "observed_ts_ns": int(event["observed_ts_us"]) * 1_000,
                    "entry": row["entry"],
                    "quantity": qty,
                    "mark_price": mark_price,
                    "notional": notional,
                    "funding_rate": rate,
                    "funding_payment": funding_payment,
                    "funding_return": funding_payment / entry_notional if entry_notional else 0.0,
                    "settlement_currency": settlement_currency,
                }
            )

    payments = pl.DataFrame(payment_rows) if payment_rows else pl.DataFrame(
        schema={
            "position_id": pl.String,
            "instrument_id": pl.String,
            "payment_ts_ns": pl.Int64,
            "payment_time": pl.String,
            "observed_ts_ns": pl.Int64,
            "entry": pl.String,
            "quantity": pl.Float64,
            "mark_price": pl.Float64,
            "notional": pl.Float64,
            "funding_rate": pl.Float64,
            "funding_payment": pl.Float64,
            "funding_return": pl.Float64,
            "settlement_currency": pl.String,
        },
    )

    return payments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Nautilus cross-venue mean-reversion smoke backtest.")
    parser.add_argument("--start", default="2026-05-22T00:00:00Z")
    parser.add_argument("--end", default="2026-06-07T00:00:00Z")
    parser.add_argument("--qty", type=Decimal, default=Decimal("1"), help="Fallback fixed quantity when --target-notional is 0.")
    parser.add_argument("--target-notional", type=Decimal, default=Decimal("1000"), help="USD-equivalent notional target per leg.")
    parser.add_argument("--max-qty", type=Decimal, default=Decimal("100"), help="Maximum base quantity per leg; set 0 to disable.")
    parser.add_argument("--lookback-minutes", type=float, default=60.0, help="Rolling diff window length for z-score signals.")
    parser.add_argument("--min-samples", type=int, default=100, help="Minimum diff observations before signals are active.")
    parser.add_argument("--min-std-pct", type=float, default=1e-6, help="Minimum rolling spread std in percent before signals are active.")
    parser.add_argument("--entry-z", type=float, default=2.0, help="Enter when rolling diff z-score crosses +/- this value.")
    parser.add_argument("--exit-z", type=float, default=0.25, help="Exit when rolling diff z-score reverts inside +/- this value.")
    parser.add_argument("--entry-upper", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--entry-lower", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--exit-diff", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--flatten-on-stop", action="store_true")
    parser.add_argument("--log-level", default="ERROR")
    parser.add_argument("--bypass-logging", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="Directory to save full orders/fills/positions reports.")
    parser.add_argument("--data-latency-ms", type=float, default=5.0, help="Delay before signal data is received.")
    parser.add_argument("--calc-latency-ms", type=float, default=1.0, help="Delay for strategy calculation after data receipt.")
    parser.add_argument("--order-latency-ms", type=float, default=5.0, help="Delay for order commands to reach the exchange.")
    return parser.parse_args()


def validate_strategy_args(args: argparse.Namespace) -> None:
    if args.target_notional < 0:
        raise ValueError("--target-notional must be non-negative")
    if args.max_qty < 0:
        raise ValueError("--max-qty must be non-negative")
    if args.lookback_minutes <= 0:
        raise ValueError("--lookback-minutes must be positive")
    if args.min_samples < 2:
        raise ValueError("--min-samples must be at least 2")
    if not math.isfinite(args.min_std_pct) or args.min_std_pct < 0:
        raise ValueError("--min-std-pct must be a non-negative finite number")
    if not math.isfinite(args.entry_z) or args.entry_z <= 0:
        raise ValueError("--entry-z must be a positive finite number")
    if not math.isfinite(args.exit_z) or args.exit_z < 0:
        raise ValueError("--exit-z must be a non-negative finite number")
    if args.entry_upper is not None or args.entry_lower is not None or args.exit_diff is not None:
        print("Ignoring legacy static threshold args; dynamic z-score args are active.")


def save_reports(output_dir: Path, orders, fills, positions, result, start: datetime, end: datetime) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    orders.to_csv(output_dir / "orders.csv")
    fills.to_csv(output_dir / "fills.csv")
    positions.to_csv(output_dir / "positions.csv")
    (output_dir / "result.txt").write_text(f"{result}\n", encoding="utf-8")
    funding_payments = compute_funding_payments(positions, start, end)
    funding_payments.write_csv(output_dir / "funding_payments.csv")

    result_json = getattr(result, "to_dict", None)
    if callable(result_json):
        (output_dir / "result.json").write_text(
            json.dumps(result_json(), indent=2, default=str),
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    validate_strategy_args(args)
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
        "Dynamic strategy: "
        f"lookback={args.lookback_minutes:g}min, "
        f"min_samples={args.min_samples}, "
        f"entry_z={args.entry_z:g}, "
        f"exit_z={args.exit_z:g}, "
        f"target_notional={args.target_notional}, "
        f"max_qty={args.max_qty}",
    )

    engine = build_engine(args, order_latency_nanos)
    data_count = run_streaming_backtest(engine, start, end, signal_latency_nanos)
    print(f"Processed {data_count:,} Nautilus data objects")

    fills = engine.trader.generate_order_fills_report()
    positions = engine.trader.generate_positions_report()
    orders = engine.trader.generate_orders_report()
    result = engine.get_result()

    if args.output_dir is not None:
        save_reports(args.output_dir, orders, fills, positions, result, start, end)
        print(f"\nSaved reports to {args.output_dir}")

    print("\nResult:")
    print(result)
    engine.dispose()


if __name__ == "__main__":
    main()
