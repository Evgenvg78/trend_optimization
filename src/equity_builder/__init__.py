from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import pandas as pd


@dataclass
class EquityConfig:
    point_value: float = 1.0
    commission_per_contract: float = 0.0


def load_log(path: str | Path) -> pd.DataFrame:
    """
    Read raw test log exported as CSV (CP1251, semicolon-separated).
    Numeric columns are normalised to floats and signal strings are stripped.
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        sep=";",
        encoding="cp1251",
        dtype=str,
        na_filter=False,
    )
    df.columns = [col.strip() for col in df.columns]
    if "" in df.columns:
        df = df.drop(columns=[""])

    df["Дата и время"] = df["Дата и время"].str.strip()
    df["datetime"] = pd.to_datetime(
        df["Дата и время"],
        format="%d.%m.%Y %H:%M:%S",
        errors="coerce",
    )

    def _to_float(series: pd.Series) -> pd.Series:
        return (
            pd.to_numeric(
                series.str.replace(" ", "", regex=False)
                .str.replace(",", ".", regex=False),
                errors="coerce",
            )
        )

    numeric_cols = [
        "Цена последней сделки",
        "Средняя цена позиции",
        "Текущее количество контрактов",
        "Доход позиции",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = _to_float(df[col])

    df["Сигнал"] = df["Сигнал"].astype(str).str.strip()
    df = df.loc[~df["Дата и время"].eq("")].reset_index(drop=True)
    return df


def select_last_test(
    df: pd.DataFrame, method: Literal["topdown", "bottomup"] = "topdown"
) -> pd.DataFrame:
    """
    Detect and return rows belonging to the last completed test run within the log.
    """
    if "datetime" not in df.columns:
        raise ValueError("DataFrame must contain 'datetime' column.")

    df = df.reset_index(drop=True)

    if method == "topdown":
        last_boundary = 0
        prev = None
        for idx, ts in enumerate(df["datetime"]):
            if prev is not None and pd.notna(ts) and pd.notna(prev) and ts < prev:
                last_boundary = idx
            prev = ts
        return df.iloc[last_boundary:].copy()

    if method == "bottomup":
        if df.empty:
            return df.copy()
        start = len(df) - 1
        t_next = df.loc[start, "datetime"]
        for idx in range(start - 1, -1, -1):
            ts = df.loc[idx, "datetime"]
            if pd.isna(ts) or pd.isna(t_next) or ts <= t_next:
                start = idx
                t_next = ts
            else:
                break
        return df.iloc[start:].copy()

    raise ValueError("Unknown method. Expected 'topdown' or 'bottomup'.")


def log_to_trades(df_last: pd.DataFrame) -> pd.DataFrame:
    """
    Convert filtered log rows into a trade stream preserving chronological order.
    """
    required_cols = {
        "datetime",
        "Сигнал",
        "Цена последней сделки",
        "Текущее количество контрактов",
    }
    missing = required_cols.difference(df_last.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    data = df_last.sort_values("datetime", kind="stable").reset_index(drop=True)

    side_map = {"Покупка": 1, "Продажа": -1}
    data["side"] = data["Сигнал"].map(side_map).fillna(0).astype(int)
    data["pos_after"] = data["Текущее количество контрактов"].ffill().fillna(0.0)
    data["pos_before"] = data["pos_after"].shift(1).fillna(0.0)
    data["qty"] = (data["pos_after"] - data["pos_before"]).abs()

    trades = data.loc[(data["side"] != 0) & (data["qty"] > 0)].copy()
    trades["price"] = trades["Цена последней сделки"]
    trades = trades.dropna(subset=["datetime", "price"])

    return trades.loc[:, ["datetime", "side", "qty", "price"]].reset_index(drop=True)


def equity_on_events(trades: pd.DataFrame, cfg: EquityConfig) -> pd.DataFrame:
    """
    Build event-driven equity curve based on executed trade prices.
    """
    trades = trades.sort_values("datetime", kind="stable").reset_index(drop=True)

    q = 0.0
    avg = 0.0
    realized = 0.0
    rows = []

    for record in trades.itertuples(index=False):
        dt = record.datetime
        side = int(record.side)
        qty = float(record.qty)
        price = float(record.price)

        q, avg, realized = _apply_trade(side, qty, price, q, avg, realized, cfg)

        if q != 0:
            direction = 1.0 if q > 0 else -1.0
            mtm = (price - avg) * direction * abs(q) * cfg.point_value
        else:
            mtm = 0.0
        equity = realized + mtm

        rows.append(
            {
                "datetime": dt,
                "price": price,
                "pos": q,
                "avg_price": avg,
                "realized": realized,
                "mtm": mtm,
                "equity": equity,
            }
        )

    return pd.DataFrame(rows)


def equity_on_ohlc(
    trades: pd.DataFrame,
    ohlc: pd.DataFrame,
    cfg: EquityConfig,
    tz_shift: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build equity curve marked to the close of each OHLC bar.
    """
    if tz_shift is not None:
        raise NotImplementedError("tz_shift support is not implemented.")

    ohlc = ohlc.copy()
    if "datetime" not in ohlc.columns or "close" not in ohlc.columns:
        raise ValueError("OHLC data must contain 'datetime' and 'close' columns.")

    ohlc["datetime"] = pd.to_datetime(ohlc["datetime"])
    ohlc = ohlc.sort_values("datetime", kind="stable").reset_index(drop=True)

    trades = trades.sort_values("datetime", kind="stable").reset_index(drop=True)
    trade_iter = trades.itertuples(index=False)
    try:
        current_trade = next(trade_iter)
        has_trade = True
    except StopIteration:
        current_trade = None
        has_trade = False

    q = 0.0
    avg = 0.0
    realized = 0.0
    rows = []

    for bar in ohlc.itertuples(index=False):
        ts = bar.datetime
        close_price = float(bar.close)

        while has_trade and current_trade.datetime <= ts:
            side = int(current_trade.side)
            qty = float(current_trade.qty)
            price = float(current_trade.price)
            q, avg, realized = _apply_trade(side, qty, price, q, avg, realized, cfg)

            try:
                current_trade = next(trade_iter)
            except StopIteration:
                has_trade = False

        if q != 0:
            direction = 1.0 if q > 0 else -1.0
            mtm = (close_price - avg) * direction * abs(q) * cfg.point_value
        else:
            mtm = 0.0
        equity = realized + mtm

        rows.append(
            {
                "datetime": ts,
                "close": close_price,
                "pos": q,
                "avg_price": avg,
                "realized": realized,
                "mtm": mtm,
                "equity": equity,
            }
        )

    return pd.DataFrame(rows)


def build_equity(
    log_path: str | Path,
    output_dir: str | Path,
    *,
    cfg: EquityConfig | None = None,
    method: Literal["topdown", "bottomup"] = "topdown",
    ohlc_path: str | Path | None = None,
    ohlc_loader: Optional[Callable[[str | Path], pd.DataFrame]] = None,
    plot: bool = False,
    tz_shift: Optional[str] = None,
) -> Tuple[Path, Optional[Path]]:
    """
    High-level helper: read a log, extract the last test, compute equity,
    and persist CSV artifacts. Returns paths to generated CSV files.
    """
    cfg = cfg or EquityConfig()

    log_df = load_log(log_path)
    last_test_df = select_last_test(log_df, method=method)
    trades = log_to_trades(last_test_df)

    if trades.empty:
        events_equity = pd.DataFrame(
            columns=["datetime", "price", "pos", "avg_price", "realized", "mtm", "equity"]
        )
        minute_equity = None
    else:
        events_equity = equity_on_events(trades, cfg)

        minute_equity = None
        if ohlc_path is not None:
            ohlc_df = _load_ohlc(ohlc_path, loader=ohlc_loader)
            minute_equity = equity_on_ohlc(trades, ohlc_df, cfg, tz_shift=tz_shift)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events_path = output_dir / "equity_events.csv"
    events_equity.to_csv(events_path, index=False)

    minute_path: Optional[Path] = None
    if minute_equity is not None:
        minute_path = output_dir / "equity_minute.csv"
        minute_equity.to_csv(minute_path, index=False)

    if plot and not events_equity.empty:
        _plot_equity(
            events_equity,
            minute_equity,
        )

    return events_path, minute_path


def _apply_trade(
    side: int,
    qty: float,
    price: float,
    q: float,
    avg: float,
    realized: float,
    cfg: EquityConfig,
) -> Tuple[float, float, float]:
    realized -= cfg.commission_per_contract * qty

    if q == 0.0 or q * side >= 0:
        new_abs = abs(q) + qty
        if new_abs > 0:
            avg = (abs(q) * avg + qty * price) / new_abs
        q += side * qty
        return q, avg, realized

    closing = min(abs(q), qty)
    direction = 1.0 if q > 0 else -1.0
    realized += (price - avg) * direction * closing * cfg.point_value
    q -= direction * closing

    residual = qty - closing
    if residual > 0:
        q = side * residual
        avg = price
    elif abs(q) < 1e-12:
        avg = 0.0

    if abs(q) < 1e-12:
        q = 0.0

    return q, avg, realized


def _load_ohlc(path: str | Path, loader: Optional[Callable[[str | Path], pd.DataFrame]] = None) -> pd.DataFrame:
    """
    Load OHLC data either via a custom loader (e.g. CSVLoader) or pandas.read_csv.
    """
    if loader is not None:
        ohlc_df = loader(path)
        if not isinstance(ohlc_df, pd.DataFrame):
            raise TypeError("Custom loader must return a pandas DataFrame.")
        return ohlc_df

    return pd.read_csv(path)


def _plot_equity(events: pd.DataFrame, minute: Optional[pd.DataFrame]) -> None:
    """
    Plot equity curves for quick inspection.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("plot=True requires matplotlib to be installed.") from exc

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(events["datetime"], events["equity"], label="Equity (events)")

    if minute is not None and not minute.empty:
        ax.plot(minute["datetime"], minute["equity"], label="Equity (minute)")

    ax.set_xlabel("Datetime")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()
