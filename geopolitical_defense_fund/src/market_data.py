"""Market data retrieval through free yfinance sources."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
import yfinance as yf

from .utils import as_float, clean_ticker, dedupe_preserve_order, get_logger, latest_valid_value


LOGGER = get_logger("market_data")


def fetch_price_history(
    tickers: Iterable[str],
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch adjusted close price history for the provided tickers."""
    ticker_list = dedupe_preserve_order(clean_ticker(ticker) for ticker in tickers)
    if not ticker_list:
        return pd.DataFrame()

    try:
        raw = yf.download(
            tickers=ticker_list,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception as exc:  # noqa: BLE001 - yfinance raises several transport errors
        LOGGER.exception("Failed to download price history: %s", exc)
        return pd.DataFrame(columns=ticker_list)

    prices = _extract_adjusted_close(raw, ticker_list)
    if prices.empty:
        return pd.DataFrame(columns=ticker_list)

    prices = prices.apply(pd.to_numeric, errors="coerce").sort_index()
    return prices.reindex(columns=ticker_list)


def fetch_fundamental_data(
    tickers: Iterable[str],
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fetch latest price and common fundamental fields from yfinance."""
    ticker_list = dedupe_preserve_order(clean_ticker(ticker) for ticker in tickers)
    rows: list[dict[str, Any]] = []

    for ticker in ticker_list:
        row: dict[str, Any] = {
            "ticker": ticker,
            "latest_price": None,
            "market_cap": None,
            "trailing_pe": None,
            "forward_pe": None,
            "beta": None,
            "currency": None,
            "data_error": None,
        }
        try:
            asset = yf.Ticker(ticker)
            fast_info = _safe_fast_info(asset)
            info = _safe_info(asset)

            row["latest_price"] = _first_float(
                fast_info.get("last_price"),
                fast_info.get("lastPrice"),
                fast_info.get("regular_market_price"),
                info.get("currentPrice"),
                info.get("regularMarketPrice"),
                _latest_price_from_history(ticker, prices),
            )
            row["market_cap"] = _first_float(
                fast_info.get("market_cap"),
                fast_info.get("marketCap"),
                info.get("marketCap"),
            )
            row["trailing_pe"] = _first_float(
                info.get("trailingPE"),
                info.get("trailingPe"),
            )
            row["forward_pe"] = _first_float(
                info.get("forwardPE"),
                info.get("forwardPe"),
            )
            row["beta"] = _first_float(info.get("beta"))
            row["currency"] = (
                fast_info.get("currency")
                or info.get("currency")
                or info.get("financialCurrency")
            )
        except Exception as exc:  # noqa: BLE001 - keep the dashboard alive per ticker
            LOGGER.warning("Fundamental data failed for %s: %s", ticker, exc)
            row["data_error"] = str(exc)
            row["latest_price"] = _latest_price_from_history(ticker, prices)

        rows.append(row)

    return pd.DataFrame(rows)


def _extract_adjusted_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract adjusted close data from yfinance's single or multi-index output."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=tickers)

    if isinstance(raw.columns, pd.MultiIndex):
        level_zero = raw.columns.get_level_values(0)
        if "Adj Close" in level_zero:
            adjusted = raw["Adj Close"]
        elif "Close" in level_zero:
            adjusted = raw["Close"]
        else:
            return pd.DataFrame(columns=tickers)
        if isinstance(adjusted, pd.Series):
            return adjusted.to_frame(name=tickers[0])
        return adjusted

    field = "Adj Close" if "Adj Close" in raw.columns else "Close"
    if field not in raw.columns:
        return pd.DataFrame(columns=tickers)
    ticker = tickers[0] if len(tickers) == 1 else "value"
    return raw[field].to_frame(name=ticker)


def _safe_fast_info(asset: yf.Ticker) -> dict[str, Any]:
    """Safely access yfinance fast_info as a plain dictionary."""
    try:
        fast_info = asset.fast_info
        return dict(fast_info) if fast_info is not None else {}
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("fast_info unavailable: %s", exc)
        return {}


def _safe_info(asset: yf.Ticker) -> dict[str, Any]:
    """Safely access yfinance info as a plain dictionary."""
    try:
        info = asset.info
        return dict(info) if info is not None else {}
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("info unavailable: %s", exc)
        return {}


def _first_float(*values: object) -> float | None:
    """Return the first finite float in a sequence of possible values."""
    for value in values:
        number = as_float(value)
        if number is not None:
            return number
    return None


def _latest_price_from_history(ticker: str, prices: pd.DataFrame | None) -> float | None:
    """Use downloaded adjusted close history as a latest-price fallback."""
    if prices is None or ticker not in prices:
        return None
    return latest_valid_value(prices[ticker])
