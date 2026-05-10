"""Shared utilities for the dashboard."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd


LOGGER_NAME = "geopolitical_defense_fund"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the project logger."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the project namespace."""
    base = setup_logging()
    return base if name is None else logging.getLogger(f"{LOGGER_NAME}.{name}")


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    """Return unique non-empty strings while preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def clean_ticker(ticker: object) -> str:
    """Normalize ticker text for downstream yfinance calls."""
    return str(ticker).strip().upper()


def display_ticker(ticker: object) -> str:
    """Return a dashboard-friendly ticker label without exchange suffixes."""
    cleaned = clean_ticker(ticker)
    return cleaned.split(".", maxsplit=1)[0]


def as_float(value: object) -> float | None:
    """Convert a value to a finite float, returning None when unusable."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def latest_valid_value(series: pd.Series | None) -> float | None:
    """Return the latest finite value from a pandas Series."""
    if series is None or series.empty:
        return None
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return None
    return float(cleaned.iloc[-1])


def latest_daily_return(series: pd.Series | None) -> float | None:
    """Return the latest one-day percentage return from a price Series."""
    if series is None or series.empty:
        return None
    returns = pd.to_numeric(series, errors="coerce").dropna().pct_change().dropna()
    if returns.empty:
        return None
    return float(returns.iloc[-1])


def format_currency(value: object, decimals: int = 2) -> str:
    """Format a number as a dollar value for display."""
    number = as_float(value)
    return "N/A" if number is None else f"${number:,.{decimals}f}"


def format_percent(value: object, decimals: int = 2) -> str:
    """Format a decimal return or weight as a percentage."""
    number = as_float(value)
    return "N/A" if number is None else f"{number * 100:,.{decimals}f}%"


def format_number(value: object, decimals: int = 2) -> str:
    """Format a number with thousands separators."""
    number = as_float(value)
    return "N/A" if number is None else f"{number:,.{decimals}f}"


def local_timestamp() -> str:
    """Return a human-readable timestamp in the local runtime timezone."""
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
