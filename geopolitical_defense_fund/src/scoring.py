"""Cross-sectional scoring model for defense and aerospace candidates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import get_scoring_weights
from .utils import as_float, get_logger, latest_valid_value


LOGGER = get_logger("scoring")
NEUTRAL_SCORE = 50.0


def score_universe(
    universe: pd.DataFrame,
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Score each defense/aerospace candidate from 0 to 100."""
    price_config = config["price_history"]
    min_price_points = int(price_config.get("min_price_points", 126))
    annualization_days = int(price_config.get("annualization_days", 252))
    momentum_windows = price_config.get("momentum_windows", {})
    three_month_days = int(momentum_windows.get("three_month", 63))
    six_month_days = int(momentum_windows.get("six_month", 126))
    volatility_window = int(price_config.get("volatility_window", 90))
    max_usable_pe = float(config.get("valuation", {}).get("max_usable_pe", 300))

    fundamental_lookup = fundamentals.set_index("ticker") if not fundamentals.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []

    for _, security in universe.iterrows():
        ticker = security["ticker"]
        price_series = prices[ticker].dropna() if ticker in prices else pd.Series(dtype=float)
        has_enough_history = len(price_series) >= min_price_points

        row: dict[str, Any] = {
            "ticker": ticker,
            "company_name": security.get("company_name", ""),
            "category": security.get("category", ""),
            "exchange": security.get("exchange", ""),
            "country": security.get("country", ""),
            "notes": security.get("notes", ""),
            "price_points": int(len(price_series)),
            "data_status": "ok" if has_enough_history else "insufficient price history",
            "return_3m": np.nan,
            "return_6m": np.nan,
            "volatility_90d": np.nan,
            "latest_price": latest_valid_value(price_series),
            "market_cap": np.nan,
            "trailing_pe": np.nan,
            "forward_pe": np.nan,
            "beta": np.nan,
        }

        if has_enough_history:
            row["return_3m"] = _window_return(price_series, three_month_days)
            row["return_6m"] = _window_return(price_series, six_month_days)
            row["volatility_90d"] = _annualized_volatility(
                price_series,
                window=volatility_window,
                annualization_days=annualization_days,
            )

        if not fundamental_lookup.empty and ticker in fundamental_lookup.index:
            fundamentals_row = fundamental_lookup.loc[ticker]
            row["latest_price"] = _first_float_or_nan(
                fundamentals_row.get("latest_price"),
                row["latest_price"],
            )
            row["market_cap"] = _float_or_nan(fundamentals_row.get("market_cap"))
            row["trailing_pe"] = _clean_pe(
                fundamentals_row.get("trailing_pe"),
                max_usable_pe=max_usable_pe,
            )
            row["forward_pe"] = _clean_pe(
                fundamentals_row.get("forward_pe"),
                max_usable_pe=max_usable_pe,
            )
            row["beta"] = _float_or_nan(fundamentals_row.get("beta"))

        rows.append(row)

    scores = pd.DataFrame(rows)
    if scores.empty:
        return scores

    valid_price_mask = (
        (scores["data_status"] == "ok")
        & scores["return_3m"].notna()
        & scores["return_6m"].notna()
        & scores["volatility_90d"].notna()
    )
    scores.loc[~valid_price_mask & (scores["data_status"] == "ok"), "data_status"] = (
        "missing return or volatility inputs"
    )

    scores["momentum_raw"] = scores[["return_3m", "return_6m"]].mean(axis=1)
    scores["momentum_score"] = normalize_score(scores["momentum_raw"], higher_is_better=True)

    volatility_score = normalize_score(scores["volatility_90d"], higher_is_better=False)
    beta_score = normalize_score(scores["beta"], higher_is_better=False)
    scores["risk_score"] = np.where(
        scores["beta"].notna(),
        (volatility_score + beta_score) / 2,
        volatility_score,
    )

    trailing_pe_score = normalize_score(scores["trailing_pe"], higher_is_better=False)
    forward_pe_score = normalize_score(scores["forward_pe"], higher_is_better=False)
    scores["valuation_score"] = (trailing_pe_score + forward_pe_score) / 2

    log_market_cap = np.log10(pd.to_numeric(scores["market_cap"], errors="coerce"))
    scores["stability_score"] = normalize_score(log_market_cap, higher_is_better=True)

    weights = get_scoring_weights(config)
    scores["total_score"] = (
        weights["momentum"] * scores["momentum_score"]
        + weights["risk"] * scores["risk_score"]
        + weights["valuation"] * scores["valuation_score"]
        + weights["stability"] * scores["stability_score"]
    )

    score_columns = [
        "momentum_score",
        "risk_score",
        "valuation_score",
        "stability_score",
        "total_score",
    ]
    scores.loc[~valid_price_mask, score_columns] = np.nan

    ordered_columns = [
        "ticker",
        "company_name",
        "category",
        "exchange",
        "country",
        "notes",
        "price_points",
        "data_status",
        "return_3m",
        "return_6m",
        "volatility_90d",
        "latest_price",
        "beta",
        "trailing_pe",
        "forward_pe",
        "market_cap",
        "momentum_score",
        "risk_score",
        "valuation_score",
        "stability_score",
        "total_score",
    ]
    return scores[ordered_columns].sort_values("total_score", ascending=False, na_position="last")


def normalize_score(
    values: pd.Series,
    *,
    higher_is_better: bool,
    neutral_score: float = NEUTRAL_SCORE,
) -> pd.Series:
    """Normalize a numeric Series to a 0-100 cross-sectional score."""
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    result = pd.Series(neutral_score, index=values.index, dtype=float)
    valid = numeric.dropna()

    if valid.empty:
        return result

    minimum = float(valid.min())
    maximum = float(valid.max())
    if np.isclose(minimum, maximum):
        result.loc[valid.index] = neutral_score
        return result

    scaled = (valid - minimum) / (maximum - minimum) * 100
    if not higher_is_better:
        scaled = 100 - scaled
    result.loc[valid.index] = scaled.clip(0, 100)
    return result


def _window_return(prices: pd.Series, window: int) -> float:
    """Calculate simple trailing return over a configured trading-day window."""
    cleaned = pd.to_numeric(prices, errors="coerce").dropna()
    if len(cleaned) < window:
        return np.nan
    starting_price = float(cleaned.iloc[-window])
    ending_price = float(cleaned.iloc[-1])
    if starting_price <= 0:
        return np.nan
    return ending_price / starting_price - 1


def _annualized_volatility(
    prices: pd.Series,
    *,
    window: int,
    annualization_days: int,
) -> float:
    """Calculate annualized trailing volatility from daily returns."""
    returns = pd.to_numeric(prices, errors="coerce").dropna().pct_change().dropna()
    if len(returns) < max(2, window):
        return np.nan
    return float(returns.tail(window).std() * np.sqrt(annualization_days))


def _clean_pe(value: object, *, max_usable_pe: float) -> float:
    """Return a usable positive P/E value or NaN."""
    number = as_float(value)
    if number is None or number <= 0 or number > max_usable_pe:
        return np.nan
    return number


def _float_or_nan(value: object) -> float:
    """Return a finite float or NaN without treating zero as missing."""
    number = as_float(value)
    return np.nan if number is None else number


def _first_float_or_nan(*values: object) -> float:
    """Return the first finite float in a sequence or NaN."""
    for value in values:
        number = as_float(value)
        if number is not None:
            return number
    return np.nan
