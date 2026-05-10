"""Fund performance and benchmark analytics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .utils import get_logger


LOGGER = get_logger("performance")


@dataclass(frozen=True)
class PerformanceResult:
    """Container for NAV history and summary performance metrics."""

    nav_history: pd.DataFrame
    summary: dict[str, Any]
    warnings: list[str]


def parse_fund_start_date(value: str | date | pd.Timestamp) -> pd.Timestamp:
    """Parse the configured fixed fund inception date."""
    start_date = pd.Timestamp(value)
    if pd.isna(start_date):
        raise ValueError("Fund start date is missing or invalid.")
    return start_date.normalize()


def calculate_performance(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    benchmark_ticker: str,
    fund_start_date: pd.Timestamp,
    initial_investment: float,
    annualization_days: int = 252,
) -> PerformanceResult:
    """Calculate fund NAV history, benchmark comparison, and summary metrics."""
    warnings: list[str] = []
    if holdings.empty:
        raise ValueError("Cannot calculate performance without holdings.")
    if prices.empty:
        raise ValueError("Cannot calculate performance without market price history.")

    clean_prices = _prepare_prices(prices)
    prices_since_start = clean_prices[clean_prices.index >= fund_start_date]
    if prices_since_start.empty:
        raise ValueError(
            "No market price history is available on or after the fixed fund start date."
        )

    weights = holdings.set_index("ticker")["weight"].astype(float)
    holding_tickers = [ticker for ticker in weights.index if ticker in prices_since_start.columns]
    missing_tickers = sorted(set(weights.index) - set(holding_tickers))
    if missing_tickers:
        warnings.append(
            "No price history was available for performance calculation: "
            + ", ".join(missing_tickers)
        )

    if not holding_tickers:
        raise ValueError("None of the selected holdings had usable price history.")

    selected_prices = prices_since_start[holding_tickers].ffill()
    selected_prices = selected_prices.dropna(how="all")
    if selected_prices.empty:
        raise ValueError("Selected holdings did not have usable prices after forward filling.")

    aligned_weights = weights.reindex(holding_tickers).fillna(0.0)
    available_weight_sum = float(aligned_weights.sum())
    if available_weight_sum <= 0:
        raise ValueError("Available holding weights summed to zero.")
    if not np.isclose(available_weight_sum, 1.0):
        warnings.append("Performance calculation normalized available holding weights to 100%.")
        aligned_weights = aligned_weights / available_weight_sum

    portfolio_returns = selected_prices.pct_change().fillna(0.0).dot(aligned_weights)
    fund_nav = initial_investment * (1.0 + portfolio_returns).cumprod()

    nav_history = pd.DataFrame(
        {
            "fund_daily_return": portfolio_returns,
            "fund_nav": fund_nav,
        }
    )

    benchmark_series = _benchmark_nav_series(
        prices_since_start,
        benchmark_ticker=benchmark_ticker,
        initial_investment=initial_investment,
    )
    if benchmark_series.empty:
        warnings.append(f"Benchmark data for {benchmark_ticker} was unavailable.")
        nav_history["benchmark_daily_return"] = np.nan
        nav_history["benchmark_nav"] = np.nan
    else:
        nav_history = nav_history.join(benchmark_series, how="left")
        nav_history["benchmark_nav"] = nav_history["benchmark_nav"].ffill()
        nav_history["benchmark_daily_return"] = nav_history["benchmark_daily_return"].fillna(0.0)

    nav_history["fund_drawdown"] = calculate_drawdown(nav_history["fund_nav"])
    nav_history.index.name = "date"

    summary = calculate_summary_metrics(
        nav_history,
        fund_start_date=fund_start_date,
        initial_investment=initial_investment,
        annualization_days=annualization_days,
        benchmark_ticker=benchmark_ticker,
    )
    return PerformanceResult(nav_history=nav_history, summary=summary, warnings=warnings)


def calculate_summary_metrics(
    nav_history: pd.DataFrame,
    *,
    fund_start_date: pd.Timestamp,
    initial_investment: float,
    annualization_days: int,
    benchmark_ticker: str,
) -> dict[str, Any]:
    """Calculate point-in-time fund and benchmark summary metrics."""
    if nav_history.empty:
        raise ValueError("Cannot summarize an empty NAV history.")

    current_nav = float(nav_history["fund_nav"].iloc[-1])
    total_return = current_nav / initial_investment - 1.0
    dollar_gain_loss = current_nav - initial_investment
    daily_return = float(nav_history["fund_daily_return"].iloc[-1])
    annualized_volatility = annualized_volatility_from_returns(
        nav_history["fund_daily_return"],
        annualization_days=annualization_days,
    )
    max_drawdown = float(nav_history["fund_drawdown"].min())

    benchmark_nav = pd.to_numeric(nav_history.get("benchmark_nav"), errors="coerce").dropna()
    benchmark_total_return = (
        float(benchmark_nav.iloc[-1] / initial_investment - 1.0)
        if not benchmark_nav.empty
        else np.nan
    )
    fund_vs_benchmark = (
        total_return - benchmark_total_return if np.isfinite(benchmark_total_return) else np.nan
    )

    return {
        "initial_nav": initial_investment,
        "current_estimated_nav": current_nav,
        "fund_start_date": fund_start_date.date().isoformat(),
        "first_market_data_date": nav_history.index.min().date().isoformat(),
        "latest_market_data_date": nav_history.index.max().date().isoformat(),
        "total_return_since_inception": total_return,
        "dollar_gain_loss_since_inception": dollar_gain_loss,
        "daily_portfolio_return": daily_return,
        "annualized_volatility_since_inception": annualized_volatility,
        "maximum_drawdown_since_inception": max_drawdown,
        "benchmark_ticker": benchmark_ticker,
        "benchmark_total_return_since_inception": benchmark_total_return,
        "fund_minus_benchmark_return": fund_vs_benchmark,
        "relative_performance_label": relative_performance_label(fund_vs_benchmark),
    }


def calculate_drawdown(nav: pd.Series) -> pd.Series:
    """Calculate drawdown from a NAV series."""
    nav_values = pd.to_numeric(nav, errors="coerce")
    running_max = nav_values.cummax()
    return nav_values / running_max - 1.0


def annualized_volatility_from_returns(
    returns: pd.Series,
    *,
    annualization_days: int,
) -> float:
    """Calculate annualized volatility from daily returns."""
    cleaned = pd.to_numeric(returns, errors="coerce").dropna()
    if len(cleaned) <= 1:
        return np.nan
    return float(cleaned.std() * np.sqrt(annualization_days))


def relative_performance_label(fund_minus_benchmark: float | None) -> str:
    """Classify fund performance versus benchmark."""
    if fund_minus_benchmark is None or not np.isfinite(fund_minus_benchmark):
        return "Benchmark comparison unavailable"
    if fund_minus_benchmark > 0:
        return "Outperforming SPY since inception"
    if fund_minus_benchmark < 0:
        return "Underperforming SPY since inception"
    return "Matching SPY since inception"


def _benchmark_nav_series(
    prices: pd.DataFrame,
    *,
    benchmark_ticker: str,
    initial_investment: float,
) -> pd.DataFrame:
    """Calculate benchmark value indexed to the initial investment."""
    if not benchmark_ticker or benchmark_ticker not in prices:
        return pd.DataFrame(columns=["benchmark_daily_return", "benchmark_nav"])

    benchmark_prices = pd.to_numeric(prices[benchmark_ticker], errors="coerce").ffill().dropna()
    if benchmark_prices.empty:
        return pd.DataFrame(columns=["benchmark_daily_return", "benchmark_nav"])

    benchmark_returns = benchmark_prices.pct_change().fillna(0.0)
    benchmark_nav = initial_investment * (1.0 + benchmark_returns).cumprod()
    return pd.DataFrame(
        {
            "benchmark_daily_return": benchmark_returns,
            "benchmark_nav": benchmark_nav,
        }
    )


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Return numeric prices indexed by timezone-naive normalized dates."""
    clean_prices = prices.copy()
    clean_prices.index = pd.to_datetime(clean_prices.index)
    if getattr(clean_prices.index, "tz", None) is not None:
        clean_prices.index = clean_prices.index.tz_convert(None)
    clean_prices.index = clean_prices.index.normalize()
    clean_prices = clean_prices[~clean_prices.index.duplicated(keep="last")]
    return clean_prices.apply(pd.to_numeric, errors="coerce").sort_index()
