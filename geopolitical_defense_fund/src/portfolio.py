"""Portfolio construction and NAV simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .utils import get_logger, latest_daily_return, latest_valid_value


LOGGER = get_logger("portfolio")


class PortfolioConstructionError(ValueError):
    """Raised when no scoreable defense/aerospace candidates are available."""


@dataclass(frozen=True)
class PortfolioResult:
    """Container for portfolio construction outputs."""

    holdings: pd.DataFrame
    selected_defense_tickers: list[str]
    nav_series: pd.DataFrame
    benchmark_series: pd.DataFrame
    estimated_nav: float
    daily_return: float | None
    benchmark_daily_return: float | None
    warnings: list[str]


def construct_portfolio(
    ranking: pd.DataFrame,
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    config: dict[str, Any],
) -> PortfolioResult:
    """Select top defense/aerospace names, add anchors, and simulate NAV."""
    selected_defense = _select_defense_holdings(ranking, config)
    if selected_defense.empty:
        raise PortfolioConstructionError(
            "No valid defense/aerospace stocks could be scored. Check price data availability "
            "or lower the minimum price history threshold in config.yaml."
        )

    warnings: list[str] = []
    holdings = _build_holdings(selected_defense, prices, fundamentals, config, warnings)
    holdings = _normalize_weights(holdings, warnings)

    starting_nav = float(
        config["portfolio"].get(
            "initial_investment",
            config["portfolio"].get("starting_nav", 100.0),
        )
    )
    nav_series = calculate_nav_series(holdings, prices, starting_nav=starting_nav)

    benchmark_ticker = str(config.get("benchmark", {}).get("ticker", "")).upper()
    benchmark_series = calculate_benchmark_series(prices, benchmark_ticker, starting_nav)

    estimated_nav = (
        float(nav_series["nav"].iloc[-1])
        if not nav_series.empty and nav_series["nav"].notna().any()
        else starting_nav
    )
    daily_return = (
        float(nav_series["portfolio_return"].iloc[-1])
        if not nav_series.empty and nav_series["portfolio_return"].notna().any()
        else None
    )
    benchmark_daily_return = (
        float(benchmark_series["benchmark_return"].iloc[-1])
        if not benchmark_series.empty and benchmark_series["benchmark_return"].notna().any()
        else None
    )

    return PortfolioResult(
        holdings=holdings,
        selected_defense_tickers=selected_defense["ticker"].tolist(),
        nav_series=nav_series,
        benchmark_series=benchmark_series,
        estimated_nav=estimated_nav,
        daily_return=daily_return,
        benchmark_daily_return=benchmark_daily_return,
        warnings=warnings,
    )


def calculate_nav_series(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    starting_nav: float,
) -> pd.DataFrame:
    """Calculate constant-weight simulated portfolio NAV."""
    if holdings.empty or prices.empty:
        return pd.DataFrame(columns=["portfolio_return", "nav"])

    weights = holdings.set_index("ticker")["weight"].astype(float)
    available = [ticker for ticker in weights.index if ticker in prices.columns]
    if not available:
        return pd.DataFrame(columns=["portfolio_return", "nav"])

    returns = prices[available].apply(pd.to_numeric, errors="coerce").pct_change()
    aligned_weights = weights.reindex(available).fillna(0.0)
    portfolio_returns = returns.fillna(0.0).dot(aligned_weights)
    nav = starting_nav * (1 + portfolio_returns).cumprod()
    return pd.DataFrame({"portfolio_return": portfolio_returns, "nav": nav})


def calculate_benchmark_series(
    prices: pd.DataFrame,
    benchmark_ticker: str,
    starting_nav: float,
) -> pd.DataFrame:
    """Calculate a normalized benchmark series when benchmark data is available."""
    if not benchmark_ticker or benchmark_ticker not in prices:
        return pd.DataFrame(columns=["benchmark_return", "benchmark_nav"])

    returns = pd.to_numeric(prices[benchmark_ticker], errors="coerce").dropna().pct_change()
    if returns.empty:
        return pd.DataFrame(columns=["benchmark_return", "benchmark_nav"])

    returns = returns.fillna(0.0)
    benchmark_nav = starting_nav * (1 + returns).cumprod()
    return pd.DataFrame({"benchmark_return": returns, "benchmark_nav": benchmark_nav})


def _select_defense_holdings(ranking: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Select the highest-scoring valid defense/aerospace candidates."""
    max_holdings = int(config["portfolio"].get("max_defense_holdings", 5))
    valid = ranking.dropna(subset=["total_score"]).copy()
    valid = valid[valid["total_score"].astype(float) >= 0]
    return valid.sort_values("total_score", ascending=False).head(max_holdings)


def _build_holdings(
    selected_defense: pd.DataFrame,
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    config: dict[str, Any],
    warnings: list[str],
) -> pd.DataFrame:
    """Create the holdings table with defense names and configured anchors."""
    defense_allocation = float(config["portfolio"].get("defense_allocation", 0.70))
    score_weights = _score_based_weights(selected_defense, defense_allocation)
    fundamental_lookup = fundamentals.set_index("ticker") if not fundamentals.empty else pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, security in selected_defense.iterrows():
        ticker = security["ticker"]
        rows.append(
            {
                "ticker": ticker,
                "company_name": security["company_name"],
                "category": security["category"],
                "weight": score_weights.get(ticker, 0.0),
                "latest_price": _latest_price(ticker, prices, fundamental_lookup),
                "daily_return": _daily_return(ticker, prices, warnings),
                "total_score": float(security["total_score"]),
                "reason_selected": "Top-ranked by measurable defense/aerospace score",
            }
        )

    for anchor in config.get("anchor_assets", []):
        ticker = str(anchor["ticker"]).upper()
        rows.append(
            {
                "ticker": ticker,
                "company_name": anchor["company_name"],
                "category": anchor["category"],
                "weight": float(anchor["allocation"]),
                "latest_price": _latest_price(ticker, prices, fundamental_lookup),
                "daily_return": _daily_return(ticker, prices, warnings),
                "total_score": np.nan,
                "reason_selected": anchor.get("reason_selected", "Configured portfolio anchor"),
            }
        )

    return pd.DataFrame(rows)


def _score_based_weights(selected_defense: pd.DataFrame, allocation: float) -> dict[str, float]:
    """Split the defense sleeve according to total scores."""
    scores = selected_defense.set_index("ticker")["total_score"].astype(float).clip(lower=0)
    score_sum = float(scores.sum())
    if score_sum <= 0:
        equal_weight = allocation / len(scores)
        return {ticker: equal_weight for ticker in scores.index}
    return (scores / score_sum * allocation).to_dict()


def _normalize_weights(holdings: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    """Force target weights to sum exactly to one after configuration changes."""
    normalized = holdings.copy()
    total = float(normalized["weight"].sum())
    if total <= 0:
        warnings.append("Configured portfolio weights sum to zero.")
        return normalized

    if not np.isclose(total, 1.0):
        warnings.append(f"Configured weights summed to {total:.4f}; normalized to 100%.")
        normalized["weight"] = normalized["weight"] / total

    residual = 1.0 - float(normalized["weight"].sum())
    if abs(residual) > 1e-10:
        largest_idx = normalized["weight"].idxmax()
        normalized.loc[largest_idx, "weight"] += residual
    return normalized


def _latest_price(
    ticker: str,
    prices: pd.DataFrame,
    fundamental_lookup: pd.DataFrame,
) -> float | None:
    """Return the most reliable latest price available."""
    if not fundamental_lookup.empty and ticker in fundamental_lookup.index:
        value = fundamental_lookup.loc[ticker].get("latest_price")
        if pd.notna(value):
            return float(value)
    if ticker in prices:
        return latest_valid_value(prices[ticker])
    return None


def _daily_return(ticker: str, prices: pd.DataFrame, warnings: list[str]) -> float | None:
    """Return latest daily return and record a warning when unavailable."""
    if ticker not in prices:
        warnings.append(f"No price column returned for {ticker}.")
        return None
    value = latest_daily_return(prices[ticker])
    if value is None:
        warnings.append(f"No daily return could be calculated for {ticker}.")
    return value
