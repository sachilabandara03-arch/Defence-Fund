"""Streamlit dashboard for the simulated Geopolitical Defense Fund."""

from __future__ import annotations

from dataclasses import asdict
from html import escape
import math
from pathlib import Path
from textwrap import dedent
from typing import Any
from urllib.parse import quote

import pandas as pd
import streamlit as st

from src.charts import (
    allocation_pie_chart,
    drawdown_chart,
    nav_comparison_chart,
    normalized_performance_chart,
    score_bar_chart,
)
from src.config import load_config
from src.data_loader import load_universe
from src.market_data import fetch_fundamental_data, fetch_price_history
from src.news import NewsHeadline, fetch_recent_headlines
from src.performance import calculate_performance, parse_fund_start_date
from src.portfolio import PortfolioConstructionError, construct_portfolio
from src.scoring import score_universe
from src.utils import (
    as_float,
    dedupe_preserve_order,
    display_ticker,
    format_currency,
    format_number,
    format_percent,
    local_timestamp,
    setup_logging,
)


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data"
UNIVERSE_PATH = DATA_DIR / "universe.csv"
HOLDINGS_OUTPUT_PATH = DATA_DIR / "latest_holdings.csv"
SCORES_OUTPUT_PATH = DATA_DIR / "latest_scores.csv"
NAV_SUMMARY_OUTPUT_PATH = DATA_DIR / "latest_nav_summary.csv"
NAV_HISTORY_OUTPUT_PATH = DATA_DIR / "latest_nav_history.csv"
NEWS_OUTPUT_PATH = DATA_DIR / "latest_news.csv"

MARKET_CACHE_TTL_SECONDS = 3600
NEWS_CACHE_TTL_SECONDS = 21600


LOGGER = setup_logging()


@st.cache_data(ttl=MARKET_CACHE_TTL_SECONDS, show_spinner=False)
def cached_config(path: str) -> dict[str, Any]:
    """Load dashboard config with Streamlit caching."""
    return load_config(path)


@st.cache_data(ttl=MARKET_CACHE_TTL_SECONDS, show_spinner=False)
def cached_universe(path: str) -> pd.DataFrame:
    """Load editable universe with Streamlit caching."""
    return load_universe(path)


@st.cache_data(ttl=MARKET_CACHE_TTL_SECONDS, show_spinner=False)
def cached_market_data(
    tickers: tuple[str, ...],
    period: str,
    interval: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Fetch market data and preserve the successful fetch timestamp in cache."""
    prices = fetch_price_history(tickers, period=period, interval=interval)
    fundamentals = fetch_fundamental_data(tickers, prices=prices)
    return prices, fundamentals, local_timestamp()


@st.cache_data(ttl=NEWS_CACHE_TTL_SECONDS, show_spinner=False)
def cached_news(
    tickers: tuple[str, ...],
    config: dict[str, Any],
) -> tuple[dict[str, list[NewsHeadline]], str]:
    """Fetch ticker headlines and preserve the successful fetch timestamp in cache."""
    return fetch_recent_headlines(tickers, config), local_timestamp()


def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(
        page_title="Geopolitical Defense Fund Dashboard",
        layout="wide",
    )

    _render_header()

    with st.sidebar:
        st.header("Data")
        st.write("Universe file")
        st.code(str(UNIVERSE_PATH.relative_to(PROJECT_ROOT)))
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    try:
        config = cached_config(str(CONFIG_PATH))
        universe = cached_universe(str(UNIVERSE_PATH))
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Startup failed: %s", exc)
        st.error(str(exc))
        st.stop()

    try:
        dashboard = _build_live_dashboard_data(config, universe)
        _save_outputs(dashboard)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Live dashboard calculation failed: %s", exc)
        try:
            dashboard = _load_saved_dashboard_data()
        except Exception as fallback_exc:  # noqa: BLE001
            st.error(
                "Live data retrieval or portfolio calculation failed, and no complete saved "
                f"CSV fallback was available. Original error: {exc}. "
                f"Fallback error: {fallback_exc}"
            )
            st.stop()
        st.warning(
            "Displaying the most recent saved dashboard files because live market data or "
            f"news refresh failed. Live error: {exc}"
    )

    _render_timestamps(dashboard)
    _render_allocation_overview(dashboard)
    _render_summary(dashboard["summary"])
    _render_holdings(dashboard["holdings"])
    _render_ranking(dashboard["ranking"])
    _render_charts(dashboard)
    _render_news(dashboard["holdings"], dashboard["news_by_ticker"])
    _render_methodology(config)


def _build_live_dashboard_data(config: dict[str, Any], universe: pd.DataFrame) -> dict[str, Any]:
    """Build the full dashboard dataset from live free data sources."""
    anchor_tickers = [str(anchor["ticker"]).upper() for anchor in config.get("anchor_assets", [])]
    benchmark_ticker = str(config.get("benchmark", {}).get("ticker", "SPY")).upper()
    all_tickers = tuple(
        dedupe_preserve_order(list(universe["ticker"]) + anchor_tickers + [benchmark_ticker])
    )

    price_config = config["price_history"]
    with st.spinner("Fetching market and fundamental data from free sources..."):
        prices, fundamentals, market_updated_at = cached_market_data(
            all_tickers,
            period=str(price_config.get("period", "1y")),
            interval=str(price_config.get("interval", "1d")),
        )

    if prices.empty or not prices.dropna(how="all").shape[0]:
        raise ValueError("No live adjusted close price history was returned.")

    _render_data_warnings(all_tickers, prices, fundamentals)

    ranking = score_universe(universe, prices, fundamentals, config)

    try:
        portfolio = construct_portfolio(ranking, prices, fundamentals, config)
    except PortfolioConstructionError:
        raise

    ranking["selected"] = ranking["ticker"].isin(portfolio.selected_defense_tickers)

    portfolio_config = config["portfolio"]
    initial_investment = float(portfolio_config.get("initial_investment", 10_000_000.0))
    fund_start_date = parse_fund_start_date(portfolio_config["fund_start_date"])
    performance = calculate_performance(
        portfolio.holdings,
        prices,
        benchmark_ticker=benchmark_ticker,
        fund_start_date=fund_start_date,
        initial_investment=initial_investment,
        annualization_days=int(price_config.get("annualization_days", 252)),
    )

    summary = dict(performance.summary)
    summary["market_data_updated_at"] = market_updated_at

    holdings = portfolio.holdings.copy()
    holdings["dollar_allocation"] = holdings["weight"] * summary["current_estimated_nav"]

    warnings = portfolio.warnings + performance.warnings
    for warning in warnings:
        st.warning(warning)

    tickers_for_news = tuple(holdings["ticker"].tolist())
    with st.spinner("Checking free news sources..."):
        news_by_ticker, news_updated_at = cached_news(tickers_for_news, config)

    summary["news_updated_at"] = news_updated_at
    summary["dashboard_data_source"] = "live"

    return {
        "source": "live",
        "config": config,
        "prices": prices,
        "holdings": holdings,
        "ranking": ranking,
        "nav_history": performance.nav_history,
        "summary": summary,
        "news_by_ticker": news_by_ticker,
    }


def _render_header() -> None:
    """Render dashboard title, disclaimer, and timestamp."""
    st.title("Geopolitical Defense Fund Dashboard")
    st.caption(
        "Educational research and portfolio simulation only. This dashboard is not financial "
        "advice and does not place real trades."
    )
    st.caption(f"Dashboard rendered: {local_timestamp()}")


def _render_timestamps(dashboard: dict[str, Any]) -> None:
    """Render successful update timestamps and latest data date."""
    summary = dashboard["summary"]
    st.caption(
        "Last successful market data update: "
        f"{summary.get('market_data_updated_at', 'N/A')} | "
        "Last successful news update: "
        f"{summary.get('news_updated_at', 'N/A')} | "
        "Latest market data date used: "
        f"{summary.get('latest_market_data_date', 'N/A')}"
    )


def _render_summary(summary: dict[str, Any]) -> None:
    """Render fund summary metrics."""
    st.subheader("Fund Summary")
    relative_label = summary.get("relative_performance_label", "Benchmark comparison unavailable")
    relative_difference = as_float(summary.get("fund_minus_benchmark_return"))
    if relative_difference is None:
        st.info(relative_label)
    elif relative_difference > 0:
        st.success(relative_label)
    elif relative_difference < 0:
        st.warning(relative_label)
    else:
        st.info(relative_label)

    row_one = st.columns(3)
    row_one[0].metric("Initial investment", format_currency(summary.get("initial_nav"), 0))
    row_one[1].metric("Fund start date", str(summary.get("fund_start_date", "N/A")))
    row_one[2].metric(
        "Current estimated NAV",
        format_currency(summary.get("current_estimated_nav"), 0),
    )

    row_two = st.columns(3)
    row_two[0].metric(
        "Dollar gain/loss",
        format_currency(summary.get("dollar_gain_loss_since_inception"), 0),
    )
    row_two[1].metric(
        "Total return",
        format_percent(summary.get("total_return_since_inception")),
    )
    row_two[2].metric("Daily return", format_percent(summary.get("daily_portfolio_return")))

    row_three = st.columns(4)
    row_three[0].metric(
        "Annualized volatility",
        format_percent(summary.get("annualized_volatility_since_inception")),
    )
    row_three[1].metric(
        "Maximum drawdown",
        format_percent(summary.get("maximum_drawdown_since_inception")),
    )
    row_three[2].metric(
        "SPY total return",
        format_percent(summary.get("benchmark_total_return_since_inception")),
    )
    row_three[3].metric(
        "Fund vs SPY",
        format_percent(summary.get("fund_minus_benchmark_return")),
    )


def _render_allocation_overview(dashboard: dict[str, Any]) -> None:
    """Render the top-of-page interactive allocation view."""
    holdings = dashboard["holdings"]
    if holdings.empty:
        return

    available_tickers = holdings["ticker"].tolist()
    selected_from_query = _ticker_from_display_label(st.query_params.get("holding"), holdings)
    if selected_from_query in available_tickers:
        st.session_state["selected_holding_ticker"] = selected_from_query

    selected_ticker = st.session_state.get("selected_holding_ticker")
    if selected_ticker not in available_tickers:
        selected_ticker = available_tickers[0]

    st.subheader("Portfolio Allocation")
    chart_col, detail_col = st.columns([1.25, 1.0])

    with chart_col:
        _render_clickable_allocation_chart(holdings, selected_ticker)

    with detail_col:
        selected_ticker = st.selectbox(
            "Selected security",
            options=available_tickers,
            index=available_tickers.index(selected_ticker),
            format_func=display_ticker,
        )
        st.session_state["selected_holding_ticker"] = selected_ticker
        _render_selected_holding_details(
            selected_ticker=selected_ticker,
            holdings=holdings,
            summary=dashboard["summary"],
            headlines_by_ticker=dashboard["news_by_ticker"],
        )


def _render_clickable_allocation_chart(
    holdings: pd.DataFrame,
    selected_ticker: str,
) -> None:
    """Render a click-through SVG donut chart for allocation selection."""
    st.markdown(
        _allocation_svg_chart(holdings, selected_ticker=selected_ticker),
        unsafe_allow_html=True,
    )


def _render_selected_holding_details(
    *,
    selected_ticker: str,
    holdings: pd.DataFrame,
    summary: dict[str, Any],
    headlines_by_ticker: dict[str, list[NewsHeadline]],
) -> None:
    """Render details for one selected allocation slice."""
    row = holdings[holdings["ticker"] == selected_ticker].iloc[0]
    st.markdown(f"### {display_ticker(selected_ticker)}")
    st.caption(str(row.get("company_name", "")))

    latest_price = as_float(row.get("latest_price"))
    dollar_allocation = as_float(row.get("dollar_allocation"))
    estimated_shares = (
        dollar_allocation / latest_price
        if latest_price is not None and latest_price > 0 and dollar_allocation is not None
        else None
    )

    metric_one = st.columns(2)
    metric_one[0].metric("Stock price", format_currency(latest_price))
    metric_one[1].metric("Portfolio weight", format_percent(row.get("weight")))

    metric_two = st.columns(2)
    metric_two[0].metric("Dollar allocation", format_currency(dollar_allocation, 0))
    metric_two[1].metric("Estimated shares", format_number(estimated_shares, 0))

    metric_three = st.columns(2)
    metric_three[0].metric("Daily return", format_percent(row.get("daily_return")))
    metric_three[1].metric("Total score", format_number(row.get("total_score")))

    st.write(f"**First invested:** {summary.get('fund_start_date', 'N/A')}")
    st.write(f"**Category:** {row.get('category', 'N/A')}")
    st.write(f"**Selection note:** {row.get('reason_selected', 'N/A')}")

    st.write("**Recent headlines**")
    headlines = headlines_by_ticker.get(selected_ticker, [])
    if not headlines:
        st.write("No recent headlines found.")
        return
    for item in headlines[:3]:
        label = item.title
        if item.publisher:
            label = f"{label} ({item.publisher})"
        if item.link:
            st.markdown(f"- [{label}]({item.link})")
        else:
            st.markdown(f"- {label}")


def _ticker_from_display_label(label: object, holdings: pd.DataFrame) -> str | None:
    """Map a displayed ticker label back to the raw market-data ticker."""
    if not label:
        return None

    display_lookup = {
        display_ticker(ticker): ticker
        for ticker in holdings["ticker"].astype(str).tolist()
    }
    return display_lookup.get(str(label))


def _allocation_svg_chart(holdings: pd.DataFrame, *, selected_ticker: str) -> str:
    """Build an SVG donut chart whose slices are links to selected holdings."""
    prepared = holdings.copy()
    prepared["display_ticker"] = prepared["ticker"].map(display_ticker)
    prepared["weight"] = pd.to_numeric(prepared["weight"], errors="coerce").fillna(0.0)
    total_weight = float(prepared["weight"].sum())
    if total_weight <= 0:
        return "<p>No allocation data available.</p>"

    colors = [
        "#356AC3",
        "#8BBFF0",
        "#EF463D",
        "#F3AAA8",
        "#5BB09E",
        "#91E2A3",
        "#F58D2D",
        "#B784E8",
    ]
    cx, cy = 230.0, 210.0
    outer_radius, inner_radius = 180.0, 78.0
    start_angle = -90.0
    slices: list[str] = []
    legend_items: list[str] = []

    for index, (_, row) in enumerate(prepared.iterrows()):
        weight = float(row["weight"])
        if weight <= 0:
            continue
        end_angle = start_angle + (weight / total_weight) * 360.0
        color = colors[index % len(colors)]
        ticker = str(row["ticker"])
        label = str(row["display_ticker"])
        is_selected = ticker == selected_ticker
        stroke = "#111827" if is_selected else "#ffffff"
        stroke_width = "4" if is_selected else "1.5"
        title = (
            f"{label}: {format_percent(weight)} | "
            f"{format_currency(row.get('dollar_allocation'), 0)} | "
            f"Stock price {format_currency(row.get('latest_price'))}"
        )
        path = _donut_segment_path(cx, cy, outer_radius, inner_radius, start_angle, end_angle)
        label_x, label_y = _polar_point(cx, cy, (outer_radius + inner_radius) / 2, (start_angle + end_angle) / 2)
        href = f"?holding={quote(label)}"
        slices.append(
            dedent(
                f"""
            <a href="{href}" target="_self" aria-label="View {escape(label)} details">
              <path class="allocation-slice" d="{path}" fill="{color}" stroke="{stroke}" stroke-width="{stroke_width}">
                <title>{escape(title)}</title>
              </path>
              <text x="{label_x:.1f}" y="{label_y - 5:.1f}" text-anchor="middle" class="slice-label">{escape(label)}</text>
              <text x="{label_x:.1f}" y="{label_y + 13:.1f}" text-anchor="middle" class="slice-label">{weight * 100:.1f}%</text>
            </a>
            """
            ).strip()
        )
        legend_items.append(
            dedent(
                f"""
            <a class="legend-item{' selected' if is_selected else ''}" href="{href}" target="_self">
              <span class="legend-swatch" style="background:{color};"></span>
              <span>{escape(label)}</span>
            </a>
            """
            ).strip()
        )
        start_angle = end_angle

    return dedent(
        f"""
    <div class="allocation-card">
      <div class="allocation-svg-wrap">
        <svg viewBox="0 0 460 430" role="img" aria-label="Portfolio allocation pie chart">
          <style>
            .allocation-slice {{ transition: filter 0.15s ease, stroke-width 0.15s ease; }}
            .allocation-slice:hover {{ filter: brightness(1.08); stroke: #111827; stroke-width: 4; cursor: pointer; }}
            .slice-label {{ pointer-events: none; font: 700 14px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #1f2937; }}
          </style>
          {''.join(slices)}
          <circle cx="{cx}" cy="{cy}" r="{inner_radius}" fill="#ffffff"></circle>
        </svg>
      </div>
      <div class="allocation-legend">
        {''.join(legend_items)}
      </div>
    </div>
    <style>
      .allocation-card {{
        display: grid;
        grid-template-columns: minmax(360px, 1fr) 140px;
        align-items: center;
        gap: 12px;
        max-width: 760px;
      }}
      .allocation-svg-wrap {{ min-height: 360px; }}
      .allocation-legend {{
        display: flex;
        flex-direction: column;
        gap: 8px;
      }}
      .legend-item {{
        display: flex;
        align-items: center;
        gap: 8px;
        color: #1f2937 !important;
        text-decoration: none !important;
        font: 600 14px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        padding: 4px 6px;
        border-radius: 6px;
      }}
      .legend-item:hover, .legend-item.selected {{ background: #f3f4f6; }}
      .legend-swatch {{
        width: 14px;
        height: 14px;
        border-radius: 3px;
        display: inline-block;
      }}
    </style>
    """
    ).strip()


def _donut_segment_path(
    cx: float,
    cy: float,
    outer_radius: float,
    inner_radius: float,
    start_angle: float,
    end_angle: float,
) -> str:
    """Return an SVG path for a donut-chart segment."""
    outer_start = _polar_point(cx, cy, outer_radius, start_angle)
    outer_end = _polar_point(cx, cy, outer_radius, end_angle)
    inner_end = _polar_point(cx, cy, inner_radius, end_angle)
    inner_start = _polar_point(cx, cy, inner_radius, start_angle)
    large_arc = 1 if end_angle - start_angle > 180 else 0
    return (
        f"M {outer_start[0]:.2f} {outer_start[1]:.2f} "
        f"A {outer_radius:.2f} {outer_radius:.2f} 0 {large_arc} 1 {outer_end[0]:.2f} {outer_end[1]:.2f} "
        f"L {inner_end[0]:.2f} {inner_end[1]:.2f} "
        f"A {inner_radius:.2f} {inner_radius:.2f} 0 {large_arc} 0 {inner_start[0]:.2f} {inner_start[1]:.2f} Z"
    )


def _polar_point(
    cx: float,
    cy: float,
    radius: float,
    angle_degrees: float,
) -> tuple[float, float]:
    """Convert polar chart coordinates to SVG coordinates."""
    angle = math.radians(angle_degrees)
    return cx + radius * math.cos(angle), cy + radius * math.sin(angle)


def _render_holdings(holdings: pd.DataFrame) -> None:
    """Render the current portfolio holdings table."""
    st.subheader("Current Portfolio Holdings")
    columns = [
        "ticker",
        "company_name",
        "category",
        "dollar_allocation",
        "weight",
        "latest_price",
        "daily_return",
        "total_score",
        "reason_selected",
    ]
    display = holdings[[column for column in columns if column in holdings]].copy()
    if "ticker" in display:
        display["ticker"] = display["ticker"].map(display_ticker)
    if "dollar_allocation" in display:
        display["dollar_allocation"] = display["dollar_allocation"].map(
            lambda value: format_currency(value, 0)
        )
    if "weight" in display:
        display["weight"] = display["weight"].map(format_percent)
    if "latest_price" in display:
        display["latest_price"] = display["latest_price"].map(format_currency)
    if "daily_return" in display:
        display["daily_return"] = display["daily_return"].map(format_percent)
    if "total_score" in display:
        display["total_score"] = display["total_score"].map(format_number)
    display = display.rename(
        columns={
            "ticker": "Ticker",
            "company_name": "Company",
            "category": "Category",
            "dollar_allocation": "Dollar allocation",
            "weight": "Weight",
            "latest_price": "Stock price",
            "daily_return": "Daily return",
            "total_score": "Total score",
            "reason_selected": "Reason selected",
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)


def _render_ranking(ranking: pd.DataFrame) -> None:
    """Render the defense universe ranking table."""
    st.subheader("Defense Universe Ranking")
    ranking_columns = [
        "ticker",
        "company_name",
        "latest_price",
        "momentum_score",
        "risk_score",
        "valuation_score",
        "stability_score",
        "total_score",
        "selected",
        "data_status",
    ]
    available_columns = [column for column in ranking_columns if column in ranking]
    display = ranking[available_columns].copy()
    if "ticker" in display:
        display["ticker"] = display["ticker"].map(display_ticker)
    score_columns = [
        "momentum_score",
        "risk_score",
        "valuation_score",
        "stability_score",
        "total_score",
    ]
    for column in score_columns:
        if column in display:
            display[column] = display[column].map(format_number)
    if "latest_price" in display:
        display["latest_price"] = display["latest_price"].map(format_currency)
    display = display.rename(
        columns={
            "ticker": "Ticker",
            "company_name": "Company",
            "latest_price": "Stock price",
            "momentum_score": "Momentum score",
            "risk_score": "Risk score",
            "valuation_score": "Valuation score",
            "stability_score": "Stability score",
            "total_score": "Total score",
            "selected": "Selected",
            "data_status": "Data status",
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)


def _render_charts(dashboard: dict[str, Any]) -> None:
    """Render dashboard charts."""
    st.subheader("NAV Performance")
    st.plotly_chart(nav_comparison_chart(dashboard["nav_history"]), width="stretch")
    st.plotly_chart(drawdown_chart(dashboard["nav_history"]), width="stretch")

    st.subheader("Portfolio and Ranking Charts")
    st.plotly_chart(score_bar_chart(dashboard["ranking"]), width="stretch")

    prices = dashboard.get("prices", pd.DataFrame())
    if isinstance(prices, pd.DataFrame) and not prices.empty:
        st.plotly_chart(
            normalized_performance_chart(prices, dashboard["holdings"]),
            width="stretch",
        )


def _render_news(
    holdings: pd.DataFrame,
    headlines_by_ticker: dict[str, list[NewsHeadline]],
) -> None:
    """Render recent headlines for selected holdings and anchors."""
    st.subheader("Recent Headlines")

    for ticker in holdings["ticker"].tolist():
        company = holdings.loc[holdings["ticker"] == ticker, "company_name"].iloc[0]
        with st.expander(f"{display_ticker(ticker)} - {company}", expanded=False):
            headlines = headlines_by_ticker.get(ticker, [])
            if not headlines:
                st.write("No recent headlines found.")
                continue
            for item in headlines:
                label = item.title
                if item.publisher:
                    label = f"{label} ({item.publisher})"
                if item.link:
                    st.markdown(f"- [{label}]({item.link})")
                else:
                    st.markdown(f"- {label}")


def _render_methodology(config: dict[str, Any]) -> None:
    """Render methodology and limitations."""
    weights = config["scoring_weights"]
    portfolio_config = config["portfolio"]
    anchors = ", ".join(
        f"{anchor['ticker']} ({float(anchor['allocation']) * 100:.0f}%)"
        for anchor in config.get("anchor_assets", [])
    )
    initial_investment = float(portfolio_config.get("initial_investment", 10_000_000.0))
    fund_start_date = str(portfolio_config.get("fund_start_date", "N/A"))

    st.subheader("Methodology")
    st.markdown(
        f"""
- **Universe source:** editable CSV file at `data/universe.csv`; companies marked as defense-adjacent remain visible in the category and notes fields.
- **Score components:** momentum uses 3-month and 6-month returns; risk uses 90-day annualized volatility and beta when available; valuation uses trailing and forward P/E; stability uses market capitalization.
- **Score weights:** momentum {weights['momentum']:.0%}, risk {weights['risk']:.0%}, valuation {weights['valuation']:.0%}, stability {weights['stability']:.0%}.
- **Portfolio rules:** top {int(portfolio_config['max_defense_holdings'])} scoreable defense/aerospace candidates receive {float(portfolio_config['defense_allocation']):.0%} of the portfolio, split by total score. Anchors are added from config: {anchors}.
- **NAV simulation:** the fund is simulated with an initial investment of {format_currency(initial_investment, 0)}. The fixed inception date is {fund_start_date}, set on 2026-05-10 as three months before setup. Future runs keep this same start date, so the simulated operating history continues to lengthen. This version applies the current selected target weights across the full period since inception; future versions may model actual historical rebalancing.
- **Limitations:** free yfinance and RSS data can be delayed, incomplete, restated, or temporarily unavailable. Scores are cross-sectional and do not incorporate position sizing constraints, transaction costs, taxes, liquidity, FX hedging, or qualitative geopolitical judgment.
"""
    )


def _render_data_warnings(
    tickers: tuple[str, ...],
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> None:
    """Show clear warnings for missing market data without stopping the app."""
    missing_prices = [
        ticker
        for ticker in tickers
        if ticker not in prices.columns or prices[ticker].dropna().empty
    ]
    if missing_prices:
        st.warning(
            "No adjusted close price history returned for: "
            + ", ".join(display_ticker(ticker) for ticker in missing_prices)
            + ". These tickers will be skipped or shown with missing fields."
        )

    if fundamentals.empty:
        st.warning("No fundamental data returned from yfinance.")
        return

    missing_fundamentals = fundamentals[
        fundamentals[["market_cap", "trailing_pe", "forward_pe", "beta"]].isna().all(axis=1)
    ]["ticker"].tolist()
    if missing_fundamentals:
        st.info(
            "Some valuation, beta, or market-cap fields are missing for: "
            + ", ".join(display_ticker(ticker) for ticker in missing_fundamentals)
            + ". Neutral scoring is used where the methodology allows it."
        )


def _save_outputs(dashboard: dict[str, Any]) -> None:
    """Persist latest dashboard outputs as local CSV files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dashboard["holdings"].to_csv(HOLDINGS_OUTPUT_PATH, index=False)
    dashboard["ranking"].to_csv(SCORES_OUTPUT_PATH, index=False)
    pd.DataFrame([dashboard["summary"]]).to_csv(NAV_SUMMARY_OUTPUT_PATH, index=False)
    dashboard["nav_history"].reset_index().to_csv(NAV_HISTORY_OUTPUT_PATH, index=False)
    _news_map_to_frame(
        dashboard["holdings"],
        dashboard["news_by_ticker"],
        news_updated_at=str(dashboard["summary"].get("news_updated_at", local_timestamp())),
    ).to_csv(
        NEWS_OUTPUT_PATH,
        index=False,
    )


def _load_saved_dashboard_data() -> dict[str, Any]:
    """Load the latest saved CSV outputs for fallback rendering."""
    required_paths = [
        HOLDINGS_OUTPUT_PATH,
        SCORES_OUTPUT_PATH,
        NAV_SUMMARY_OUTPUT_PATH,
        NAV_HISTORY_OUTPUT_PATH,
    ]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing saved output file(s): " + ", ".join(missing))

    holdings = pd.read_csv(HOLDINGS_OUTPUT_PATH)
    ranking = pd.read_csv(SCORES_OUTPUT_PATH)
    summary = pd.read_csv(NAV_SUMMARY_OUTPUT_PATH).iloc[0].to_dict()
    nav_history = pd.read_csv(NAV_HISTORY_OUTPUT_PATH)
    if "date" not in nav_history:
        raise ValueError("Saved NAV history is missing the date column.")
    nav_history["date"] = pd.to_datetime(nav_history["date"])
    nav_history = nav_history.set_index("date")

    news_by_ticker: dict[str, list[NewsHeadline]] = {}
    if NEWS_OUTPUT_PATH.exists():
        news_by_ticker = _news_frame_to_map(pd.read_csv(NEWS_OUTPUT_PATH))

    summary["dashboard_data_source"] = "saved"
    return {
        "source": "saved",
        "prices": pd.DataFrame(),
        "holdings": holdings,
        "ranking": ranking,
        "nav_history": nav_history,
        "summary": summary,
        "news_by_ticker": news_by_ticker,
    }


def _news_map_to_frame(
    holdings: pd.DataFrame,
    headlines_by_ticker: dict[str, list[NewsHeadline]],
    *,
    news_updated_at: str,
) -> pd.DataFrame:
    """Flatten ticker news into a CSV-friendly DataFrame."""
    rows: list[dict[str, Any]] = []
    for ticker in holdings["ticker"].tolist():
        headlines = headlines_by_ticker.get(ticker, [])
        if not headlines:
            rows.append(
                {
                    "ticker": ticker,
                    "title": "",
                    "publisher": "",
                    "published_at": "",
                    "link": "",
                    "source": "",
                    "no_headlines": True,
                    "news_updated_at": news_updated_at,
                }
            )
            continue
        for headline in headlines:
            row = asdict(headline)
            row["ticker"] = ticker
            row["no_headlines"] = False
            row["news_updated_at"] = news_updated_at
            rows.append(row)
    return pd.DataFrame(rows)


def _news_frame_to_map(news: pd.DataFrame) -> dict[str, list[NewsHeadline]]:
    """Rebuild ticker news mapping from the saved CSV."""
    if news.empty or "ticker" not in news:
        return {}

    result: dict[str, list[NewsHeadline]] = {}
    for ticker, group in news.groupby("ticker"):
        headlines: list[NewsHeadline] = []
        for _, row in group.iterrows():
            if _as_bool(row.get("no_headlines", False)):
                continue
            title = str(row.get("title", "")).strip()
            if not title:
                continue
            headlines.append(
                NewsHeadline(
                    title=title,
                    publisher=_optional_text(row.get("publisher")),
                    published_at=_optional_text(row.get("published_at")),
                    link=_optional_text(row.get("link")),
                    source=_optional_text(row.get("source")),
                )
            )
        result[str(ticker)] = headlines
    return result


def _optional_text(value: object) -> str | None:
    """Return a non-empty string or None for saved optional text fields."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _as_bool(value: object) -> bool:
    """Coerce saved CSV boolean-ish values."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


if __name__ == "__main__":
    main()
