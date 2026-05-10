"""Plotly chart builders for the dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .utils import display_ticker


def allocation_pie_chart(holdings: pd.DataFrame) -> go.Figure:
    """Create a portfolio allocation pie chart."""
    if holdings.empty:
        return _empty_figure("No holdings available")

    display = holdings.copy()
    display["display_ticker"] = display["ticker"].map(display_ticker)
    display["weight_percent"] = display["weight"] * 100
    for column in ["latest_price", "dollar_allocation"]:
        if column not in display:
            display[column] = None
    fig = px.pie(
        display,
        names="display_ticker",
        values="weight",
        custom_data=[
            "ticker",
            "company_name",
            "category",
            "weight_percent",
            "latest_price",
            "dollar_allocation",
        ],
        title="Portfolio Allocation",
        hole=0.35,
        color_discrete_sequence=[
            "#356AC3",
            "#8BBFF0",
            "#EF463D",
            "#F3AAA8",
            "#5BB09E",
            "#91E2A3",
            "#F58D2D",
            "#B784E8",
        ],
    )
    fig.update_traces(
        textposition="inside",
        textinfo="label+percent",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "%{customdata[1]}<br>"
            "Category: %{customdata[2]}<br>"
            "Weight: %{customdata[3]:.2f}%<br>"
            "Latest price: $%{customdata[4]:,.2f}<br>"
            "Dollar allocation: $%{customdata[5]:,.0f}"
            "<extra></extra>"
        ),
    )
    fig.update_layout(
        legend_title_text="",
        margin=dict(l=20, r=20, t=60, b=20),
        clickmode="event+select",
    )
    return fig


def score_bar_chart(ranking: pd.DataFrame) -> go.Figure:
    """Create a total-score bar chart for the defense universe."""
    if ranking.empty or ranking["total_score"].dropna().empty:
        return _empty_figure("No score data available")

    display = ranking.dropna(subset=["total_score"]).copy()
    display = display.sort_values("total_score", ascending=True)
    display["display_ticker"] = display["ticker"].map(display_ticker)
    display["selected_label"] = display["selected"].map({True: "Selected", False: "Not selected"})

    fig = px.bar(
        display,
        x="total_score",
        y="display_ticker",
        orientation="h",
        color="selected_label",
        hover_data=["company_name", "category", "momentum_score", "risk_score"],
        title="Defense Universe Total Scores",
        labels={
            "total_score": "Total score",
            "display_ticker": "Ticker",
            "selected_label": "",
            "company_name": "Company",
            "category": "Category",
            "momentum_score": "Momentum score",
            "risk_score": "Risk score",
        },
    )
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), xaxis_range=[0, 100])
    return fig


def normalized_performance_chart(
    prices: pd.DataFrame,
    holdings: pd.DataFrame,
) -> go.Figure:
    """Create a normalized price performance chart for selected holdings."""
    if prices.empty or holdings.empty:
        return _empty_figure("No price data available")

    tickers = [ticker for ticker in holdings["ticker"].tolist() if ticker in prices.columns]
    if not tickers:
        return _empty_figure("No selected holding price data available")

    fig = go.Figure()
    for ticker in tickers:
        series = pd.to_numeric(prices[ticker], errors="coerce").dropna()
        if series.empty:
            continue
        normalized = series / series.iloc[0] * 100
        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized,
                mode="lines",
                name=display_ticker(ticker),
            )
        )

    if not fig.data:
        return _empty_figure("No selected holding price data available")

    fig.update_layout(
        title="Recent Normalized Price Performance",
        yaxis_title="Indexed price, first available date = 100",
        xaxis_title="Date",
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified",
    )
    return fig


def nav_comparison_chart(
    nav_history: pd.DataFrame,
    benchmark_series: pd.DataFrame | None = None,
) -> go.Figure:
    """Create a fund NAV chart with optional SPY benchmark comparison."""
    if nav_history.empty:
        return _empty_figure("No NAV data available")

    fund_column = "fund_nav" if "fund_nav" in nav_history else "nav"
    benchmark_column = "benchmark_nav" if "benchmark_nav" in nav_history else None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=nav_history.index,
            y=nav_history[fund_column],
            mode="lines",
            name="Simulated fund NAV",
        )
    )
    if benchmark_column is not None and nav_history[benchmark_column].notna().any():
        fig.add_trace(
            go.Scatter(
                x=nav_history.index,
                y=nav_history[benchmark_column],
                mode="lines",
                name="SPY indexed value",
            )
        )
    elif benchmark_series is not None and not benchmark_series.empty:
        fig.add_trace(
            go.Scatter(
                x=benchmark_series.index,
                y=benchmark_series["benchmark_nav"],
                mode="lines",
                name="SPY indexed value",
            )
        )
    fig.update_layout(
        title="Simulated Fund NAV vs SPY",
        yaxis_title="Portfolio value",
        xaxis_title="Date",
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified",
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    return fig


def drawdown_chart(nav_history: pd.DataFrame) -> go.Figure:
    """Create a fund drawdown chart from inception."""
    if nav_history.empty or "fund_drawdown" not in nav_history:
        return _empty_figure("No drawdown data available")

    drawdown = pd.to_numeric(nav_history["fund_drawdown"], errors="coerce")
    if drawdown.dropna().empty:
        return _empty_figure("No drawdown data available")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=nav_history.index,
            y=drawdown,
            mode="lines",
            name="Fund drawdown",
            fill="tozeroy",
        )
    )
    fig.update_layout(
        title="Fund Drawdown Since Inception",
        yaxis_title="Drawdown",
        xaxis_title="Date",
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified",
    )
    fig.update_yaxes(tickformat=".1%")
    return fig


def _empty_figure(message: str) -> go.Figure:
    """Return a Plotly figure with a centered no-data message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig
