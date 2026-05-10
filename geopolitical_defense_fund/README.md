# Geopolitical Defense Fund Dashboard

A Streamlit dashboard for simulating a Canadian/TSX-focused defense, aerospace, and geopolitical-risk portfolio. The project is built for education, research, and portfolio simulation only. It is not financial advice and does not place real trades.

## What the App Does

- Loads an editable stock universe from `data/universe.csv`.
- Pulls free market and fundamental data with `yfinance`.
- Scores eligible defense, aerospace, and defense-adjacent candidates using measurable inputs.
- Selects the top 5 scoreable candidates.
- Adds configured gold and oil/energy anchors: `GLD` and `XLE`.
- Simulates a fund with an initial investment of `$10,000,000`.
- Uses a fixed fund inception date of `2026-02-10`, set as 3 months before the initial setup date of `2026-05-10`.
- Tracks historical fund NAV from the start date through the latest available market data date.
- Compares fund performance against `SPY` indexed to `$10,000,000`.
- Displays holdings, rankings, NAV, drawdown, allocation charts, and recent headlines.
- Saves latest dashboard outputs as local CSV files for fallback rendering.

## Scoring Model

Each scoreable defense/aerospace candidate receives a total score from 0 to 100. The weights live in `config.yaml`.

| Component | Inputs | Preference | Default Weight |
|---|---|---:|---:|
| Momentum | 3-month return and 6-month return | Higher is better | 35% |
| Risk | 90-day annualized volatility and beta when available | Lower is better | 25% |
| Valuation | Trailing P/E and forward P/E | Lower positive values are better | 20% |
| Stability | Market capitalization | Larger is better | 20% |

Missing valuation data receives a neutral valuation score rather than excluding the stock. Negative or unusable P/E values are treated as missing. Missing market capitalization receives a neutral stability score. If beta is missing, the risk score uses volatility only.

## Portfolio Weighting

- The top 5 valid defense/aerospace candidates are selected by total score.
- The selected candidates receive 70% of the portfolio.
- The 70% sleeve is split according to each selected candidate's total score.
- `GLD` receives 15%.
- `XLE` receives 15%.
- If fewer than 5 candidates have enough price data, the dashboard selects as many as can be scored.
- If no candidates can be scored, the dashboard shows a clear error message or loads saved outputs if available.

## NAV and Performance Tracking

The simulated fund starts with `$10,000,000`. The fixed inception date is `2026-02-10`, which was set on `2026-05-10` as three months before the initial setup date. Future app runs keep this same inception date, so three months from now the fund will show roughly six months of simulated operating history rather than resetting to a rolling 3-month window.

The first version applies the current selected target weights across the full period since inception. It does not yet model historical rebalancing. Daily weighted returns are compounded to create the fund NAV path.

The dashboard reports:

- Initial NAV
- Current estimated NAV
- Dollar gain/loss since inception
- Total return since inception
- Daily portfolio return
- Annualized volatility since inception
- Maximum drawdown since inception
- SPY total return since inception
- Fund minus SPY return difference
- Whether the simulated fund is outperforming or underperforming SPY

## SPY Benchmark

`SPY` is pulled through `yfinance` and indexed to `$10,000,000` at the same start point as the simulated fund. The dashboard compares fund NAV against the SPY indexed value and reports the return spread.

## Caching and Refresh Data

The app uses Streamlit caching for local development and future free hosting:

- Market and fundamental data are cached for 1 hour.
- News data is cached for 6 hours.

Use the sidebar **Refresh Data** button to clear cached data, refetch the latest available market/news data, recalculate scores, rebuild holdings, recalculate NAV and performance metrics, and update saved CSV outputs.

## Saved Output Files

After a successful live calculation, the app saves:

```text
data/latest_holdings.csv
data/latest_scores.csv
data/latest_nav_summary.csv
data/latest_nav_history.csv
data/latest_news.csv
```

If live market data or news fetching fails, the dashboard tries to render from the most recent saved files. If no saved files exist, it shows a clear error message.

## Install

```bash
cd geopolitical_defense_fund
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Edit the Universe

The stock universe is stored in:

```text
data/universe.csv
```

Required columns:

- `ticker`
- `company_name`
- `category`
- `exchange`
- `country`
- `notes`

You can add or remove tickers without changing Python code. Keep TSX tickers in Yahoo Finance format, such as `CAE.TO` or `BBD-B.TO`.

## Configuration

Core settings live in `config.yaml`, including:

- scoring weights
- `$10,000,000` simulated initial investment
- fixed fund inception date
- max selected defense holdings
- portfolio sleeve allocations
- benchmark ticker
- anchor asset tickers and weights
- price history period
- scoring lookback windows
- free news source settings

## Known Limitations

- `yfinance` data can be delayed, incomplete, or temporarily unavailable.
- Some Canadian small-cap securities may have sparse fundamental data.
- RSS and yfinance news availability varies by ticker.
- The model uses simplified constant-weight NAV simulation.
- Current selected weights are applied across the full history since fixed inception.
- Scores are cross-sectional and depend on the current universe file.
- The dashboard does not model taxes, FX, liquidity, slippage, transaction costs, or mandate constraints.
- The first version intentionally uses measurable financial metrics only; news does not affect initial weights.

## Disclaimer

This project is for educational research and portfolio simulation only. It is not investment advice, financial advice, a recommendation, or an offer to buy or sell securities. The dashboard does not place real trades.
