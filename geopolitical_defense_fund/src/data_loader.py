"""Universe CSV loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import clean_ticker, get_logger


LOGGER = get_logger("data_loader")

REQUIRED_UNIVERSE_COLUMNS = [
    "ticker",
    "company_name",
    "category",
    "exchange",
    "country",
    "notes",
]


class UniverseError(ValueError):
    """Raised when the editable security universe is invalid."""


def load_universe(path: str | Path) -> pd.DataFrame:
    """Load and validate the editable security universe CSV."""
    universe_path = Path(path)
    if not universe_path.exists():
        raise UniverseError(
            f"Universe file not found: {universe_path}. Create data/universe.csv first."
        )

    universe = pd.read_csv(universe_path)
    validate_universe(universe)

    cleaned = universe.copy()
    for column in REQUIRED_UNIVERSE_COLUMNS:
        cleaned[column] = cleaned[column].fillna("").astype(str).str.strip()

    cleaned["ticker"] = cleaned["ticker"].map(clean_ticker)
    before = len(cleaned)
    cleaned = cleaned[cleaned["ticker"] != ""].drop_duplicates("ticker", keep="first")
    if len(cleaned) < before:
        LOGGER.warning("Dropped %s empty or duplicate ticker rows.", before - len(cleaned))

    return cleaned.reset_index(drop=True)


def validate_universe(universe: pd.DataFrame) -> None:
    """Validate required columns in the universe DataFrame."""
    missing = [column for column in REQUIRED_UNIVERSE_COLUMNS if column not in universe]
    if missing:
        raise UniverseError(f"Universe CSV missing required column(s): {', '.join(missing)}")

    if universe.empty:
        raise UniverseError("Universe CSV is empty.")
