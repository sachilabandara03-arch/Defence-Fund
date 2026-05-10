"""Configuration loading and validation."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when the YAML configuration is missing required settings."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the YAML configuration file and validate required sections."""
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError(
            "PyYAML is required. Install dependencies with pip install -r requirements.txt."
        ) from exc

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate the high-level dashboard configuration."""
    required_sections = [
        "scoring_weights",
        "portfolio",
        "anchor_assets",
        "benchmark",
        "price_history",
    ]
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ConfigError(f"Missing config section(s): {', '.join(missing)}")

    weights = config["scoring_weights"]
    required_weights = ["momentum", "risk", "valuation", "stability"]
    missing_weights = [name for name in required_weights if name not in weights]
    if missing_weights:
        raise ConfigError(f"Missing scoring weight(s): {', '.join(missing_weights)}")

    total_weight = sum(float(weights[name]) for name in required_weights)
    if total_weight <= 0:
        raise ConfigError("Scoring weights must have a positive total.")

    anchors = config.get("anchor_assets") or []
    if not isinstance(anchors, list) or not anchors:
        raise ConfigError("At least one anchor asset must be configured.")

    for anchor in anchors:
        for key in ["ticker", "company_name", "category", "allocation"]:
            if key not in anchor:
                raise ConfigError(f"Anchor asset missing required key: {key}")

    portfolio = config.get("portfolio", {})
    if float(portfolio.get("initial_investment", 0)) <= 0:
        raise ConfigError("portfolio.initial_investment must be positive.")
    if "fund_start_date" not in portfolio:
        raise ConfigError("portfolio.fund_start_date is required.")
    try:
        date.fromisoformat(str(portfolio["fund_start_date"]))
    except ValueError as exc:
        raise ConfigError("portfolio.fund_start_date must use YYYY-MM-DD format.") from exc


def get_scoring_weights(config: dict[str, Any]) -> dict[str, float]:
    """Return normalized scoring weights from the configuration."""
    raw_weights = config["scoring_weights"]
    weights = {key: float(value) for key, value in raw_weights.items()}
    total = sum(weights.values())
    if total <= 0:
        raise ConfigError("Scoring weights must sum to a positive number.")
    return {key: value / total for key, value in weights.items()}
