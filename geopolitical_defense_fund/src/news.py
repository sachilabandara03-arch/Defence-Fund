"""Free, replaceable news collection layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import requests
import yfinance as yf

from .utils import clean_ticker, dedupe_preserve_order, get_logger


LOGGER = get_logger("news")


@dataclass(frozen=True)
class NewsHeadline:
    """A normalized news headline item."""

    title: str
    publisher: str | None = None
    published_at: str | None = None
    link: str | None = None
    source: str | None = None


def fetch_recent_headlines(
    tickers: list[str] | tuple[str, ...],
    config: dict[str, Any],
) -> dict[str, list[NewsHeadline]]:
    """Fetch recent headlines per ticker from yfinance and configured RSS feeds."""
    news_config = config.get("news", {})
    max_items = int(news_config.get("max_headlines_per_ticker", 5))
    timeout = int(news_config.get("request_timeout_seconds", 8))
    rss_feeds = news_config.get("rss_feeds", [])

    results: dict[str, list[NewsHeadline]] = {}
    for ticker in dedupe_preserve_order(clean_ticker(ticker) for ticker in tickers):
        headlines: list[NewsHeadline] = []
        headlines.extend(_fetch_yfinance_news(ticker, max_items=max_items))
        if len(headlines) < max_items:
            headlines.extend(
                _fetch_rss_news(
                    ticker,
                    rss_feeds=rss_feeds,
                    timeout=timeout,
                    remaining=max_items - len(headlines),
                )
            )
        results[ticker] = _dedupe_headlines(headlines)[:max_items]

    return results


def _fetch_yfinance_news(ticker: str, *, max_items: int) -> list[NewsHeadline]:
    """Fetch yfinance headlines where available."""
    try:
        raw_items = yf.Ticker(ticker).news or []
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("yfinance news unavailable for %s: %s", ticker, exc)
        return []

    headlines: list[NewsHeadline] = []
    for item in raw_items[:max_items]:
        parsed = _parse_yfinance_item(item)
        if parsed is not None:
            headlines.append(parsed)
    return headlines


def _fetch_rss_news(
    ticker: str,
    *,
    rss_feeds: list[str],
    timeout: int,
    remaining: int,
) -> list[NewsHeadline]:
    """Fetch RSS headlines for a ticker from configured feed templates."""
    if remaining <= 0:
        return []

    headers = {"User-Agent": "geopolitical-defense-fund-dashboard/1.0"}
    headlines: list[NewsHeadline] = []
    for template in rss_feeds:
        if len(headlines) >= remaining:
            break
        try:
            url = template.format(ticker=ticker)
            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()
            feed = feedparser.parse(response.text)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("RSS news unavailable for %s: %s", ticker, exc)
            continue

        for entry in feed.entries:
            title = str(getattr(entry, "title", "")).strip()
            if not title:
                continue
            headlines.append(
                NewsHeadline(
                    title=title,
                    publisher=getattr(entry, "source", {}).get("title")
                    if isinstance(getattr(entry, "source", None), dict)
                    else getattr(feed.feed, "title", None),
                    published_at=_parse_rss_datetime(getattr(entry, "published", None)),
                    link=getattr(entry, "link", None),
                    source="rss",
                )
            )
            if len(headlines) >= remaining:
                break

    return headlines


def _parse_yfinance_item(item: dict[str, Any]) -> NewsHeadline | None:
    """Normalize both legacy and current yfinance news item shapes."""
    if not isinstance(item, dict):
        return None

    content = item.get("content") if isinstance(item, dict) else None
    if isinstance(content, dict):
        title = str(content.get("title", "")).strip()
        provider = content.get("provider") or {}
        canonical_url = content.get("canonicalUrl") or {}
        published_at = content.get("pubDate") or content.get("displayTime")
        link = canonical_url.get("url") if isinstance(canonical_url, dict) else None
        publisher = provider.get("displayName") if isinstance(provider, dict) else None
        source = "yfinance"
    else:
        title = str(item.get("title", "")).strip()
        publisher = item.get("publisher")
        link = item.get("link")
        published_at = _parse_epoch(item.get("providerPublishTime"))
        source = "yfinance"

    if not title:
        return None
    return NewsHeadline(
        title=title,
        publisher=publisher,
        published_at=published_at,
        link=link,
        source=source,
    )


def _parse_epoch(value: object) -> str | None:
    """Parse a Unix timestamp to ISO text."""
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _parse_rss_datetime(value: object) -> str | None:
    """Parse an RSS date string to ISO text when possible."""
    if not value:
        return None
    try:
        return parsedate_to_datetime(str(value)).isoformat()
    except (TypeError, ValueError, IndexError, AttributeError):
        return str(value)


def _dedupe_headlines(headlines: list[NewsHeadline]) -> list[NewsHeadline]:
    """Remove duplicate headlines while preserving order."""
    seen: set[str] = set()
    result: list[NewsHeadline] = []
    for headline in headlines:
        key = headline.title.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(headline)
    return result
