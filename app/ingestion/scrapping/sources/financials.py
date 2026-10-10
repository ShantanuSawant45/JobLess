import asyncio
import logging
from datetime import datetime, timezone

import yfinance as yf

from app.ingestion.scrapping.models import Document

logger = logging.getLogger(__name__)


def _fmt_money(value: float | int | None, currency: str | None) -> str | None:
    if not value:
        return None
    magnitude = abs(value)
    if magnitude >= 1e12:
        amount = f"{value / 1e12:.2f} trillion"
    elif magnitude >= 1e9:
        amount = f"{value / 1e9:.2f} billion"
    elif magnitude >= 1e6:
        amount = f"{value / 1e6:.2f} million"
    else:
        amount = f"{value:,.0f}"
    return f"{amount} {currency or ''}".strip()


def _find_symbol(company: str) -> str | None:
    quotes = yf.Search(company, max_results=8).quotes
    needle = company.lower()
    for quote in quotes:
        if quote.get("quoteType") != "EQUITY":
            continue
        names = f"{quote.get('shortname', '')} {quote.get('longname', '')}".lower()
        if needle in names:
            return quote["symbol"]
    return None


def _fetch_sync(company: str) -> Document | None:
    symbol = _find_symbol(company)
    if not symbol:
        return None  # likely a private company

    info = yf.Ticker(symbol).info
    currency = info.get("financialCurrency") or info.get("currency")
    name = info.get("longName") or info.get("shortName") or company

    market_cap = _fmt_money(info.get("marketCap"), info.get("currency"))
    revenue = _fmt_money(info.get("totalRevenue"), currency)
    summary = info.get("longBusinessSummary")

    if not market_cap and not summary:
        return None

    lines = [f"{name} is a publicly listed company (ticker {symbol}, exchange {info.get('exchange', 'unknown')})."]

    if info.get("sector") or info.get("industry"):
        lines.append(f"Sector: {info.get('sector', 'n/a')}. Industry: {info.get('industry', 'n/a')}.")

    location = ", ".join(p for p in (info.get("city"), info.get("country")) if p)
    if location:
        lines.append(f"Headquarters: {location}.")

    if info.get("fullTimeEmployees"):
        lines.append(f"Full-time employees: {info['fullTimeEmployees']:,}.")

    if market_cap:
        lines.append(f"Market capitalization: {market_cap}.")

    if revenue:
        lines.append(f"Total revenue (trailing twelve months): {revenue}.")

    if summary:
        lines.append(f"Business summary: {summary}")

    lines.append(f"Data from Yahoo Finance, retrieved {datetime.now(timezone.utc):%Y-%m-%d}.")

    return Document(
        company=company,
        source_type="financials",
        url=f"https://finance.yahoo.com/quote/{symbol}",
        title=f"{name} financial overview",
        text="\n".join(lines),
    )


async def fetch(company: str) -> list[Document]:
    try:
        document = await asyncio.to_thread(_fetch_sync, company)
    except Exception as exc:
        logger.warning("Financials lookup failed for %s: %s", company, exc)
        return []
    return [document] if document else []