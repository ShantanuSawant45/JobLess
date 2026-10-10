import logging
from urllib.parse import quote

import httpx

from app.ingestion.models import Document

logger = logging.getLogger(__name__)

API_URL = "https://en.wikipedia.org/w/api.php"

# Wikimedia requires a descriptive User-Agent with contact info
HEADERS = {"User-Agent": "CompanyRAG/0.1 (student project; your-email@example.com)"}

TRAILING_SECTIONS = (
    "== References ==",
    "== External links ==",
    "== See also ==",
    "== Notes ==",
    "== Further reading ==",
)

MIN_TEXT_CHARS = 200
MAX_TEXT_CHARS = 20_000


async def _find_title(client: httpx.AsyncClient, company: str) -> str | None:
    response = await client.get(
        API_URL,
        params={
            "action": "query",
            "list": "search",
            "srsearch": f"{company} company",
            "srlimit": 5,
            "format": "json",
            "formatversion": 2,
        },
    )
    response.raise_for_status()

    needle = company.lower()
    for result in response.json()["query"]["search"]:
        # Only accept pages whose title contains the company name
        if needle in result["title"].lower():
            return result["title"]
    return None


async def _get_text(client: httpx.AsyncClient, title: str) -> str:
    response = await client.get(
        API_URL,
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": title,
            "redirects": 1,
            "format": "json",
            "formatversion": 2,
        },
    )
    response.raise_for_status()

    pages = response.json()["query"]["pages"]
    return (pages[0].get("extract") or "") if pages else ""


def _trim(text: str) -> str:
    cut = len(text)
    for heading in TRAILING_SECTIONS:
        index = text.find(heading)
        if index != -1:
            cut = min(cut, index)
    return text[:cut].strip()


async def fetch(company: str) -> list[Document]:
    try:
        async with httpx.AsyncClient(timeout=20, headers=HEADERS) as client:
            title = await _find_title(client, company)
            if not title:
                return []
            text = _trim(await _get_text(client, title))
    except Exception as exc:
        logger.warning("Wikipedia lookup failed for %s: %s", company, exc)
        return []

    # Skip stubs and disambiguation pages
    if len(text) < MIN_TEXT_CHARS or "may refer to" in text[:300]:
        return []

    return [
        Document(
            company=company,
            source_type="wikipedia",
            url=f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
            title=title,
            text=text[:MAX_TEXT_CHARS],
        )
    ]