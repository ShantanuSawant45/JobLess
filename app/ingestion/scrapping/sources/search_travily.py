import asyncio
import logging

import httpx

from app.config import get_settings
from app.ingestion.models import Document

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"

QUERY_TEMPLATES = [
    "{company} company overview what they do",
    "{company} work culture employee experience",
    "{company} salary compensation",
    "{company} interview experience",
    "{company} funding revenue market value",
]

# Sites that forbid scraping; keep them out of results entirely
EXCLUDE_DOMAINS = ["glassdoor.com", "linkedin.com", "ambitionbox.com"]

MIN_TEXT_CHARS = 200
MAX_TEXT_CHARS = 20_000


async def _search(client: httpx.AsyncClient, query: str, api_key: str) -> list[dict]:
    response = await client.post(
        TAVILY_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "query": query,
            "search_depth": "basic",
            "max_results": 5,
            "include_raw_content": True,
            "exclude_domains": EXCLUDE_DOMAINS,
        },
    )
    response.raise_for_status()
    return response.json().get("results", [])


async def fetch(company: str) -> list[Document]:
    api_key = get_settings().tavily_api_key.get_secret_value()

    async with httpx.AsyncClient(timeout=30) as client:
        batches = await asyncio.gather(
            *(
                _search(client, template.format(company=company), api_key)
                for template in QUERY_TEMPLATES
            ),
            return_exceptions=True,
        )

    documents: list[Document] = []
    seen_urls: set[str] = set()

    for batch in batches:
        if isinstance(batch, Exception):
            logger.warning("Tavily query failed for %s: %s", company, batch)
            continue

        for result in batch:
            url = result.get("url")
            text = (result.get("raw_content") or result.get("content") or "").strip()

            if not url or url in seen_urls or len(text) < MIN_TEXT_CHARS:
                continue

            seen_urls.add(url)
            documents.append(
                Document(
                    company=company,
                    source_type="search",
                    url=url,
                    title=result.get("title") or "",
                    text=text[:MAX_TEXT_CHARS],
                )
            )

    return documents