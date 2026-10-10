import asyncio
import html
import logging
import re
from collections import defaultdict

import httpx

from app.ingestion.scrapping.models import Document

logger = logging.getLogger(__name__)

SEARCH_URL = "https://hn.algolia.com/api/v1/search"

QUERY_TEMPLATES = [
    "{company} culture",
    "{company} salary",
    "{company} interview",
    "{company} working at",
]

MIN_COMMENT_CHARS = 80
MIN_DOC_CHARS = 200
MAX_DOC_CHARS = 20_000

TAG_RE = re.compile(r"<[^>]+>")


def _clean(comment_html: str) -> str:
    text = comment_html.replace("<p>", "\n")
    text = TAG_RE.sub("", text)
    return html.unescape(text).strip()


async def _search(client: httpx.AsyncClient, query: str) -> list[dict]:
    response = await client.get(
        SEARCH_URL,
        params={"query": query, "tags": "comment", "hitsPerPage": 20},
    )
    response.raise_for_status()
    return response.json().get("hits", [])


async def fetch(company: str) -> list[Document]:
    async with httpx.AsyncClient(timeout=20) as client:
        batches = await asyncio.gather(
            *(_search(client, t.format(company=company)) for t in QUERY_TEMPLATES),
            return_exceptions=True,
        )

    needle = company.lower()
    seen_ids: set[str] = set()
    by_story: dict[str, dict] = defaultdict(lambda: {"title": "", "comments": []})

    for batch in batches:
        if isinstance(batch, Exception):
            logger.warning("Hacker News query failed for %s: %s", company, batch)
            continue

        for hit in batch:
            object_id = hit.get("objectID")
            story_id = hit.get("story_id")
            raw = hit.get("comment_text") or ""

            if not object_id or not story_id or object_id in seen_ids:
                continue

            text = _clean(raw)
            # Algolia matching is loose; keep only comments that name the company
            if needle not in text.lower() or len(text) < MIN_COMMENT_CHARS:
                continue

            seen_ids.add(object_id)
            story = by_story[str(story_id)]
            story["title"] = hit.get("story_title") or story["title"]
            date = (hit.get("created_at") or "")[:10]
            story["comments"].append(f"- {hit.get('author', 'unknown')} ({date}): {text}")

    documents: list[Document] = []
    for story_id, story in by_story.items():
        title = story["title"] or f"Hacker News thread {story_id}"
        body = f"Hacker News discussion: {title}\n\n" + "\n\n".join(story["comments"])

        if len(body) < MIN_DOC_CHARS:
            continue

        documents.append(
            Document(
                company=company,
                source_type="hackernews",
                url=f"https://news.ycombinator.com/item?id={story_id}",
                title=title,
                text=body[:MAX_DOC_CHARS],
            )
        )

    return documents