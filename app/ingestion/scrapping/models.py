from datetime import datetime, timezone

from pydantic import BaseModel, Field


class Document(BaseModel):
    company: str
    source_type: str  # "search", "wikipedia", "hackernews", ...
    url: str
    title: str = ""
    text: str
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))