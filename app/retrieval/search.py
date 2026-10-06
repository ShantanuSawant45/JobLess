from qdrant_client.models import Filter, FieldCondition, MatchValue, Query

from app.ingestion.embedder import embed
from app.retrieval.store import COLLECTION_NAME, _client


def search(question: str, company: str, top_k: int = 5) -> list[dict]:

    client = _client()
    query_vector = embed([question])[0]

    company_filter = Filter(
        must=[
            FieldCondition(
                key="company",
                match=MatchValue(value=company),
            )
        ]
    )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=company_filter,
        limit=top_k,
        with_payload=True,
    )

    return [hit.payload for hit in response.points]
