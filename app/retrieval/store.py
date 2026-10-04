import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from app.ingestion.embedder import embed

from app.config import get_settings

COLLECTION_NAME = "companies"
VECTOR_SIZE = 384


def _client() -> QdrantClient:
    """Return a Qdrant client pointing at the configured URL."""
    return QdrantClient(url=get_settings().qdrant_url)


def create_collection() -> None:
    """Create the 'companies' collection if it doesn't already exist."""
    client = _client()

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists, skipping.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )
    print(f"Created collection '{COLLECTION_NAME}' ({VECTOR_SIZE}d, cosine).")


def upsert_points(
    vectors: list[list[float]],
    company: str,
    source: str,
    texts: list[str],
) -> None:
    """
    Upsert vectors into Qdrant with metadata payload.

    Each point gets:
        - company:     e.g. "razorpay"
        - source:      file name or URL the chunk came from
        - chunk_index: position within that source
        - text:        the original chunk text (for display in answers)
    """
    client = _client()

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "company": company,
                "source": source,
                "chunk_index": i,
                "text": text,
            },
        )
        for i, (vec, text) in enumerate(zip(vectors, texts))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)


if __name__ == "__main__":
    create_collection()
    print("Done.")


def get_documents(question: str) -> list[str]:
    client = _client()
    
    embedded_vector = embed([question])[0]
    
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedded_vector,
        limit=5,
    )
    return [point.payload["text"] for point in search_result]