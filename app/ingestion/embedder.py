from sentence_transformers import SentenceTransformer

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    model = _get_model()

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()



if __name__ == "__main__":
    samples = [
        "Razorpay is an Indian payment gateway.",
        "Atlassian makes Jira and Confluence.",
    ]
    result = embed(samples)
    print(f"Embedded {len(result)} texts")
    print(f"Vector size: {len(result[0])} dimensions")  # should be 384
    print(f"First 5 values: {result[0][:5]}")
