import re
from typing import Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.ingestion.embedder import embed
from app.retrieval.store import COLLECTION_NAME, _client
from app.generation.llm import generate

# Load reranker model globally
# Using BAAI/bge-reranker-base to save memory and loading time while retaining high accuracy.
reranker = CrossEncoder("BAAI/bge-reranker-base")

def rewrite_query(question: str, company: str) -> str:
    prompt = f"""You are a search query rewriting expert.
Given the target company and the user's question, rewrite the question into a highly effective search query for a vector database.
Include the company name and key terms. Return ONLY the rewritten query text, nothing else.

Company: {company}
Question: {question}
"""
    try:
        rewritten = generate(prompt, max_retries=8).strip()
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1]
        return rewritten
    except Exception as e:
        print(f"Query rewrite failed: {e}")
        return f"{company} {question}"


def rrf_merge(dense_results: list[dict], bm25_results: list[dict], k: int = 60) -> list[dict]:
    scores = {}
    payloads = {}
    
    for rank, doc in enumerate(dense_results):
        doc_id = doc.get("text")
        if not doc_id:
            continue
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        payloads[doc_id] = doc
        
    for rank, doc in enumerate(bm25_results):
        doc_id = doc.get("text")
        if not doc_id:
            continue
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        payloads[doc_id] = doc
        
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [payloads[doc_id] for doc_id, score in sorted_docs]


def tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def search(question: str, company: str, top_k: int = 5, source_type: Optional[str] = None) -> list[dict]:
    client = _client()
    
    # 1. Query Rewriting
    rewritten_query = rewrite_query(question, company)
    print(f"Original Query: {question} | Rewritten: {rewritten_query}")

    # Metadata filtering
    must_conditions = [FieldCondition(key="company", match=MatchValue(value=company))]
    if source_type:
        must_conditions.append(FieldCondition(key="source", match=MatchValue(value=source_type)))
    
    company_filter = Filter(must=must_conditions)
    
    # 2. Dense Vector Search
    query_vector = embed([rewritten_query])[0]
    dense_response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=company_filter,
        limit=20,
        with_payload=True, 
    )
    dense_results = [hit.payload for hit in dense_response.points]
    
    # 3. BM25 Search
    records, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=company_filter,
        limit=10000,
        with_payload=True,
        with_vectors=False
    )
    all_docs = [r.payload for r in records]
    
    if all_docs:
        corpus = [doc.get("text", "") for doc in all_docs]
        tokenized_corpus = [tokenize(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        
        tokenized_query = tokenize(rewritten_query)
        bm25_scores = bm25.get_scores(tokenized_query)
        
        scored_bm25 = list(zip(all_docs, bm25_scores))
        scored_bm25.sort(key=lambda x: x[1], reverse=True)
        bm25_results = [doc for doc, score in scored_bm25[:20]]
    else:
        bm25_results = []
        
    # 4. RRF Merge
    merged_results = rrf_merge(dense_results, bm25_results)
    
    if not merged_results:
        return []
        
    # 5. Cross-Encoder Reranking
    pairs = [[rewritten_query, doc.get("text", "")] for doc in merged_results]
    rerank_scores = reranker.predict(pairs)
    
    scored_merged = list(zip(merged_results, rerank_scores))
    scored_merged.sort(key=lambda x: x[1], reverse=True)
    
    final_top_k = [doc for doc, score in scored_merged[:top_k]]
    
    return final_top_k
