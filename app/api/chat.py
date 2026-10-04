from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.retrieval.search import search
from app.generation.prompts import build_prompt
from app.generation.llm import generate

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    company: str
    question: str


class ChatResponse(BaseModel):
    company: str
    question: str
    answer: str
    sources: list[str]


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:

    company = request.company.strip().lower()
    question = request.question.strip()

    if not company:
        raise HTTPException(status_code=400, detail="company must not be empty")
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    chunks = search(question=question, company=company, top_k=5)

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for company '{company}'. "
                   "Make sure it has been ingested first.",
        )

    prompt = build_prompt(question=question, chunks=chunks)

    answer = generate(prompt)

    sources = list(dict.fromkeys(c.get("source", "") for c in chunks))

    return ChatResponse(
        company=request.company,
        question=question,
        answer=answer,
        sources=sources,
    )
