from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.retrieval.search import search
from app.generation.prompts import build_prompt
from app.generation.llm import generate

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    company: str
    question: str
    source_type: str | None = None
    history: list[ChatMessage] = Field(default_factory=list)


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

    chunks = search(
        question=question, 
        company=company, 
        top_k=5, 
        source_type=request.source_type
    )

    if not chunks or not any(c.get("text", "").strip() for c in chunks):
        return ChatResponse(
            company=request.company,
            question=question,
            answer="I don't have enough evidence in my sources to answer this question. Please ensure the company data has been ingested or try rephrasing your question.",
            sources=[],
        )

    # Pass last 10 messages (up to 5 conversational turns) to keep context manageable
    history_turns = [msg.model_dump() for msg in request.history[-10:]]
    prompt = build_prompt(question=question, chunks=chunks, history=history_turns)

    answer = generate(prompt)

    # Extract distinct non-empty source URLs or citations from retrieved chunks
    sources = [
        s for s in dict.fromkeys(
            c.get("url") or c.get("source") for c in chunks
        ) if s
    ]

    return ChatResponse(
        company=request.company,
        question=question,
        answer=answer,
        sources=sources,
    )
