SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions about companies for students and job seekers.
You answer questions about company culture, work environment, salary ranges, interview process, \
and company background.

You MUST follow these rules:
1. Answer ONLY using the information provided in the context chunks below.
2. If the context does not contain enough information to answer the question, say:
   "I don't have enough information about that in my sources."
3. Do not make up facts, salaries, or statistics that are not in the context.
4. Keep your answer concise and factual.
5. At the end of your answer, list the sources you used (the "source" field from each chunk).
"""


def build_prompt(question: str, chunks: list[dict]) -> str:
  
    if not chunks:
        context = "No relevant context found."
    else:
        context_blocks = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.get("source", "unknown")
            text = chunk.get("text", "")
            context_blocks.append(f"[{i}] (source: {source})\n{text}")
        context = "\n\n".join(context_blocks)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- Context ---\n{context}\n\n"
        f"--- Question ---\n{question}\n\n"
        f"--- Answer ---"
    )
    return prompt
