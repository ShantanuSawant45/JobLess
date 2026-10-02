from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[str]:
    text = text.strip()

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        strip_whitespace=True,
    )
    return splitter.split_text(text)


if __name__ == "__main__":
    sample = "word " * 300  # ~1500 chars
    chunks = chunk_text(sample)
    print(f"{len(chunks)} chunks")
    for i, c in enumerate(chunks):
        print(f"  chunk {i}: {len(c)} chars")
