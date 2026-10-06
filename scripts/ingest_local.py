

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import embed
from app.retrieval.store import create_collection, upsert_points

RAW_DIR = Path("data/raw")


def main() -> None:
    # Step 1: ensure the Qdrant collection exists
    create_collection()

    stats: dict[str, int] = defaultdict(int)

    # Step 2: walk data/raw/<company>/<source>.txt
    for company_dir in sorted(RAW_DIR.iterdir()):
        if not company_dir.is_dir():
            continue

        company = company_dir.name  # e.g. "razorpay"

        for file_path in sorted(company_dir.glob("*.txt")):
            text = file_path.read_text(encoding="utf-8")

            # Skip empty files
            if not text.strip():
                print(f"  Skipping empty file: {file_path}")
                continue

            # Step 3: chunk
            chunks = chunk_text(text)
            if not chunks:
                continue

            # Step 4: embed all chunks in one batch
            vectors = embed(chunks)

            # Step 5: upsert into Qdrant
            source = file_path.name  # e.g. "about.txt"
            upsert_points(vectors=vectors, company=company, source=source, texts=chunks)

            stats[company] += len(chunks)
            print(f"  {company}/{source}: {len(chunks)} chunks")

    # Summary
    print("\n--- Summary ---")
    total = 0
    for company, count in sorted(stats.items()):
        print(f"  {company}: {count} chunks")
        total += count
    print(f"  Total: {total} chunks")


if __name__ == "__main__":
    main()
