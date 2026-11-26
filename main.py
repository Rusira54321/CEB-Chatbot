"""
Complete Pipeline Runner
------------------------
1. Loads 13 hydro power plant chunks from chunks.py
2. Generates semantic embeddings (Title + Content) with BAAI/bge-m3.
3. Persists all chunks with metadata into persistent ChromaDB.
4. Demonstrates semantic search query against stored chunks.
"""

import re

from chunks import all_chunks
from Embedding import EmbeddingService
from vector_store import ChromaVectorStore


def main() -> None:
    source_url = "https://www.ceb.lk/demand-side-management/en"

    # ---------------------------------------------------------
    # STEP 1: Prepare Documents and Metadata from Chunks
    # ---------------------------------------------------------
    
    print("=" * 65)
    print(f"STEP 1: Preparing {len(all_chunks)} Chunks for Ingestion")
    print("=" * 65)

    # Combine Title + Content together for stronger semantic embeddings.
    # .strip()          → removes leading/trailing whitespace from content
    # re.sub(r'[ \t]+') → collapses multiple internal spaces/tabs into one space
    # Result: clean text with no noise spaces sent to the embedding model
    documents = [
        f"{c['title']}\n\n{re.sub(r'[ \t]+', ' ', c['content'].strip())}"
        for c in all_chunks
    ]

    # Attach metadata with source URL, title, category, and language
    metadatas = [
        {
            "source": source_url,
            "title": c["title"],
            "category": "demand_side_management",
            "language": "en"
        }
        for c in all_chunks
    ]

    # Clean, readable unique IDs for each chunk (e.g. dsm_chunk_1 ... dsm_chunk_10)
    doc_ids = [f"dsm_chunk_{i + 1}" for i in range(len(all_chunks))]

    print(f"Prepared {len(documents)} documents with metadata.")

    # ---------------------------------------------------------
    # STEP 2: Generate Vector Embeddings (Batch of 13 chunks)
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP 2: Generating Embeddings with BAAI/bge-m3...")
    print("=" * 65)

    embedding_service = EmbeddingService()
    embeddings = embedding_service.encode(text=documents)
    print(f"Generated Embeddings Shape: {embeddings.shape}")
    # Expected shape: (13, 1024) → 13 chunks, each with 1024-dim vector

    # ---------------------------------------------------------
    # STEP 3: Store in Persistent ChromaDB
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP 3: Storing Chunks in Persistent ChromaDB...")
    print("=" * 65)

    vector_store = ChromaVectorStore(
        persist_directory="./chroma_db",
        collection_name="ceb_knowledge_base"
    )

    vector_store.add_documents(
        doc_ids=doc_ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Successfully stored all {len(all_chunks)} chunks!")
    print(f"Total documents in ChromaDB collection: {vector_store.count()}")

    # ---------------------------------------------------------
    # STEP 4: Test Semantic Search against Stored Chunks
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP 4: Testing Semantic Search against Stored Chunks")
    print("=" * 65)

    test_query = "What is demand side management"
    print(f"User Query: '{test_query}'")

    # Embed the query using the same model for a fair comparison
    embedding_service = EmbeddingService()
    query_vector = embedding_service.encode(text=test_query)
    vector_store = ChromaVectorStore(
            persist_directory="./chroma_db",
            collection_name="ceb_knowledge_base"
        )
    results = vector_store.query(query_embedding=query_vector, n_results=1)

    print("\n--- Best Matching Chunk Found ---")
    matched_title = results["metadatas"][0][0]["title"]
    matched_text = results["documents"][0][0]
    matched_source = results["metadatas"][0][0]["source"]

    print(f"Title  : {matched_title}")
    print(f"Source : {matched_source}")
    print(f"\nContent:\n{matched_text}")

    print("\n" + "=" * 65)
    print("SUCCESS: Pipeline completed! Data persisted in ./chroma_db")
    print("=" * 65)


if __name__ == "__main__":
    main()
