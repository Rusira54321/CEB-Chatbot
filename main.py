"""
Complete Pipeline Runner
------------------------
1. Loads 13 hydro power plant chunks from chunks.py
2. Generates semantic embeddings (Title + Content) with BAAI/bge-m3.
3. Persists all chunks with metadata into persistent ChromaDB.
4. Demonstrates semantic search query against stored chunks.
"""

from chunks import all_chunks
from Embedding import EmbeddingService
from vector_store import ChromaVectorStore


def main() -> None:
    source_url = "https://www.ceb.lk/plant-operation/en"

    # ---------------------------------------------------------
    # STEP 1: Prepare Documents and Metadata from Chunks
    # ---------------------------------------------------------
    print("=" * 65)
    print(f"STEP 1: Preparing {len(all_chunks)} Chunks for Ingestion")
    print("=" * 65)

    # Combine Title + Content together for stronger semantic embeddings.
    # Example output: "Kotmale Complex\n\nKotmale reservoir is located..."
    documents = [f"{c['title']}\n\n{c['content']}" for c in all_chunks]

    # Attach metadata with source URL, title, category, and language
    metadatas = [
        {
            "source": source_url,
            "title": c["title"],
            "category": "hydro_power_plants",
            "language": "en"
        }
        for c in all_chunks
    ]

    # Clean, readable unique IDs for each chunk (e.g. hydro_chunk_1 ... hydro_chunk_13)
    doc_ids = [f"hydro_chunk_{i + 1}" for i in range(len(all_chunks))]

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

    test_query = "What is the capacity of Kotmale power station?"
    print(f"User Query: '{test_query}'")

    # Embed the query using the same model for a fair comparison
    query_vector = embedding_service.encode(text=test_query)
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
