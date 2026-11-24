"""
Complete Pipeline Runner
------------------------
1. Scrapes webpage body content cleanly (removes header/footer/hidden tags).
2. Generates semantic embeddings with BAAI/bge-m3.
3. Persists documents, embeddings, and metadata into a local ChromaDB database.
4. Demonstrates a semantic search query against the persistent vector database.
"""

import uuid
from Embedding import EmbeddingService
from scraper import scrape_webpage_body
from vector_store import ChromaVectorStore


def main() -> None:
    # ---------------------------------------------------------
    # 1. Target URL to Scrape
    # ---------------------------------------------------------
    url = "https://www.ceb.lk/ceb-history/en"

    print("=" * 65)
    print(f"STEP 1: Scraping Body Content from: {url}")
    print("=" * 65)

    try:
        # Scrape clean text without headers, footers, or hidden debug elements
        text_content = scrape_webpage_body(url, as_text=True)
        print(f"Extracted clean text length: {len(text_content)} characters\n")

        # Save to local text file for reference
        output_txt = "scraped_content.txt"
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(text_content)
        print(f"Saved text to: {output_txt}\n")

        # ---------------------------------------------------------
        # 2. Generate Vector Embeddings (BAAI/bge-m3)
        # ---------------------------------------------------------
        print("=" * 65)
        print("STEP 2: Generating Vector Embeddings with BAAI/bge-m3...")
        print("=" * 65)
        embedding_service = EmbeddingService()
        embeddings = embedding_service.encode(text=text_content)
        print(f"Generated Embedding shape: {embeddings.shape} (1024 dimensions)\n")
        
        # 3. Store in Persistent ChromaDB
        # ---------------------------------------------------------
        print("=" * 65)
        print("STEP 3: Storing in Persistent ChromaDB...")
        print("=" * 65)
        
        # Initialize persistent Chroma store (data saved to ./chroma_db)
        vector_store = ChromaVectorStore(
            persist_directory="./chroma_db",
            collection_name="ceb_knowledge_base"
        )
        
        # Generate a random, unique document ID (UUIDv4)
        doc_id = str(uuid.uuid4())
        print(f"Generated Random Document ID: {doc_id}")
        metadata = {
            "source": url,
            "section": "History Of Electricity in Sri Lanka",
            "language":"en"
        }

        vector_store.add_document(
            doc_id=doc_id,
            document=text_content,
            embedding=embeddings,
            metadata=metadata
        )
        
        print(f"Document successfully stored and persisted!")
        print(f"Total documents in ChromaDB: {vector_store.count()}\n")

        """
        # ---------------------------------------------------------
        # 4. Verify: Semantic Search Query against ChromaDB
        # ---------------------------------------------------------
        print("=" * 65)
        print("STEP 4: Testing Semantic Search against Persistent ChromaDB")
        print("=" * 65)

        test_query = "When was the Ceylon Electricity Board established by parliament?"
        print(f"User Query: '{test_query}'")
        
        # Embed the query with the same model
        query_vector = embedding_service.encode(text=test_query)

        # Query ChromaDB
        results = vector_store.query(query_embedding=query_vector, n_results=1)
        
        print("\n--- Search Results from ChromaDB ---")
        if results and results.get("documents") and results["documents"][0]:
            retrieved_doc = results["documents"][0][0]
            retrieved_meta = results["metadatas"][0][0]
            distance = results["distances"][0][0]

            print(f"Matched ID: {results['ids'][0][0]}")
            print(f"Distance (Cosine): {distance:.4f} (Lower distance = stronger match)")
            print(f"Metadata: {retrieved_meta}")
            print(f"Matched Text (first 250 chars):\n{retrieved_doc[:250]}...")
        else:
            print("No matching documents found.")
        
        print("\n" + "=" * 65)
        print("SUCCESS: End-to-end pipeline completed!")
        print("Data is safely persisted in the './chroma_db' folder.")
        print("=" * 65)
        """
    except Exception as error:
        print(f"An error occurred in the pipeline: {error}")


if __name__ == "__main__":
    main()
