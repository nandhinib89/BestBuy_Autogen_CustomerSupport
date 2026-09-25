from rag.retriever import search
from rag.reranker import rerank


query = "What is Best Buy Canada's return policy?"

print("\n--- FAISS RESULTS ---\n")

documents = search(
    query,
    k=8,
    domain="support",
)

for i, document in enumerate(documents, start=1):
    print(f"RESULT {i}")
    print(f"Source: {document.metadata.get('filename')}")
    print(document.page_content[:500])
    print("-" * 80)


print("\n--- RERANKED RESULTS ---\n")

reranked_documents = rerank(
    query,
    documents,
    top_k=3,
)

for i, document in enumerate(reranked_documents, start=1):
    print(f"RESULT {i}")
    print(f"Source: {document.metadata.get('filename')}")
    print(document.page_content[:500])
    print("-" * 80)