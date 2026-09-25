from rag.retriever import search


query = "What is Best Buy Canada's return policy?"


print("\n==============================")
print("FAISS ONLY")
print("==============================")

faiss_results = search(
    query,
    k=4,
    domain="support",
    rerank_results=False,
)

for i, document in enumerate(faiss_results, start=1):
    print(
        f"{i}. "
        f"{document.metadata.get('filename')}"
    )


print("\n==============================")
print("FAISS + RERANKING")
print("==============================")

reranked_results = search(
    query,
    k=4,
    domain="support",
    rerank_results=True,
)

for i, document in enumerate(reranked_results, start=1):
    print(
        f"{i}. "
        f"{document.metadata.get('filename')}"
    )