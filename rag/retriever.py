from pathlib import Path

from langchain_community.vectorstores import FAISS

from rag.embeddings import get_embeddings
from rag.reranker import rerank
from rag.evaluation_trace import record_retrieval


VECTOR_DB_DIR = Path("vector_db")


def get_vector_store():
    embeddings = get_embeddings()

    return FAISS.load_local(
        str(VECTOR_DB_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def search(
    query: str,
    k: int = 4,
    domain: str | None = None,
    rerank_results: bool = True,
):
    """
    Retrieve relevant documents using FAISS and optionally rerank them.

    Args:
        query: User's question.
        k: Number of final documents to return.
        domain: Optional domain filter.
        rerank_results: Whether to apply cross-encoder reranking.
    """

    vector_store = get_vector_store()

    # Retrieve more candidates than the final number needed.
    retrieval_k = max(k * 2, 8)

    if domain is None:
        results = vector_store.similarity_search(
            query,
            k=retrieval_k,
        )
    else:
        results = vector_store.similarity_search(
            query,
            k=retrieval_k,
            filter={"domain": domain},
        )

    if not rerank_results:
       final_results = results[:k]
    else:
       final_results = rerank(
        query,
        results,
        top_k=k,
    )

    record_retrieval(
       query=query,
       domain=domain,
       documents=final_results,
)

    return final_results