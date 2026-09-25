from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None


def get_reranker():
    global _reranker

    if _reranker is None:
        _reranker = CrossEncoder(MODEL_NAME)

    return _reranker


def rerank(query: str, documents: list, top_k: int = 3):
    """
    Rerank retrieved documents using a cross-encoder.

    Args:
        query: User's question.
        documents: Documents retrieved from FAISS.
        top_k: Number of documents to return after reranking.

    Returns:
        Reranked documents.
    """

    if not documents:
        return []

    reranker = get_reranker()

    pairs = [
        (query, document.page_content)
        for document in documents
    ]

    scores = reranker.predict(pairs)

    ranked_documents = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return [
        document
        for document, score in ranked_documents[:top_k]
    ]