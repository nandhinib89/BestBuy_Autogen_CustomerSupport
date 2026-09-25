from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from rag.embeddings import get_embeddings


DATA_DIR = Path("data")
VECTOR_DB_DIR = Path("vector_db")


# Map each knowledge-base document to its specialist domain.
DOCUMENT_DOMAINS = {
    "01_order_status.txt": "order",
    "02_finding_order_number.txt": "order",
    "03_cancelling_editing_orders.txt": "order",
    "04_shipping_delivery.txt": "order",

    "05_payment_methods.txt": "payment",

    "06_return_exchange_policy.txt": "support",
    "07_return_exchange_store.txt": "support",
    "08_defective_products.txt": "support",
    "09_large_item_returns.txt": "support",
    "10_marketplace_returns.txt": "support",
    "11_geek_squad_repair.txt": "support",

    "12_product_availability_pickup.txt": "product",
}


def load_documents():

    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="[0-9]*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} documents.")

    # Add domain metadata to every document.
    for document in documents:

        filename = Path(document.metadata["source"]).name

        domain = DOCUMENT_DOMAINS.get(filename)

        if domain is None:
            raise ValueError(
                f"No domain mapping found for document: {filename}"
            )

        document.metadata["domain"] = domain
        document.metadata["filename"] = filename

    return documents


def create_vector_store():

    documents = load_documents()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    # Show domain distribution.
    domain_counts = {}

    for chunk in chunks:
        domain = chunk.metadata["domain"]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    print("\nChunks by domain:")

    for domain, count in domain_counts.items():
        print(f"  {domain}: {count}")

    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    vector_store.save_local(str(VECTOR_DB_DIR))

    print(f"\nVector store saved to: {VECTOR_DB_DIR}")

    return vector_store


if __name__ == "__main__":
    create_vector_store()