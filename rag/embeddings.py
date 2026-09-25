from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def get_embeddings():
    """
    Create and return the OpenAI embedding model.
    """
    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )