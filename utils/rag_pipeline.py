from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

from utils.cohere_client import get_embedding, generate_answer


class CohereEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [get_embedding(text, input_type="search_document") for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return get_embedding(text, input_type="search_query")


def build_vector_store(text: str):
    if not text or not text.strip():
        raise ValueError("Uploaded file does not contain readable text.")

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]

    if not chunks:
        raise ValueError("No text chunks were generated from the uploaded file.")

    docs = [Document(page_content=chunk) for chunk in chunks]

    embedding_model = CohereEmbeddings()

    vector_store = FAISS.from_documents(
        documents=docs,
        embedding=embedding_model
    )

    return vector_store


def retrieve_answer(query: str, vector_store):
    docs = vector_store.similarity_search(query, k=3)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

    return generate_answer(prompt)