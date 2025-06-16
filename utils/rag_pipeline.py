from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from utils.cohere_client import get_embedding, generate_answer
import numpy as np

class CohereEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [get_embedding(text) for text in texts]

    def embed_query(self, text):
        return get_embedding(text)

def build_vector_store(text: str):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    embedding_model = CohereEmbeddings()
    vector_store = FAISS.from_documents(docs, embedding_model)
    return vector_store

def retrieve_answer(query: str, vector_store):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}"
    return generate_answer(prompt)
