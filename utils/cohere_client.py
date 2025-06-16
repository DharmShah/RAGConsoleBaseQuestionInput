import cohere
import os

co = cohere.Client("dgC5thM9IhUhtg0q7ibSjCOgBQWzSh9UvGW7O8Df")  # 🔑 Replace with your Cohere API key


def get_embedding(text: str):
    response = co.embed(
        texts=[text],
        model="embed-english-v3.0",
        input_type="search_document"  # 👈 required!
    )
    return response.embeddings[0]


def generate_answer(prompt: str):
    response = co.chat(message=prompt, model="command-r-plus")
    return response.text
