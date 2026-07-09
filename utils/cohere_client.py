import cohere
import os
from getpass import getpass
from dotenv import load_dotenv

load_dotenv()

co = None


def get_cohere_client():
    global co

    if co is not None:
        return co

    api_key = os.getenv("COHERE_API_KEY")

    if not api_key:
        api_key = getpass("Enter your Cohere API key: ").strip()

    if not api_key:
        raise RuntimeError("Cohere API key is required to continue.")

    co = cohere.Client(api_key)
    return co


def get_embedding(text: str, input_type: str = "search_document"):
    client = get_cohere_client()
    response = client.embed(
        texts=[text],
        model="embed-english-v3.0",
        input_type=input_type
    )
    return response.embeddings[0]


def generate_answer(prompt: str):
    client = get_cohere_client()
    response = client.chat(message=prompt, model="command-r-plus-08-2024")
    return response.text
