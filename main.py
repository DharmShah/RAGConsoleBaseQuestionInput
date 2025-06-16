import os
from utils.file_handler import save_file_and_load_text
from utils.rag_pipeline import build_vector_store, retrieve_answer

def main():
    os.makedirs("docs", exist_ok=True)
    file_path = input("Enter the path of the file to upload: ")

    if not os.path.isfile(file_path):
        print("File not found!")
        return

    print("Processing file...")
    text = save_file_and_load_text(file_path)
    vector_store = build_vector_store(text)
    print("File processed. You can now ask questions!")

    while True:
        query = input("\nAsk your question (or type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            print("Bye! 👋")
            break

        answer = retrieve_answer(query, vector_store)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()
