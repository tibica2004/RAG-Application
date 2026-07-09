from ollama import embed

def create_embeddings(chunks):
    texts = [chunk["text"] for chunk in chunks]
    response = embed(
        model="nomic-embed-text",
        input=texts
    )

    return response["embeddings"]
