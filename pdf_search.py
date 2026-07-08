import fitz
from ollama import embed, chat 
import numpy as np
import json 
import os

def read_pdf(path: str) -> str:
    """Citește tot textul dintr-un PDF."""
    document = fitz.open(path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text


def split_into_chunks(text: str, chunk_size: int = 500, overlap:int =100):
    chunks = []

    start = 0
    while (start <len(text)):
        end=start+chunk_size
        chunk=text[start:end]
        chunks.append(chunk)
        start+=chunk_size-overlap
    return chunks


def create_embeddings(chunks):
    response = embed(
        model="nomic-embed-text",
        input=chunks
    )

    return response["embeddings"]


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(query, chunks, chunk_embeddings, top_k=3):
    query_embedding = embed(
        model="nomic-embed-text",
        input=query
    )["embeddings"][0]

    results = []

    for chunk, embedding in zip(chunks, chunk_embeddings):
        score = cosine_similarity(query_embedding, embedding)
        results.append((score, chunk))

    results.sort(reverse=True)

    return results[:top_k]

def generate_answer(query,results):
    context = "\n\n".join(chunk for _, chunk in results)

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the information from the context below.

If the answer cannot be found in the context, reply:
"I don't know."

Context:
{context}

Question:
{query}
"""

    response = chat(
        model="llama3.1:8b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]

def save_embeddings(chunks,chunk_embeddings,filename="embeddings.json"):
    data={
        "chunks":chunks,
        "embeddings":chunk_embeddings
    }

    with open(filename,"w", encoding="utf-8") as f:
        json.dump(data,f)

def load_embeddings(filename="embeddings.json"):
    with open(filename,"r",encoding="utf-8") as f:
        data=json.load(f)

    return data["chunks"], data["embeddings"]

def main():
    if os.path.exists("embeddings.json"):
        print("Încărcarea embeddings din fișier...")

        chunks, chunk_embeddings=load_embeddings()
    else:
        print ("Crearea embeddings...")
        pdf_text = read_pdf(
        "/home/besleaga/Projects/AI/01-embeddings/data/Summer_Practice_2026_PetPal.pdf"
    )
        chunks=split_into_chunks(pdf_text,chunk_size=500,overlap=100)
        chunk_embeddings=create_embeddings(chunks)
        save_embeddings(chunks,chunk_embeddings)
    print(f"Număr chunk-uri: {len(chunks)}")
    print(f"Număr embeddings: {len(chunk_embeddings)}")
    print(f"Dimensiunea unui embedding: {len(chunk_embeddings[0])}")

    query = input("\nÎntrebare: ")

    results = search(query, chunks, chunk_embeddings)

    answer = generate_answer(query, results)

    print("\nRezultate:\n")

    for score, chunk in results:
        print("=" * 60)
        print(f"Similarity: {score:.4f}\n")
        print(chunk)
        print()
    
    print("=" * 60)
    print("Răspunsul modelului:\n")
    print(answer)


if __name__ == "__main__":
    main()