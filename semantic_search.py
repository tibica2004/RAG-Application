from ollama import embed
import numpy as np

documents = [
    "Cats are domestic animals.",
    "A kitten is sleeping on the sofa.",
    "Dogs are loyal companions.",
    "The car engine is broken.",
    "Python is a programming language.",
    "Artificial Intelligence is changing the world.",
    "Machine learning is a branch of Artificial Intelligence.",
    "Pizza is one of the most popular foods.",
]

query = "Tell me about cats."


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


print("Generating document embeddings...")

doc_response = embed(
    model="nomic-embed-text",
    input=documents
)

doc_embeddings = doc_response["embeddings"]

print("Generating query embedding...")

query_embedding = embed(
    model="nomic-embed-text",
    input=query
)["embeddings"][0]


results = []

for document, embedding in zip(documents, doc_embeddings):

    similarity = cosine_similarity(
        query_embedding,
        embedding
    )

    results.append((similarity, document))


results.sort(reverse=True)

print("\nQuery:")
print(query)

print("\nTop results:\n")

for score, document in results[:3]:
    print(f"{score:.4f}   {document}")