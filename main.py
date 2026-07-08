from ollama import embed
import numpy as np

sentences = [
    "The cat is sleeping.",
    "The tomcat is taking a nap.",
    "I bought a car.",
    "The car engine is broken."
]

response =embed(model="nomic-embed-text",
                input=sentences)
embeddings = response["embeddings"]


def cosine_similarity(vec1, vec2):
    a=np.array(vec1)
    b=np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print ("\nSimilaritati:\n")
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"Similaritatea dintre '{sentences[i]}' si '{sentences[j]}' este: {sim:.4f}") 