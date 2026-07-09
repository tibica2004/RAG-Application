import numpy as np
import faiss
from ollama import embed
import json
def create_faiss_index(chunk_embeddings):
    vectors=np.array(chunk_embeddings,dtype=np.float32)
    dimension=vectors.shape[1]
    index=faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index

def search(query, chunks, index, top_k=3):
    query_embedding = embed(
        model="nomic-embed-text",
        input=query
    )["embeddings"][0]

    query_vector=np.array([query_embedding],dtype=np.float32)
    faiss.normalize_L2(query_vector)
    scores,indices=index.search(query_vector,top_k)
    results=[]

    for score,idx in zip(scores[0],indices[0]):
        results.append((score,chunks[idx]))

    return results

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

def save_faiss_index(index,filename="faiss.index"):
    faiss.write_index(index,filename)

def load_faiss_index(filename="faiss.index"):
    return faiss.read_index(filename)