import os
from src.pdf_reader import read_pdf, get_pdf_file
from src.chunking import split_into_chunks
from src.embeddings import create_embeddings
from src.vector_store import create_faiss_index, search
from src.rag import generate_answer
from src.vector_store import save_embeddings, load_embeddings, save_faiss_index, load_faiss_index

def main():
    if os.path.exists("embeddings.json") and os.path.exists("faiss.index"):
        print("Încărcarea embeddings din fișier...")

        chunks, chunk_embeddings=load_embeddings()
        index=load_faiss_index()
    else:
        print ("Crearea embeddings...")
        pdf_files = get_pdf_file("data")
        print(pdf_files)
        chunks=[]
        for pdf_file in pdf_files:
            pdf_text = read_pdf(pdf_file)
            pdf_chunks = split_into_chunks(pdf_text, chunk_size=500, overlap=100)
            for i, chunk in enumerate(pdf_chunks):
                chunks.append({
                    "text": chunk,
                    "pdf_file": pdf_file,
                    "chunk_id": i
                })
        chunk_embeddings=create_embeddings(chunks)
        save_embeddings(chunks,chunk_embeddings)
        index=create_faiss_index(chunk_embeddings)
        save_faiss_index(index)
    print(f"Număr chunk-uri: {len(chunks)}")
    print(f"Număr embeddings: {len(chunk_embeddings)}")
    print(f"Dimensiunea unui embedding: {len(chunk_embeddings[0])}")
    conversation=[]
    MAX_TURNS = 5
    MAX_HISTORY = MAX_TURNS * 2
    while True:
        query = input("\nTU: ")
        if query.lower() == "exit":
            print("Ieșire din program.")
            return
        results = search(query, chunks, index)

        answer = generate_answer(query, results,conversation)
        conversation.append({
            "role":"user",
            "content":query
        })

        conversation.append({
            "role":"assistant",
            "content":answer
        })
        
        if len(conversation)>MAX_HISTORY:
            conversation=conversation[-MAX_HISTORY:]
        print("\nRezultate:\n")

        for score, chunk in results:
            print("=" * 60)
            print(f"Similarity: {score:.4f}\n")
            print(f"PDF: {chunk['pdf_file']}")
            print(f"Chunk: {chunk['chunk_id']}")
            print()
            print(chunk)
            print()
        
        print("=" * 60)
        print("Răspunsul modelului:\n")
        print(answer)


if __name__ == "__main__":
    main()