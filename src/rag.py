from ollama import chat

def generate_answer(query,results):
    context = "\n\n".join(
    chunk["text"] for _, chunk in results
)

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