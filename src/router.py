from ollama import chat

def route_query(query):
    prompt = f"""
    You are a routing assistant.

    Your job is to decide whether the user's question requires searching the indexed PDF documents.

    Return ONLY one word.

    chat
    retrieval

    Choose "chat" if:
    - greetings
    - introductions
    - jokes
    - casual conversation
    - questions about yourself
    - general knowledge
    - math
    - programming questions that do NOT explicitly mention the documents

    Choose "retrieval" ONLY if:
    - the user asks about the uploaded PDFs
    - the user asks to summarize a document
    - the answer should come from the indexed documents
    - the user refers to something previously discussed from the documents

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

    return response["message"]["content"].strip().lower()