from ollama import chat


def generate_queries(query, num_queries=3):
    prompt = f"""
    You are helping a retrieval system.

    Generate {num_queries} different versions of the user's question.

    Rules:
    - Keep the same meaning.
    - Use different wording.
    - Put each question on a new line.
    - Do not number the questions.
    - Do not add explanations.

    Question:
    {query}
    """
    response=chat(
        model="llama3.1:8b",
        messages=[{
            "role":"user",
            "content":prompt,
        }]
    )
    generated_text=response["message"]["content"]
    queries=generated_text.splitlines()
    queries=[q.strip() for q in queries if q.strip()]
    queries = list(dict.fromkeys([query] + queries))
    return queries

