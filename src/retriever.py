from src.query_expansion import *
from src.reranker import rerank
from src.vector_store import search

def retrieve(query,chunks,index,top_k=4,num_queries=5):
    queries = generate_queries(query, num_queries)
    all_results = []
    for q in queries:
        results = search(q, chunks, index)
        all_results.extend(results)
    unique_results = {}
    for score, chunk in all_results:
        key = (chunk['pdf_file'], chunk['chunk_id'])
        if key not in unique_results:
            unique_results[key] = (score, chunk)
        elif score > unique_results[key][0]:
            unique_results[key] = (score, chunk)
    top_results = list(unique_results.values())
    top_results.sort(reverse=True)
    top_results = top_results[:20]
    top_results=rerank(query,top_results)
    return top_results