from flashrank import Ranker,RerankRequest

ranker = Ranker()

def rerank(query,results):
    documents=[]
    for i, (_,chunk) in enumerate(results):
        documents.append({
            "id":i,
            "text":chunk["text"]
        }
        )
    request=RerankRequest(
        query=query,
        passages=documents
    )
    reranked=ranker.rerank(request)
    reranked_results=[]
    for item in reranked:
        doc_id=item["id"]
        score,chunk=results[doc_id]
        reranked_results.append((
            item["score"],
            chunk
        ))
    return reranked_results