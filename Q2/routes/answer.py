import os
import time
import logging
from fastapi import APIRouter, Query, Request
from groq import AsyncGroq
from dotenv import load_dotenv

from rag_pipeline import retrieve
from routes.schemas import QueryRequest, QueryResponse, RetrievedChunk
from routes.guardrail import check_denylist, check_pii, check_confidence

load_dotenv()

logger      = logging.getLogger(__name__)
router      = APIRouter()
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
LLM_MODEL   = os.getenv("LLM_MODEL", "llama3-8b-8192")


async def generate_answer(query: str, context: str) -> str:
    res = await groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"Answer the question using ONLY the context below. "
                f"If the answer is not in the context, say you don't know.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}"
            )
        }],
        temperature=0,
        max_tokens=200,
    )
    return res.choices[0].message.content.strip()


@router.post("/answer", response_model=QueryResponse)
async def answer(
    http_request: Request,
    request: QueryRequest,
    k: int      = Query(default=3, ge=1, le=10),
    metric: str = Query(default="cosine", pattern="^(cosine|dot)$"),
):
    start = time.time()

    check_denylist(request.query)
    check_pii(request.query)

    index     = http_request.app.state.index
    snippets  = retrieve(request.query, index, k=k, metric=metric)
    top_score = snippets[0]["score"] if snippets else 0.0

    check_confidence(top_score)

    answer_text = await generate_answer(request.query, snippets[0]["text"])
    latency_ms  = round((time.time() - start) * 1000, 2)

    logger.info(f"query='{request.query}' top_score={top_score:.3f} latency_ms={latency_ms} k={k} metric={metric}")

    return QueryResponse(
        query=request.query,
        answer=answer_text,
        sources=[RetrievedChunk(**s) for s in snippets],
    )