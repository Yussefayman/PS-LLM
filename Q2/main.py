import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from rag_pipeline import build_index
from routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.index = build_index("data/corpus.json")
    yield


app = FastAPI(title="Banking RAG API", lifespan=lifespan)
app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}