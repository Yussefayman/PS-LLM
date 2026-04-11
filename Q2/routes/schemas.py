from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str


class RetrievedChunk(BaseModel):
    id: int
    text: str
    score: float
    metadata: dict = {}


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[RetrievedChunk]