from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    token: str  
    nik: str       

class SourceDocument(BaseModel):
    content: str
    metadata: dict

class QueryResponse(BaseModel):
    answer: str
    source_documents: list[SourceDocument]