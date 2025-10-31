from fastapi import HTTPException, APIRouter
from app.schemas.chatagent import QueryRequest, SourceDocument, QueryResponse
from app.utils.flow import getAgent

import asyncio
import time

router = APIRouter()

# --- Endpoint API ---
@router.get("/")
def readRoot():
    """ Endpoint dasar untuk cek status. """
    agentInstance = getAgent()
    status = "OK" if agentInstance else "Loading/Error (Agent NOT ready)"
    return {"status": status, "message": "MateBot API is running."}

@router.post("/chat", response_model=QueryResponse)
async def chatEndpoint(request: QueryRequest):
    try:
        agentInstance = getAgent()
        agent_start_time = time.time()

        if agentInstance is None:
            raise HTTPException(
                status_code=503, 
                detail="Agent belum selesai diinisialisasi atau mengalami kegagalan."
            )
    
        response_data = await asyncio.to_thread(
            agentInstance.generateResponse,
            query=request.query,
            nik=request.nik,      
            token=request.token   
        )

        converted_sources = [
            SourceDocument(
                content=doc.page_content, 
                metadata=doc.metadata     
            ) 
            for doc in response_data["source_documents"]
        ]
        
        agent_end_time = time.time()
        agent_process_time = agent_end_time - agent_start_time
        print(f"DEBUG: Total Agent Thread Execution Time: {agent_process_time:.2f}s") 

        return QueryResponse(
            answer=response_data["answer"],
            source_documents=converted_sources
        )
    except Exception as e:
        print(f"Error saat memproses permintaan chat: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Terjadi kesalahan internal saat memproses query."
        )