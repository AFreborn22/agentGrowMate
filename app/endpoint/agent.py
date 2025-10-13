from fastapi import HTTPException, APIRouter
from app.schemas.chatagent import QueryRequest, SourceDocument, QueryResponse
from app.utils.flow import getAgent

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
    """ Endpoint untuk berinteraksi dengan Chatbot Agent. """
    agentInstance = getAgent()

    if agentInstance is None:
        raise HTTPException(
            status_code=503, 
            detail="Agent belum selesai diinisialisasi atau mengalami kegagalan."
        )

    try:
        response_data = agentInstance.generateResponse(request.query)

        converted_sources = [
            SourceDocument(
                content=doc.page_content, 
                metadata=doc.metadata     
            ) 
            for doc in response_data["source_documents"]
        ]
        
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