import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.endpoint import agent
from app.utils.flow import initializeAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("myapp")

app = FastAPI(
    title="MateBot RAG API",
    description="API untuk Asisten Pakar Gizi dan Pencegahan Stunting (MateBot).",
    version="1.0.0",
    swagger_ui_parameters={
        "persistAuthorization": True,
        "docExpansion": "none",
    },
)

@app.on_event("startup")
async def startup_event():
    try:
        initializeAgent()
        app.include_router(agent.router, prefix="/api", tags=["chat"])
    except Exception as e:
        raise RuntimeError(f"Gagal terhubung dengan agent: {e}")

@app.middleware("http")
async def logRequests(request: Request, call_next):
    start_time = time.time() # <-- Mulai waktu
    logger.info(f"Request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    end_time = time.time() # <-- Akhir waktu
    process_time = end_time - start_time
    
    # Log durasi total Backend
    logger.info(f"Response status: {response.status_code} | Total Backend Time: {process_time:.2f}s") 
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET, POST, PUT, DELETE"],  
    allow_headers=["*"], 
)

app.include_router(agent.router, prefix="/api", tags=["chat"])

schema = app.openapi()
schema.setdefault("security", [{"HTTPBearer": []}])
app.openapi_schema = schema