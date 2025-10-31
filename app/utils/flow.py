from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from dotenv import load_dotenv
from urllib.parse import urlparse
from langchain.tools import tool 
from typing import List, Optional
from pydantic import EmailStr
from datetime import date

import os
import functools
import chromadb
import httpx

load_dotenv()
if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GEMINI_API_KEY atau GOOGLE_API_KEY tidak ditemukan di environment variables.")

chromaDB = os.getenv("CHROMA_DB_URL")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8080/api/auth/update")

parsed_url = urlparse(chromaDB)
host = parsed_url.hostname
port = parsed_url.port
client = chromadb.HttpClient(host=host, port=port)

EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
LLM_MODEL = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
INDEX_NAME = "gizi_index"

class DataProcessor:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.textSplitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", " "]
        )

    def processText(self, text: str) -> list[Document]:
        """ Memecah teks panjang menjadi chunks dan konversi ke objek Document """
        chunks = self.textSplitter.split_text(text)
        return [Document(page_content=chunk) for chunk in chunks]
    
class VectorDatabase:
    def __init__(self, embeddingFunction : EMBEDDING_MODEL, vectorStore : Chroma, client : client):
        self.embeddingFunction = embeddingFunction
        self.vectorStore = vectorStore
        self.client = client
    
    def createIndex(self, documents: list[Document]):
        """ Membuat indeks vector store menggunakan FAISS dengan wrapper LangChain """

        self.vectorStore = self.vectorStore.get_or_create_collection("gizi_data")

        for doc in documents:
            content = doc.page_content if hasattr(doc, 'page_content') else doc['content']

            embedding = self.embeddingFunction.embed_documents([content])[0]
            
            self.vectorStore.add(
                [embedding], 
                metadatas=[{
                    "id": doc.get("id", "N/A"),
                    "topic": doc.get("topic", "N/A"),
                    "sub_topic": doc.get("sub_topic", "N/A"),
                    "source": doc.get("source", "N/A"),
                    "date_updated": doc.get("date_updated", "N/A"),
                }],
                ids=[f"doc_{doc['id']}"] 
            )
            
        return self.vectorStore
    
class ChatbotAgent:
    def __init__(self, vectorStore: Chroma, llm_model: ChatGoogleGenerativeAI):
        self.vectorStore = vectorStore
        self.llm = llm_model
        
        self.token = None 
        self.nik = None
        self.last_retrieved_docs: List[Document] = [] 
        
        # --- RAG CONSTANTS ---
        self.DISTANCE_THRESHOLD = 0.35
        self.DOC_COUNT = 3

        @tool
        def Gizi_Retriever_Tool(query: str) -> str:
            """
            Alat untuk mencari informasi spesifik mengenai gizi, kehamilan, dan stunting. 
            Gunakan alat ini untuk menjawab semua pertanyaan yang berhubungan dengan kesehatan/gizi.
            Alat ini akan secara otomatis melakukan filtering berdasarkan relevansi skor (0.35).
            """
            scoredDocs = self.vectorStore.similarity_search_with_score(query, k=self.DOC_COUNT)
            
            retrieved_docs: List[Document] = [doc for doc, score in scoredDocs if score <= self.DISTANCE_THRESHOLD]
            
            self.last_retrieved_docs = retrieved_docs 

            if not retrieved_docs:
                trigger_message = "KONTEKS GIZI TIDAK DITEMUKAN. Silakan jawab pertanyaan ini menggunakan pengetahuan umum Anda HANYA JIKA topik masih berhubungan erat dengan Gizi, Kehamilan, atau Stunting, dan berikan disclaimer (informasi umum)."
                return trigger_message
            else:
                context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
                return context

        @tool
        def updateData(
            nama: Optional[str] | None = None,
            tempat_lahir: Optional[str] | None = None ,
            tanggal_lahir: Optional[date] | None = None ,
            tanggal_kehamilan_pertama: Optional[date] | None = None ,
            pal: Optional[str] | None = None ,
            alamat: Optional[str] | None = None ,
            email: Optional[EmailStr] | None = None ,  
            berat_badan: Optional[float] | None = None ,
            tinggi_badan: Optional[float] | None = None ,
            lingkar_lengan_atas: Optional[float] | None = None 
        ) :
            """
            Alat ini digunakan untuk memperbarui data profil kesehatan ibu hamil. 
            Hanya panggil jika Bunda secara eksplisit meminta untuk mengubah 
            data profil seperti berat badan atau tinggi badan. (Perhatian: Tanggal format YYYY-MM-DD)
            """
            token = self.token 

            if tanggal_kehamilan_pertama:
                tanggal_kehamilan_pertama = tanggal_kehamilan_pertama.isoformat()  
            if tanggal_lahir:
                tanggal_lahir = tanggal_lahir.isoformat() 

            updatePayload = {
                "nama" : nama, "tempat_lahir" : tempat_lahir, "tanggal_lahir" : tanggal_lahir,
                "tanggal_kehamilan_pertama" : tanggal_kehamilan_pertama, "pal" : pal, 
                "alamat" : alamat, "email" : email, "berat_badan" : berat_badan, 
                "tinggi_badan" : tinggi_badan, "lingkar_lengan_atas" : lingkar_lengan_atas
            }

            updatePayload ={k: v for k, v in updatePayload.items() if v is not None}
            print(updatePayload)

            if not updatePayload :
                return "Gagal Memperbaharui data: Tidak ada data yang valid untuk diperbarui."
            
            try :
                headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
                response = httpx.put(f"{API_URL}", json=updatePayload, headers=headers)
                response.raise_for_status()

                return f"Data Anda telah berhasil diperbarui: {updatePayload}"
            except httpx.HTTPStatusError as e:
                return f"Gagal memanggil API Backend (Status {e.response.status_code}). Pastikan token valid."
            except Exception as e:
                return f"Terjadi error saat update data: {str(e)}"
        
        # --- GABUNG SEMUA TOOLS ---
        tools = [Gizi_Retriever_Tool, updateData] 
        
        system_message = """
            Anda adalah Asisten Pakar Gizi dan Pencegahan Stunting (MateBot). Panggil setiap pengguna 'Bunda'. 
            
            1. **Gizi/Stunting:** Wajib gunakan alat 'Gizi_Retriever_Tool'. Jawab berdasarkan konteks RAG.
                - Jika konteks kosong, berikan jawaban umum (Zero-Shot) HANYA untuk topik Gizi/Kehamilan/Stunting.
            2. **Update Data:** Wajib gunakan alat 'updateData'.
            3. **Gaya Bahasa:** Formal, ramah, dan informatif.
            4. **Di Luar Topik:** Jawab singkat: 'Maaf Bunda, saya hanya dilatih untuk memberikan informasi spesifik mengenai gizi, pencegahan stunting, atau mengelola data profil Anda.'
        """
        
        agent_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"), 
        ])

        llm_with_tools = self.llm.bind_tools(tools).bind(stop=["Sekali lagi, Bunda, ini adalah informasi umum."])
        agent_chain = create_tool_calling_agent(llm_with_tools, tools, agent_prompt)
        self.agent_executor = AgentExecutor(agent=agent_chain, 
                                            tools=tools, 
                                            verbose=True,
                                            max_iterations=2)

    def generateResponse(self, query: str, nik: str, token: str): 
        self.nik = nik
        self.token = token
        self.last_retrieved_docs = [] 

        try:
            import time
            invoke_start_time = time.time() 
            
            response = self.agent_executor.invoke({"input": query})
            
            invoke_end_time = time.time() 
            invoke_process_time = invoke_end_time - invoke_start_time
            print(f"DEBUG: LLM/RAG Invoke Time: {invoke_process_time:.2f}s")
            answer = response["output"]
            
            documents = []
            
            update_keywords = ["ubah", "ganti", "perbarui", "update"]
            is_update_request = any(keyword in query.lower() for keyword in update_keywords)

            if not is_update_request:
                sourceDocuments = self.last_retrieved_docs 
                ids = set()
    
                for i, d in enumerate(sourceDocuments):
                    docId = d.metadata.get("id")
                    if docId and docId in ids:
                        continue
                        
                    ids.add(docId)
                    content = d.page_content
                    
                    normalized_md = {
                        "source": "Database Gizi RAG",
                        "id": docId if docId else f"doc_{i}" 
                    }
                    documents.append(Document(page_content=content, metadata=normalized_md))
            
            return {
                "answer": answer,
                "source_documents": documents
            }
        
        finally:
            self.nik = None
            self.token = None
            self.last_retrieved_docs = []
    
_GLOBAL_AGENT_INSTANCE = None 
    
@functools.lru_cache(maxsize=1) 
def initializeAgent() -> ChatbotAgent:
    global _GLOBAL_AGENT_INSTANCE
    
    collection_name = "gizi_data" 
    print(f"Agent mencoba memuat index dari ChromaDB di {host}:{port}")

    try:
        client.get_collection(collection_name)

        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=EMBEDDING_MODEL,
        )
        print("✅ Agent: Vector Store berhasil dimuat dari disk.")

        agent_instance = ChatbotAgent(
            vectorStore=vector_store, 
            llm_model=LLM_MODEL
        )
        
        _GLOBAL_AGENT_INSTANCE = agent_instance
        return agent_instance
    except Exception as e:
        print(f"❌ Agent: Gagal memuat Vector Store. Error: {e}")
        raise RuntimeError("Gagal memuat Vector Store untuk agent.") from e


def getAgent() -> ChatbotAgent:
    return _GLOBAL_AGENT_INSTANCE