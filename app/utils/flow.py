from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from dotenv import load_dotenv
from urllib.parse import urlparse
from app.utils.tools import UPDATE_TOOLS

import os
import functools
import chromadb

load_dotenv()
if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GEMINI_API_KEY atau GOOGLE_API_KEY tidak ditemukan di environment variables.")

chromaDB = os.getenv("CHROMA_DB_URL")
parsed_url = urlparse(chromaDB)
host = parsed_url.hostname
port = parsed_url.port

client = chromadb.HttpClient(host=host, port=port)

EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
LLM_MODEL = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

FLOW_DIR = os.path.dirname(os.path.abspath(__file__)) 
APP_DIR = os.path.dirname(FLOW_DIR)
ROOT_DIR = os.path.dirname(APP_DIR)

VECTOR_INDEX_FOLDER = os.path.join(ROOT_DIR, "data", "gizi_vector_index")
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
                [embedding],  # 
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
    
class Retriever:
    def __init__(self, vectorStore : Chroma):
        self.retriever = vectorStore.as_retriever(search_kwargs={"k": 5})
        self.vectorStore = vectorStore
    
    def retrieve(self, query: str) -> list[Document]:
        """ Melakukan pencarian pada vector store untuk menemukan data relevan """
        docs = self.retriever.get_relevant_documents(query)
        return docs
    
class ChatbotAgent:
    def __init__(self, vectorStore: Chroma, llm_model: ChatGoogleGenerativeAI):
        self.vectorStore = vectorStore
        self.llm = llm_model
        self.retriever = Retriever(vectorStore)
        
        self.token = None 
        self.nik = None   

        rag_tool = self.retriever.retriever.as_tool(
            name="Gizi_Retriever", 
            description="Alat untuk mencari informasi spesifik mengenai gizi, kehamilan, dan stunting. Gunakan alat ini untuk menjawab semua pertanyaan yang berhubungan dengan kesehatan/gizi."
        )
        tools = [rag_tool] + UPDATE_TOOLS 
        
        system_message = """
            Anda adalah Asisten Pakar Gizi dan Pencegahan Stunting (MateBot), panggil setiap user 'Bunda'. 
            
            1. **Untuk Pertanyaan Gizi/Stunting:** Wajib gunakan alat 'Gizi_Retriever' untuk mendapatkan konteks. Jawablah berdasarkan konteks yang diberikan oleh alat tersebut.
            2. **Untuk Permintaan Update Data:** Wajib gunakan alat 'updateData' HANYA JIKA Bunda secara eksplisit meminta untuk mengubah data profil (misalnya berat badan, usia, alamat, dll).
            3. **Gaya Bahasa:** Jawab dengan bahasa Indonesia formal, ramah, dan informatif.
            4. **Ketentuan Jawaban RAG:** Jika Anda menggunakan Gizi_Retriever, pastikan jawaban Anda menyebutkan Definisi, Penyebab, dan Pencegahan jika relevan, dan gunakan bullet point atau penomoran.
            5. **Di Luar Topik:** Jika pertanyaan sama sekali tidak berhubungan dengan topik ini dan tidak memerlukan update data, jawab: 'Maaf Bunda, saya hanya dilatih untuk memberikan informasi spesifik mengenai gizi, pencegahan stunting, atau mengelola data profil Anda.'
        """
        
        agent_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"), 
        ])

        llm_with_tools = self.llm.bind_tools(tools)
        agent_chain = create_tool_calling_agent(llm_with_tools, tools, agent_prompt)
        self.agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True)

    def generateResponse(self, query: str, nik: str, token: str): 
        """ 
        Menghasilkan respons menggunakan Agent Executor.
        """
        
        self.nik = nik
        self.token = token

        
        try:
            retrievedDocs = self.retriever.retrieve(query)
            response = self.agent_executor.invoke({"input": query})

            print(retrievedDocs)
            
            ids =set()
            sourceDocuments = []

            for d in retrievedDocs :
                md = d.metadata if hasattr(d, "metadata") else d.get("metadata", "")
                docId = md.get("id")
                if docId :
                    if docId in ids :
                        continue
                    ids.add(docId)
                    sourceDocuments.append(d)

            max_excerpt_chars = 1200
            context_parts = []
            for i, d in enumerate(sourceDocuments):
                content = d.page_content if hasattr(d, "page_content") else d.get("content", "")
                excerpt = content[:max_excerpt_chars].replace("\n", " ")
                md = d.metadata if hasattr(d, "metadata") else d.get("metadata", {}) or {}
                source_label = md.get("source", md.get("id", f"doc_{i}"))
                context_parts.append(f"[Sumber {i+1} - {source_label}]: {excerpt}")

            documents = []
            for i, d in enumerate(sourceDocuments):
                content = d.page_content if hasattr(d, "page_content") else d.get("content", "")
                md = d.metadata if hasattr(d, "metadata") else d.get("metadata", {}) or {}
                normalized_md = {
                    "id": md.get("id", f"doc_{i}"),
                    "source": md.get("source", "-"),
                    "topic": md.get("topic", "-"),
                    "sub_topic": md.get("sub_topic", "-"),
                    "date_updated": md.get("date_updated", "-"),
                    **{k: v for k, v in md.items() if k not in ("id", "source", "topic", "sub_topic", "date_updated")}
                }
                documents.append(Document(page_content=content, metadata=normalized_md))
            
            return {
                "answer": response["output"],
                "source_documents": documents
            }
        
        finally:
            self.nik = None
            self.token = None
    
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