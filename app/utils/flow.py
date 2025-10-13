from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from dotenv import load_dotenv
from app.utils.tools import UPDATE_TOOLS

import os
import functools

load_dotenv()
if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GEMINI_API_KEY atau GOOGLE_API_KEY tidak ditemukan di environment variables.")

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
    def __init__(self, embeddingFunction: GoogleGenerativeAIEmbeddings):
        self.embeddingFunction = embeddingFunction
        self.vectorStore = None
    
    def createIndex(self, documents: list[Document]):
        """ Membuat indeks vector store menggunakan FAISS dengan wrapper LangChain """

        self.vectorStore = FAISS.from_documents(
            documents=documents, 
            embedding=self.embeddingFunction
        )
        return self.vectorStore
    
class Retriever:
    def __init__(self, vectorStore: FAISS):
        self.retriever = vectorStore.as_retriever(search_kwargs={"k": 5})
    
    def retrieve(self, query: str) -> list[Document]:
        """ Melakukan pencarian pada vector store untuk menemukan data relevan """

        return self.retriever.invoke(query)
    
class ChatbotAgent:
    def __init__(self, vectorStore: FAISS, llm_model: ChatGoogleGenerativeAI):
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
            response = self.agent_executor.invoke({"input": query})
            
            return {
                "answer": response["output"],
                "source_documents": [] 
            }
        
        finally:
            self.nik = None
            self.token = None
    
_GLOBAL_AGENT_INSTANCE = None 
    
@functools.lru_cache(maxsize=1) 
def initializeAgent() -> ChatbotAgent:
    global _GLOBAL_AGENT_INSTANCE
    
    """ Memuat Vector Store dan membuat instance ChatbotAgent. """
    print(f"Agent mencoba memuat index dari: {VECTOR_INDEX_FOLDER}")
    try:
        vector_store = FAISS.load_local(
            folder_path=VECTOR_INDEX_FOLDER, 
            embeddings=EMBEDDING_MODEL,
            index_name=INDEX_NAME,
            allow_dangerous_deserialization=True
        )
        print("✅ Agent: Vector Store berhasil dimuat dari disk.")
        agent_instance = ChatbotAgent(
        vectorStore=vector_store, 
        llm_model=LLM_MODEL)
        
        _GLOBAL_AGENT_INSTANCE = agent_instance
        return agent_instance
    except Exception as e:
        print(f"❌ Agent: Gagal memuat Vector Store. Pastikan index ada di {VECTOR_INDEX_FOLDER}. Error: {e}")
        raise RuntimeError("Gagal memuat Vector Store untuk agent.") from e


def getAgent() -> ChatbotAgent:
    return _GLOBAL_AGENT_INSTANCE