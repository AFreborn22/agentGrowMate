from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader
from langchain_core.prompts import ChatPromptTemplate
from handleUpdateData import updateTool
from dotenv import load_dotenv

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
        self.tools = [updateTool]
    
        # PROMPT RAG
        self.promptTemplate = ChatPromptTemplate.from_messages([
            ("system", """
                Anda adalah Asisten Pakar Gizi dan Pencegahan stunting untuk ibu hamil bernama MateBot panggil setiap user Bunda. Jawablah pertanyaan pengguna **HANYA** berdasarkan konteks yang diberikan di bawah.
                Pastikan jawaban Anda:
                1. Menggunakan bahasa Indonesia formal, ramah, dan informatif.
                2. Menyebutkan **Definisi**, **Penyebab**, dan **Pencegahan** jika relevan.
                3. Gunakan bullet point atau penomoran untuk memudahkan pembacaan.

                KONTEKS:
                {context}
                """),
                ("user", "{query}")
        ])

        self.FALLBACK_PROMPT = ChatPromptTemplate.from_messages([
            ("system", """
                Bunda bertanya tentang '{query}'. Informasi spesifik tidak ditemukan di basis data gizi MateBot.
                Jawab pertanyaan ini menggunakan pengetahuan umum Anda HANYA JIKA topik tersebut masih berhubungan erat dengan Gizi, Kehamilan, atau Stunting. 
                # Jika pertanyaan sama sekali tidak berhubungan dengan topik ini, jawab: 'Maaf Bunda, saya hanya dilatih untuk memberikan informasi spesifik mengenai gizi dan pencegahan stunting.' 
                Berikan jawaban dengan memanggil user 'Bunda' dan berikan disclaimer bahwa ini adalah informasi umum.
            """),
            ("user", "{query}")
        ])
    
    def generateResponse(self, query: str):
        """ Menghasilkan respons menggunakan retriever dan model generatif (Gemini) """
        
        DOC_COUNT = 5 
        DISTANCE_THRESHOLD = 0.4 

        scoredDocs = self.vectorStore.similarity_search_with_score(query, k=DOC_COUNT)
        
        bestDistance = scoredDocs[0][1] if scoredDocs else 999.0
        
        retrievedData = []
        
        if bestDistance > DISTANCE_THRESHOLD:
            response = self.llm.invoke(self.FALLBACK_PROMPT.format(query=query))
            retrievedData.append(Document(page_content="[SUMBER: Pengetahuan Umum MateBot (Tidak bersumber dari data gizi spesifik).]", metadata={"source": "Gemini Knowledge"}))
            
        else:
            retrievedData = self.retriever.retrieve(query)
            context = "\n---\n".join([doc.page_content for doc in retrievedData])
            
            response = self.llm.invoke(
                self.promptTemplate.format_messages(context=context, query=query)
            )

        # for tool in self.tools:
        #     if tool.name == "Update Data" and "berat badan" in query.lower():
        #         response = tool.func(query, "nik") 
        #         break
        #     else:
        #         response = self.llm.invoke(self.promptTemplate.format_messages(context=context, query=query))

        return {
            "answer": response.content,
            "source_documents": retrievedData
        }
    
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