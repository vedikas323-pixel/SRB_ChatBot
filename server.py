# server.py — Groq LLM + FastEmbed (no API key required)
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# KEYS
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing.")

PDF_FILE_PATH = "SRB.pdf"

# FAISS store path
STORE_DIR = Path("faiss_store")
STORE_DIR.mkdir(exist_ok=True)
FAISS_INDEX_DIR = STORE_DIR / "faiss_index"

retriever = None
chat_engine = None

# temporary helper for debugging retrieval
def debug_retrieve(query, top_k=5):
    docs = retriever.get_relevant_documents(query)
    out = []
    for i, d in enumerate(docs[:top_k]):
        out.append({
            "rank": i+1,
            "page": d.metadata.get("source_page"),
            "text_snippet": d.page_content[:400].replace("\n"," "),
            "meta": d.metadata
        })
    return out



def build_knowledge_base(force_rebuild=False):
    global retriever, chat_engine

    # Try to load FAISS
    if FAISS_INDEX_DIR.exists() and not force_rebuild:
        try:
            print("Loading saved FAISS index...")
            from langchain_community.embeddings import HuggingFaceEmbeddings

            print("Creating embeddings using HuggingFace MiniLM...")
            embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            vector_store = FAISS.load_local(
                str(FAISS_INDEX_DIR),
                embeddings,
                allow_dangerous_deserialization=True
            )
            retriever = vector_store.as_retriever(search_kwargs={"k": 6})
            print("FAISS loaded successfully.")
        except Exception:
            force_rebuild = True

    # Build FAISS fresh
    if force_rebuild or not FAISS_INDEX_DIR.exists():
        print("Building knowledge base...")

        loader = PyPDFLoader(PDF_FILE_PATH)
        documents = loader.load()

        for i, doc in enumerate(documents):
            if not doc.metadata:
                doc.metadata = {}
                doc.metadata["source_page"] = doc.metadata.get("page", i)

    # Stronger overlap and slightly smaller chunks helps capture context spanning pages
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
        chunks = splitter.split_documents(documents)
        print(f"Chunks created: {len(chunks)}")

        print("Creating embeddings using FastEmbed (no API key)...")
        print("Creating embeddings using Groq Embeddings...")
        from langchain_groq import GroqEmbeddings
        embeddings = GroqEmbeddings(model="nomic-embed-text-v1.5")


        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(str(FAISS_INDEX_DIR))

        retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # Build RAG engine
    print("Building Groq RAG engine...")
    llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.2
)




    prompt = ChatPromptTemplate.from_template("""
Answer strictly using the NMIMS SRB context below.
If the information is missing, say:

“I’m sorry, the Student Resource Book does not provide that information.”

Context:
{context}

Question:
{input}
""")

    doc_chain = create_stuff_documents_chain(llm, prompt)
    chat_engine = create_retrieval_chain(retriever, doc_chain)

    print("RAG engine ready!")


if __name__ == "__main__":
    force = len(sys.argv) > 1 and sys.argv[1].lower() in ("rebuild", "force")
    build_knowledge_base(force)
    print("✔ Finished building knowledge base.")

# Auto-load FAISS + RAG engine when imported by app.py
if chat_engine is None or retriever is None:
    print("Loading RAG knowledge base (import)...")
    build_knowledge_base(force_rebuild=False)
