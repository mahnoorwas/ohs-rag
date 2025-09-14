# data_ingestion.py
import joblib
from langchain_chroma import Chroma
from rag_pipeline import SimpleTFIDFEmbeddings, vectorizer
import PyPDF2

# ---------- Load PDF and split into chunks ----------
def load_pdf_and_split(pdf_path, chunk_size=500):
    """
    Load a PDF and split it into text chunks.
    chunk_size = approximate number of characters per chunk
    """
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    
    # Split text into chunks
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk.strip())
    return chunks

docs = load_pdf_and_split("./ohs_clean.pdf")
print(f"📄 Loaded {len(docs)} chunks from PDF")


embeddings = SimpleTFIDFEmbeddings(vectorizer)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
vectorstore.add_texts(docs)

print(f"✅ Ingested {len(docs)} chunks into ChromaDB")
