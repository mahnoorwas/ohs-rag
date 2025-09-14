
import joblib
from langchain_chroma import Chroma

vectorizer = joblib.load("tfidf_vectorizer.pkl")

class SimpleTFIDFEmbeddings:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def embed_documents(self, texts):
        return self.vectorizer.transform(texts).toarray().tolist()

    def embed_query(self, text):
        return self.vectorizer.transform([text]).toarray()[0].tolist()

embeddings = SimpleTFIDFEmbeddings(vectorizer)

def load_vectorstore(persist_directory="./chroma_db"):
    """Load the ChromaDB collection with the TF-IDF embeddings"""
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

def ask_question(query: str, persist_directory="./chroma_db"):
    """Search ChromaDB for similar documents and return concatenated text"""
    vectorstore = load_vectorstore(persist_directory)
    results = vectorstore.similarity_search(query, k=3)
    answer = "\n\n".join([doc.page_content for doc in results])
    return answer

if __name__ == "__main__":
    test_q = "What are the main objectives of an OHS management system?"
    print("❓", test_q)
    print("💡", ask_question(test_q))
