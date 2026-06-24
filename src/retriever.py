from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class ComplaintRetriever:
    def __init__(self, persist_directory="vector_store/"):
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vector_db = Chroma(
                persist_directory=persist_directory, 
                embedding_function=self.embeddings
            )
        except Exception as e:
            print(f"Error loading vector store: {e}")
            self.vector_db = None

    def get_context(self, question, k=5):
        if not self.vector_db:
            return "Error: Database not initialized."
        try:
            results = self.vector_db.similarity_search(question, k=k)
            context = "\n\n".join([doc.page_content for doc in results])
            return context, results
        except Exception as e:
            return f"Retrieval error: {e}", []