import json
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_PATH = os.path.join(ROOT_DIR, "data", "knowledge_base", "crop_data.json")
DB_PATH = os.path.join(ROOT_DIR, "vector_db")

class AgriculturalVectorDB:
    def __init__(self, persist_directory=DB_PATH):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = None

    def load_and_vectorize(self, json_path=DATA_PATH):
        print(f"Loading Knowledge Base from: {json_path}")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Knowledge base not found at {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []
        for item in data:
            content = (
                f"Crop: {item['crop']}\n"
                f"Disease: {item['disease']}\n"
                f"Symptoms: {item['symptoms']}\n"
                f"Treatment: {item['treatment']}\n"
                f"Prevention: {item['prevention']}"
            )
            
            metadata = {
                "class_name": item["class_name"], 
                "crop": item["crop"]
            }
            
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        print(f"Vectorizing {len(documents)} records into ChromaDB...")
        
        self.db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.db.persist()
        print(f"Vector Database built successfully at: {self.persist_directory}")

    def get_retriever(self):
        if not self.db:
            self.db = Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=self.embeddings
            )
        return self.db.as_retriever(search_kwargs={"k": 1})

if __name__ == "__main__":
    vector_db = AgriculturalVectorDB()
    vector_db.load_and_vectorize()