"""ChromaDB vector store for agricultural knowledge retrieval.

Embeds crop disease documents from data/knowledge_base/crop_data.json
into a persistent ChromaDB collection using HuggingFace sentence
embeddings. Provides a LangChain-compatible retriever for the RAG
pipeline (Lab 4) and chatbot (Lab 1).

Dependencies:
    - chromadb
    - langchain-community
    - sentence-transformers
"""

import json
import os
from typing import Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so they work from any cwd
# ---------------------------------------------------------------------------
_CURRENT_DIR: str = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR: str = os.path.dirname(os.path.dirname(_CURRENT_DIR))
_DATA_PATH: str = os.path.join(_ROOT_DIR, "data", "knowledge_base", "crop_data.json")
_DB_PATH: str = os.path.join(_ROOT_DIR, "vector_db")

# Embedding model
_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"


class AgriculturalVectorDB:
    """Persistent ChromaDB vector store for crop disease knowledge.

    On first use, call load_and_vectorize() to build the DB from
    the JSON knowledge base. After that, get_retriever() loads the
    persisted collection without re-embedding.

    Args:
        persist_directory: Path where ChromaDB stores its data.
    """

    def __init__(self, persist_directory: str = _DB_PATH) -> None:
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)
        self.db: Optional[Chroma] = None

    def load_and_vectorize(self, json_path: str = _DATA_PATH) -> None:
        """Build the vector database from the JSON knowledge base.

        Reads crop_data.json, converts each record to a LangChain
        Document, and indexes them into ChromaDB.

        Args:
            json_path: Path to the crop disease JSON file.

        Raises:
            FileNotFoundError: If json_path does not exist.
        """
        print(f"[VectorDB] Loading knowledge base from: {json_path}")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Knowledge base not found at {json_path}")

        with open(json_path, "r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)

        documents: list[Document] = []
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
                "crop": item["crop"],
            }
            documents.append(Document(page_content=content, metadata=metadata))

        print(f"[VectorDB] Vectorizing {len(documents)} records into ChromaDB...")

        self.db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        print(f"[VectorDB] Database built at: {self.persist_directory}")

    def get_retriever(self, top_k: int = 1):
        """Return a LangChain retriever over the persisted vector store.

        If the DB is not loaded yet, opens the existing persisted store.

        Args:
            top_k: Number of documents to retrieve per query.

        Returns:
            LangChain retriever object.
        """
        if not self.db:
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
        return self.db.as_retriever(search_kwargs={"k": top_k})


if __name__ == "__main__":
    vector_db = AgriculturalVectorDB()
    vector_db.load_and_vectorize()
    print("VectorDB smoke test passed.")