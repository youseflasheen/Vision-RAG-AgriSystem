"""RAG generator — retrieves context and generates LLM responses.

Connects the ChromaDB retriever to the Groq LLM to produce
expert agricultural advice grounded in verified knowledge.

Dependencies:
    - langchain-groq
    - GROQ_API_KEY environment variable
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.lab4_rag.vector_store import AgriculturalVectorDB

# Load environment variables for GROQ_API_KEY
_PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))


class RAGGenerator:
    """Retrieval-Augmented Generation pipeline for disease advice.

    Queries ChromaDB for relevant crop disease knowledge, then
    passes the context + disease class to a Groq LLM to generate
    a structured expert response.

    Args:
        model_name: Groq model identifier.
        temperature: LLM sampling temperature.
    """

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
    ) -> None:
        self.vector_db = AgriculturalVectorDB()
        self.retriever = self.vector_db.get_retriever()

        self.llm = ChatGroq(model_name=model_name, temperature=temperature)

        self.prompt_template = ChatPromptTemplate.from_template(
            """You are a professional Senior Agricultural Engineer.
            Answer the inquiry regarding the crop disease.
            First, use the provided context. If the context is missing or insufficient, rely on your vast internal expert knowledge to provide a comprehensive, highly detailed, and accurate response. Do not say you lack context; instead, answer as the expert.

            Context:
            {context}

            Predicted Disease/Topic: {disease_class}

            Provide a technical response including:
            1. Disease Identification
            2. Key Symptoms
            3. Treatment Plan
            4. Prevention Strategy

            Response format: Markdown
            """
        )

        self.rag_chain = (
            {"context": self.retriever, "disease_class": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def generate_advice(self, predicted_class: str) -> str:
        """Generate expert advice for a detected disease class.

        Args:
            predicted_class: PlantVillage disease class name from Lab 5.

        Returns:
            Markdown-formatted expert advice string.
        """
        print(f"\n[RAG] Querying knowledge base for: {predicted_class}...")
        return self.rag_chain.invoke(predicted_class)


if __name__ == "__main__":
    rag = RAGGenerator()
    print("\n" + "=" * 20)
    print("RUNNING TEST: KNOWN DISEASE")
    print("=" * 20)
    known_disease = "Tomato___Early_blight"
    advice = rag.generate_advice(known_disease)
    print(advice)
    print("RAG Generator smoke test passed.")