import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.rag.vector_store import AgriculturalVectorDB

# Load environment variables for GROQ_API_KEY
load_dotenv()

class RAGGenerator:
    def __init__(self, model_name="llama-3.3-70b-versatile", temperature=0.3):
        # Initialize the local Vector Database
        self.vector_db = AgriculturalVectorDB()
        self.retriever = self.vector_db.get_retriever()
        
        # Initialize Groq LLM with the specified model
        self.llm = ChatGroq(model_name=model_name, temperature=temperature)
        
        # Define the system prompt - Fixed to respond in English
        self.prompt_template = ChatPromptTemplate.from_template(
            """You are a professional Senior Agricultural Engineer.
            Using the provided context, answer the inquiry regarding the crop disease.
            If the context is insufficient, state that you do not have enough information.
            Do NOT hallucinate information.

            Context:
            {context}

            Predicted Disease: {disease_class}

            Provide a technical response including:
            1. Disease Identification
            2. Key Symptoms
            3. Treatment Plan
            4. Prevention Strategy

            Response format: Markdown
            """
        )

        # Connect the RAG pipeline
        self.rag_chain = (
            {"context": self.retriever, "disease_class": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def generate_advice(self, predicted_class):
        print(f"\n[RAG System] Querying Knowledge Base for: {predicted_class}...")
        return self.rag_chain.invoke(predicted_class)

if __name__ == "__main__":
    # Initialize the generator
    rag = RAGGenerator()
    
    # --- TEST 1: Existing Disease ---
    # We use a class name that we know exists in crop_data.json
    print("\n" + "="*20)
    print("RUNNING TEST 1: KNOWN DISEASE")
    print("="*20)
    
    known_disease = "Tomato___Early_blight"
    advice_1 = rag.generate_advice(known_disease)
    print(advice_1)
    
    # --- TEST 2: Unknown Disease ---
    # We use a class name that does NOT exist in our JSON to check grounding
    print("\n" + "="*20)
    print("RUNNING TEST 2: UNKNOWN DISEASE")
    print("="*20)
    
    unknown_disease = "Grape___Black_rot"
    advice_2 = rag.generate_advice(unknown_disease)
    print(advice_2)