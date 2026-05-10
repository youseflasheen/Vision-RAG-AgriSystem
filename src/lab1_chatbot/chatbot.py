"""Domain Q&A chatbot for farmer pest management queries.

Wraps the Lab 4 RAG retriever and Lab 2 prompt templates into a
stateful chat interface. Each session maintains conversation history
so farmers can ask follow-up questions naturally.

Dependencies:
    - src/lab2_prompts/prompt_templates.py  (Lab 2)
    - src/lab4_rag/vector_store.py          (Lab 4)
    - GROQ_API_KEY environment variable
"""

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.lab2_prompts.prompt_templates import (
    build_chatbot_system_prompt,
    build_followup_prompt,
    build_disease_analysis_prompt,
)
from src.lab4_rag.vector_store import AgriculturalVectorDB


def _load_environment_variables() -> None:
    """Load environment variables from stable project paths.

    Uvicorn's reload process can run with a different working directory,
    so relying on implicit dotenv discovery is fragile. This loader uses
    explicit paths relative to this file.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidate_files = [
        os.path.join(project_root, ".env"),
        os.path.join(project_root, ".env.local"),
    ]
    for candidate in candidate_files:
        if os.path.exists(candidate):
            load_dotenv(dotenv_path=candidate, override=False)


_load_environment_variables()

# Maximum conversation turns kept in memory (user + assistant pairs)
MAX_HISTORY_TURNS: int = 6


class AgriculturalChatbot:
    """Stateful farmer Q&A chatbot grounded in the RAG knowledge base.

    Each instance holds one conversation session. Create a new instance
    to start a fresh session (e.g. per Streamlit user session).

    Args:
        model_name:  Groq model identifier.
        temperature: LLM sampling temperature (lower = more factual).
    """

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
    ) -> None:
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError(
                "GROQ_API_KEY is missing. Set it in your shell or add it to "
                "project-root/.env before starting uvicorn."
            )
        self.llm = ChatGroq(model_name=model_name, temperature=temperature)
        self.vector_db = AgriculturalVectorDB()
        self.retriever = self.vector_db.get_retriever()
        self.conversation_history: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _retrieve_context(self, query: str) -> str:
        """Retrieve relevant knowledge passages for a query.

        Args:
            query: Farmer's question or disease class name.

        Returns:
            Concatenated passage text from ChromaDB retrieval.
        """
        documents = self.retriever.invoke(query)
        if not documents:
            return "No specific information found in the knowledge base for this query."
        return "\n\n".join(doc.page_content for doc in documents)

    def _trim_history(self) -> None:
        """Keep conversation history within MAX_HISTORY_TURNS pairs."""
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ask(self, question: str) -> str:
        """Answer a farmer's question using RAG-grounded context.

        On the first message, builds a full system prompt with the
        retrieved context. On follow-up messages, appends the new
        question with updated context to the existing history.

        Args:
            question: Natural language question from the farmer.

        Returns:
            LLM response string.
        """
        rag_context = self._retrieve_context(question)
        is_first_message = len(self.conversation_history) == 0

        if is_first_message:
            system_prompt = build_chatbot_system_prompt(rag_context)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question),
            ]
        else:
            message_dicts = build_followup_prompt(
                user_question=question,
                conversation_history=self.conversation_history,
                rag_context=rag_context,
            )
            messages = []
            for message_dict in message_dicts:
                if message_dict["role"] == "user":
                    messages.append(HumanMessage(content=message_dict["content"]))
                else:
                    messages.append(AIMessage(content=message_dict["content"]))

        response = self.llm.invoke(messages)
        answer = response.content

        # Update history for next turn
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})
        self._trim_history()

        return answer

    def ask_about_detection(
        self,
        disease_class: str,
        confidence: float,
    ) -> str:
        """Generate an expert report for a Lab 5 model detection.

        Args:
            disease_class: PlantVillage class name from ResNet-50.
            confidence:    Softmax confidence of the prediction (0–1).

        Returns:
            Markdown-formatted expert analysis string.
        """
        rag_context = self._retrieve_context(disease_class)
        prompt = build_disease_analysis_prompt(disease_class, rag_context)

        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        answer = response.content

        confidence_note = (
            f"\n\n---\n*Detection confidence: {confidence:.1%} — "
            + ("high confidence result.*" if confidence >= 0.75 else
               "moderate confidence — consider a second opinion.*")
        )

        return answer + confidence_note

    def reset_session(self) -> None:
        """Clear conversation history to start a new session."""
        self.conversation_history = []


if __name__ == "__main__":
    print("AgriBot smoke test — requires GROQ_API_KEY and ChromaDB to be built\n")

    chatbot = AgriculturalChatbot()

    questions = [
        "What are the symptoms of early blight on tomatoes?",
        "What fungicide should I use for that?",
        "How do I prevent it next season?",
    ]

    for question in questions:
        print(f"Farmer: {question}")
        answer = chatbot.ask(question)
        print(f"AgriBot: {answer[:300]}...\n")

    print(f"History length: {len(chatbot.conversation_history)} messages")
    chatbot.reset_session()
    assert len(chatbot.conversation_history) == 0
    print("Chatbot smoke test passed.")