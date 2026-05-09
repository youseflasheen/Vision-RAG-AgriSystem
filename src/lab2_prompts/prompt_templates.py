"""Prompt template library for the pest detection and decision support system.

All LLM prompt strings are defined here and nowhere else.
Lab 1 (chatbot), Lab 4 (RAG generator), and Lab 9 (DSS) import
from this module so that prompt changes propagate system-wide.

No LLM calls are made in this module — it is pure string construction.
"""

from typing import Optional


# ---------------------------------------------------------------------------
# System personas
# ---------------------------------------------------------------------------

AGRICULTURAL_ENGINEER_PERSONA: str = (
    "You are a professional Senior Agricultural Engineer with 20 years of field experience. "
    "You specialise in crop disease diagnosis and integrated pest management. "
    "You give technically accurate, actionable advice grounded in the provided context. "
    "You never fabricate information — if the context is insufficient, say so clearly. "
    "You always respond in English using plain language a farmer can act on immediately."
)

CHATBOT_PERSONA: str = (
    "You are AgriBot, an AI assistant for farmers built on verified agricultural knowledge. "
    "You answer questions about crop diseases, treatments, and prevention strategies. "
    "You base every answer on the retrieved knowledge context provided to you. "
    "When you are uncertain, you say so and recommend consulting a local agronomist. "
    "Keep responses concise, practical, and free of unnecessary jargon."
)


# ---------------------------------------------------------------------------
# Template 1: Disease analysis (used by Lab 4 RAG generator)
# ---------------------------------------------------------------------------

def build_disease_analysis_prompt(
    disease_class: str,
    rag_context: str,
) -> str:
    """Build a structured disease analysis prompt for the RAG pipeline.

    This replaces the hardcoded prompt in src/rag/generator.py and
    is the canonical version used by both Lab 4 and Lab 9.

    Args:
        disease_class: PlantVillage-format disease class name from Lab 5.
        rag_context:   Retrieved knowledge passages from ChromaDB (Lab 4).

    Returns:
        Formatted prompt string ready for the LLM.
    """
    return (
        f"{AGRICULTURAL_ENGINEER_PERSONA}\n\n"
        f"## Retrieved Knowledge Context\n"
        f"{rag_context}\n\n"
        f"## Detected Disease\n"
        f"{disease_class.replace('___', ' — ')}\n\n"
        f"## Your Task\n"
        f"Using ONLY the context above, provide a technical response with these four sections:\n"
        f"1. **Disease Identification** — confirm the disease and crop affected\n"
        f"2. **Key Symptoms** — what the farmer should look for in the field\n"
        f"3. **Treatment Plan** — specific products and application methods\n"
        f"4. **Prevention Strategy** — long-term steps to avoid recurrence\n\n"
        f"Format your response in Markdown. Be specific and actionable."
    )


# ---------------------------------------------------------------------------
# Template 2: Chatbot system prompt (used by Lab 1)
# ---------------------------------------------------------------------------

def build_chatbot_system_prompt(rag_context: str) -> str:
    """Build the system prompt for the farmer-facing chatbot (Lab 1).

    Injects the RAG-retrieved context so the chatbot answers are
    grounded in verified knowledge rather than LLM parametric memory.

    Args:
        rag_context: Retrieved knowledge passages from ChromaDB (Lab 4).

    Returns:
        System prompt string for the chatbot LLM call.
    """
    return (
        f"{CHATBOT_PERSONA}\n\n"
        f"## Knowledge Base Context (verified agricultural data)\n"
        f"{rag_context}\n\n"
        f"Answer the farmer's question using the context above. "
        f"If the context does not contain enough information to answer confidently, "
        f"state that clearly rather than guessing."
    )


# ---------------------------------------------------------------------------
# Template 3: Intervention explanation (used by Lab 9 DSS)
# ---------------------------------------------------------------------------

def build_intervention_explanation_prompt(
    disease_name: str,
    best_strategy: str,
    crop_loss_saving: float,
    rag_context: str,
) -> str:
    """Build a plain-language explanation of the DSS recommendation.

    Called by the DSS engine to generate a farmer-readable paragraph
    explaining why the top-ranked intervention was chosen.

    Args:
        disease_name:      Detected disease class from Lab 5.
        best_strategy:     Name of the top-ranked intervention from Lab 9.
        crop_loss_saving:  Projected crop loss saving vs no intervention (%).
        rag_context:       RAG-retrieved treatment protocol from Lab 4.

    Returns:
        Formatted prompt string for the explanation LLM call.
    """
    return (
        f"{AGRICULTURAL_ENGINEER_PERSONA}\n\n"
        f"## Situation\n"
        f"The AI system has detected **{disease_name.replace('___', ' — ')}** "
        f"in a field and run a spread simulation comparing three intervention strategies.\n\n"
        f"## Recommended Strategy\n"
        f"**{best_strategy}** was ranked first, projecting to save "
        f"approximately **{crop_loss_saving:.1f}%** of the crop compared to no action.\n\n"
        f"## Supporting Agricultural Protocol\n"
        f"{rag_context}\n\n"
        f"## Your Task\n"
        f"Write 2-3 sentences in plain language explaining to a farmer:\n"
        f"- Why this strategy is recommended\n"
        f"- What they should do in the next 24 hours\n"
        f"Do not use technical scoring numbers. Focus on practical next steps."
    )


# ---------------------------------------------------------------------------
# Template 4: Follow-up question (used by Lab 1 chatbot multi-turn)
# ---------------------------------------------------------------------------

def build_followup_prompt(
    user_question: str,
    conversation_history: list[dict[str, str]],
    rag_context: Optional[str] = None,
) -> list[dict[str, str]]:
    """Build a message list for a follow-up question in the chatbot.

    Formats the full conversation history into the message structure
    expected by the LLM API (list of role/content dicts).

    Args:
        user_question:        The farmer's current question.
        conversation_history: Previous turns as list of
                              {'role': 'user'|'assistant', 'content': str}.
        rag_context:          Optional updated RAG context for this question.
                              If None, relies on the existing system context.

    Returns:
        List of message dicts ready for the LLM messages parameter.
    """
    context_note = (
        f"\n\n[Updated context for this question]\n{rag_context}"
        if rag_context
        else ""
    )

    messages = list(conversation_history)
    messages.append({
        "role": "user",
        "content": user_question + context_note,
    })
    return messages


# ---------------------------------------------------------------------------
# Prompt registry — for inspection and testing
# ---------------------------------------------------------------------------

PROMPT_REGISTRY: dict[str, str] = {
    "disease_analysis":          "build_disease_analysis_prompt",
    "chatbot_system":            "build_chatbot_system_prompt",
    "intervention_explanation":  "build_intervention_explanation_prompt",
    "followup":                  "build_followup_prompt",
}


if __name__ == "__main__":
    # Smoke test — verify all templates render without error
    test_context = "Tomato Early Blight: Apply chlorothalonil fungicide at 7-day intervals."
    test_disease = "Tomato___Early_blight"

    prompt_1 = build_disease_analysis_prompt(test_disease, test_context)
    assert "Disease Identification" in prompt_1
    assert "Early_blight" not in prompt_1   # Should be replaced with " — "
    print("Template 1 (disease analysis): OK")

    prompt_2 = build_chatbot_system_prompt(test_context)
    assert "AgriBot" in prompt_2
    print("Template 2 (chatbot system): OK")

    prompt_3 = build_intervention_explanation_prompt(
        disease_name=test_disease,
        best_strategy="Pesticide Application",
        crop_loss_saving=37.5,
        rag_context=test_context,
    )
    assert "37.5" in prompt_3
    print("Template 3 (intervention explanation): OK")

    prompt_4 = build_followup_prompt(
        user_question="What fungicide should I use?",
        conversation_history=[
            {"role": "user", "content": "My tomatoes look sick."},
            {"role": "assistant", "content": "I see early blight symptoms."},
        ],
        rag_context=test_context,
    )
    assert prompt_4[-1]["role"] == "user"
    assert len(prompt_4) == 3
    print("Template 4 (follow-up): OK")

    print("\nAll prompt templates passed smoke tests.")