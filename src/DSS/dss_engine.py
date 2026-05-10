"""DSS Engine — main orchestrator for the Decision Support System.

Single entry point called by the FastAPI backend (api/main.py).
Wires together vision inference, simulation, RAG, scoring, and reporting:

    ResNet-50 prediction + confidence
        -> Spread simulation (src/simulation/simulator.py)
            -> RAG treatment advice (src/rag/generator.py)
                -> Intervention scoring (src/DSS/scoring.py)
                    -> Structured report (src/DSS/report.py)
                        -> returned to API -> frontend

Usage from FastAPI:
    from src.DSS.dss_engine import run_dss
    report = run_dss(disease_name="Tomato___Septoria_leaf_spot", model_confidence=0.94)
"""

from __future__ import annotations

import sys
import os
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — allows importing sibling modules when called from api/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SIMULATION_PATH = os.path.join(_PROJECT_ROOT, "src", "simulation")
_DSS_PATH = os.path.join(_PROJECT_ROOT, "src", "DSS")

for _path in (_SIMULATION_PATH, _DSS_PATH):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from simulator import run_simulation
from scoring import rank_interventions, InterventionScore
from report import build_report


# ---------------------------------------------------------------------------
# RAG integration — calls the real RAGGenerator from src/rag/generator.py
# ---------------------------------------------------------------------------

def _query_rag(disease_name: str) -> tuple[str, float]:
    """Retrieve treatment advice from the RAG pipeline.

    Calls RAGGenerator.generate_advice() from src/rag/generator.py,
    which queries ChromaDB and generates a Groq LLM response.
    Falls back gracefully if the RAG service is unavailable.

    Args:
        disease_name: Predicted disease class name from ResNet-50.

    Returns:
        Tuple of (expert_advice_markdown, relevance_score).
    """
    try:
        rag_src_path = os.path.join(_PROJECT_ROOT, "src")
        if rag_src_path not in sys.path:
            sys.path.insert(0, rag_src_path)

        from rag.generator import RAGGenerator  # actual module in src/rag/generator.py

        rag = RAGGenerator()
        advice = rag.generate_advice(disease_name)

        # RAGGenerator returns a markdown string — relevance is not exposed
        # by the current implementation so we use a fixed high-confidence value
        return str(advice), 0.90

    except Exception as rag_error:
        print(f"[DSS] RAG query failed: {rag_error}. Using fallback context.")
        crop = disease_name.split("___")[0].replace("_", " ")
        disease = disease_name.split("___")[-1].replace("_", " ") if "___" in disease_name else disease_name
        fallback = (
            f"## {crop} — {disease}\n\n"
            "**Note:** Knowledge base temporarily unavailable. "
            "General protocol: apply appropriate fungicide or pesticide per local "
            "agricultural extension guidelines. Monitor spread daily and isolate affected zones."
        )
        return fallback, 0.60


# ---------------------------------------------------------------------------
# Main DSS entry point
# ---------------------------------------------------------------------------

def run_dss(
    disease_name: str,
    model_confidence: float,
    simulation_days: int = 30,
    grid_rows: int = 20,
    grid_cols: int = 20,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Run the full Decision Support System pipeline for one detection.

    Args:
        disease_name:      Predicted disease class from ResNet-50.
        model_confidence:  Softmax confidence of the prediction (0.0–1.0).
        simulation_days:   Number of days to run spread simulation.
        grid_rows:         Rows in the farm grid.
        grid_cols:         Columns in the farm grid.
        random_seed:       Seed for reproducible simulation results.

    Returns:
        Complete DSS report dict (see report.py for schema).

    Raises:
        ValueError: If model_confidence is outside [0.0, 1.0].
    """
    if not 0.0 <= model_confidence <= 1.0:
        raise ValueError(
            f"model_confidence must be in [0.0, 1.0], got {model_confidence}"
        )

    # --- Healthy plant short-circuit ---
    if "healthy" in disease_name.lower():
        crop_name = disease_name.split("___")[0].replace("_", " ")
        return {
            "disease_name":   disease_name,
            "is_healthy":     True,
            "urgency_level":  "NONE",
            "vision_detection": {
                "predicted_class":  disease_name,
                "confidence":       round(model_confidence * 100, 2),
                "confidence_label": "High Confidence" if model_confidence >= 0.75 else "Moderate Confidence",
            },
            "rag_treatment_protocol": {
                "retrieved_context": (
                    f"## {crop_name} — No Disease Detected\n\n"
                    "Your crop appears **healthy**. No treatment is required at this time.\n\n"
                    "**Recommended Preventive Measures:**\n"
                    "- Continue regular monitoring every 7–10 days\n"
                    "- Maintain proper irrigation and avoid waterlogging\n"
                    "- Ensure good air circulation between plants\n"
                    "- Apply balanced fertilizer per crop schedule\n"
                    "- Keep records of field conditions for early detection"
                ),
                "relevance_score": 1.0,
            },
            "simulation_summary":    None,
            "ranked_interventions":  [],
            "recommendation": {
                "best_strategy": "No Action Required",
                "action_summary": (
                    f"The AI system detected a healthy {crop_name} plant with "
                    f"{model_confidence:.1%} confidence. "
                    "No intervention is needed. Continue your current farming practices "
                    "and schedule the next monitoring check in 7–10 days."
                ),
                "projected_loss": 0.0,
            },
            "scenarios": [],
        }

    # --- Step 1: Spread simulation ---
    print(f"[DSS] Running simulation for: {disease_name}")
    simulation_results = run_simulation(
        disease_name=disease_name,
        num_days=simulation_days,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        random_seed=random_seed,
    )

    # --- Step 2: RAG treatment advice ---
    print("[DSS] Querying RAG knowledge base...")
    rag_context, rag_relevance = _query_rag(disease_name)

    # --- Step 3: Score and rank interventions ---
    print("[DSS] Scoring interventions...")
    ranked_scores: list[InterventionScore] = rank_interventions(
        simulation_results=simulation_results,
        model_confidence=model_confidence,
        rag_relevance=rag_relevance,
    )

    # --- Step 4: Build structured report ---
    print("[DSS] Building report...")
    report = build_report(
        disease_name=disease_name,
        model_confidence=model_confidence,
        expert_advice=rag_context,
        rag_relevance=rag_relevance,
        simulation_results=simulation_results,
        ranked_scores=ranked_scores,
    )

    print(
        f"[DSS] Done. Urgency: {report['urgency_level']} | "
        f"Best strategy: {report['recommendation']['best_strategy']} | "
        f"Projected loss: {report['recommendation']['projected_loss']}%"
    )

    return report


# ---------------------------------------------------------------------------
# FastAPI integration helper
# ---------------------------------------------------------------------------

def run_dss_from_api_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Convenience wrapper for the FastAPI /full_analysis/ endpoint.

    Args:
        payload: Dict with 'predicted_class' (str) and 'confidence' (float).
                 Optional: 'simulation_days', 'grid_rows', 'grid_cols'.

    Returns:
        DSS report dict.

    Raises:
        KeyError: If required keys are missing.
    """
    for key in ("predicted_class", "confidence"):
        if key not in payload:
            raise KeyError(f"Missing required payload key: '{key}'")

    return run_dss(
        disease_name=payload["predicted_class"],
        model_confidence=float(payload["confidence"]),
        simulation_days=int(payload.get("simulation_days", 30)),
        grid_rows=int(payload.get("grid_rows", 20)),
        grid_cols=int(payload.get("grid_cols", 20)),
    )


if __name__ == "__main__":
    import json

    test_report = run_dss(
        disease_name="Tomato___Septoria_leaf_spot",
        model_confidence=0.94,
    )
    printable = {k: v for k, v in test_report.items() if k != "scenarios"}
    print(json.dumps(printable, indent=2))
    assert test_report["recommendation"]["best_strategy"] != "Unknown"
    print("\nDSS Engine smoke test passed.")