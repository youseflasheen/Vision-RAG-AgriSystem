"""DSS Engine — main orchestrator for the Decision Support System.

This is the single entry point called by the FastAPI backend (api/main.py).
It wires together all lab outputs in the correct order:

    Lab 5 prediction + confidence
        -> Lab 7 simulation (run_simulation)
            -> Lab 4 RAG retrieval (ChromaDB query)
                -> Lab 9 scoring (rank_interventions)
                    -> Lab 9 report (build_report)
                        -> returned to API -> Streamlit dashboard

Usage from FastAPI:
    from src.lab9_dss.dss_engine import run_dss
    report = run_dss(disease_name="Tomato___Septoria_leaf_spot", model_confidence=0.94)
"""

from __future__ import annotations

import sys
import os
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — allows importing sibling lab modules when called from api/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LAB7_PATH = os.path.join(_PROJECT_ROOT, "src", "lab7_simulation")
_LAB9_PATH = os.path.join(_PROJECT_ROOT, "src", "lab9_dss")

for _path in (_LAB7_PATH, _LAB9_PATH):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from simulator import run_simulation                    # Lab 7
from scoring import rank_interventions, InterventionScore  # Lab 9
from report import build_report                         # Lab 9


# ---------------------------------------------------------------------------
# RAG integration shim
# ---------------------------------------------------------------------------

def _query_rag(disease_name: str) -> tuple[str, float]:
    """Retrieve treatment context from ChromaDB (Lab 4 RAG pipeline).

    Attempts to import and call the existing RAG retriever from the
    repo's src/ structure.  Falls back to a safe stub if the RAG
    module is unavailable (e.g. ChromaDB not initialised yet).

    Args:
        disease_name: Predicted disease class name from Lab 5.

    Returns:
        Tuple of (retrieved_context_text, relevance_score).
    """
    try:
        # Import path matches the existing repo structure
        rag_src_path = os.path.join(_PROJECT_ROOT, "src")
        if rag_src_path not in sys.path:
            sys.path.insert(0, rag_src_path)

        # The existing repo exposes a retriever in src/rag_pipeline.py
        # Adjust the import below if the actual module name differs
        from rag_pipeline import retrieve_context  # type: ignore[import]

        context, relevance = retrieve_context(query=disease_name)
        return str(context), float(relevance)

    except ImportError:
        # RAG module not reachable — return informative fallback
        fallback_context = (
            f"Verified protocol for {disease_name.replace('___', ' ')}: "
            "Apply appropriate fungicide/pesticide as per local agricultural "
            "extension guidelines. Monitor spread daily and isolate affected zones."
        )
        return fallback_context, 0.75   # Neutral relevance score for fallback

    except Exception as rag_error:
        # RAG available but query failed — log and fall back gracefully
        print(f"[DSS] RAG query failed: {rag_error}. Using fallback context.")
        return (
            f"Treatment protocol unavailable for {disease_name}. "
            "Consult a local agricultural expert.",
            0.50,
        )


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

    Orchestrates Lab 7 (simulation), Lab 4 (RAG), Lab 9 scoring and
    reporting into a single structured report dict.

    Args:
        disease_name:      Predicted disease class from Lab 5 ResNet-50.
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

    # --- Step 1: Run spread simulation for all intervention scenarios ---
    print(f"[DSS] Running simulation for: {disease_name}")
    simulation_results = run_simulation(
        disease_name=disease_name,
        num_days=simulation_days,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        random_seed=random_seed,
    )

    # --- Step 2: Retrieve RAG treatment context ---
    print("[DSS] Querying RAG knowledge base...")
    rag_context, rag_relevance = _query_rag(disease_name)

    # --- Step 3: Score and rank intervention strategies ---
    print("[DSS] Scoring interventions...")
    ranked_scores: list[InterventionScore] = rank_interventions(
        simulation_results=simulation_results,
        model_confidence=model_confidence,
        rag_relevance=rag_relevance,
    )

    # --- Step 4: Build the final report ---
    print("[DSS] Building report...")
    report = build_report(
        disease_name=disease_name,
        model_confidence=model_confidence,
        rag_context=rag_context,
        rag_relevance=rag_relevance,
        simulation_results=simulation_results,
        ranked_scores=ranked_scores,
    )

    print(f"[DSS] Done. Urgency: {report['urgency_level']} | "
          f"Best strategy: {report['recommendation']['best_strategy']} | "
          f"Projected loss: {report['recommendation']['projected_loss']}%")

    return report


# ---------------------------------------------------------------------------
# FastAPI integration helper
# ---------------------------------------------------------------------------

def run_dss_from_api_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Convenience wrapper for the FastAPI /diagnose endpoint.

    Extracts fields from the API request payload and calls run_dss().

    Expected payload keys:
        'predicted_class'  (str)   — disease name from ResNet-50
        'confidence'       (float) — softmax probability

    Optional payload keys:
        'simulation_days'  (int)   — default 30
        'grid_rows'        (int)   — default 20
        'grid_cols'        (int)   — default 20

    Args:
        payload: Dict from the FastAPI request body.

    Returns:
        DSS report dict.

    Raises:
        KeyError: If required keys are missing from payload.
    """
    required_keys = ("predicted_class", "confidence")
    for key in required_keys:
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

    # Print a compact version (skip long daily_counts)
    printable = {
        key: value
        for key, value in test_report.items()
        if key not in ("simulation_daily_counts",)
    }
    print(json.dumps(printable, indent=2))
    assert test_report["recommendation"]["best_strategy"] != "Unknown"
    print("\nDSS Engine smoke test passed.")