"""DSS report builder.

Assembles all lab outputs into a single structured report dictionary
that the Streamlit dashboard (Lab 8) renders and the FastAPI endpoint
(api/main.py) returns as JSON.

The report is the final user-facing output of the entire system.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.lab9_dss.scoring import InterventionScore


# Urgency thresholds based on model confidence + peak infection rate
_URGENCY_HIGH_CONFIDENCE_THRESHOLD: float = 0.80
_URGENCY_HIGH_PEAK_THRESHOLD: float = 0.40   # fraction of total cells


def _determine_urgency(
    model_confidence: float,
    peak_infected: int,
    total_cells: int,
) -> str:
    """Classify alert urgency as HIGH, MEDIUM, or LOW.

    Args:
        model_confidence: ResNet-50 softmax confidence (0.0–1.0).
        peak_infected:    Maximum infected cell count across all simulation days.
        total_cells:      Total cells in the farm grid.

    Returns:
        String: 'HIGH', 'MEDIUM', or 'LOW'.
    """
    peak_fraction = peak_infected / total_cells if total_cells > 0 else 0.0

    if model_confidence >= _URGENCY_HIGH_CONFIDENCE_THRESHOLD and peak_fraction >= _URGENCY_HIGH_PEAK_THRESHOLD:
        return "HIGH"
    if model_confidence >= _URGENCY_HIGH_CONFIDENCE_THRESHOLD or peak_fraction >= _URGENCY_HIGH_PEAK_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _format_ranked_strategies(ranked_scores: list[InterventionScore]) -> list[dict[str, Any]]:
    """Convert InterventionScore objects into serialisable dicts.

    Args:
        ranked_scores: Sorted list of InterventionScore from scoring.py.

    Returns:
        List of plain dicts safe for JSON serialisation.
    """
    formatted: list[dict[str, Any]] = []
    for scored_intervention in ranked_scores:
        formatted.append({
            "rank":               scored_intervention.rank,
            "intervention_name":  scored_intervention.intervention_name,
            "strategy_key":       scored_intervention.strategy_key,
            "crop_loss_percent":  scored_intervention.crop_loss_percent,
            "final_score":        scored_intervention.final_score,
        })
    return formatted


def build_report(
    disease_name: str,
    model_confidence: float,
    expert_advice: str,
    rag_relevance: float,
    simulation_results: dict[str, Any],
    ranked_scores: list[InterventionScore],
) -> dict[str, Any]:
    """Assemble the complete DSS decision report.

    Args:
        disease_name:        Predicted disease class from Lab 5.
        model_confidence:    ResNet-50 softmax confidence (0.0–1.0).
        expert_advice:       Retrieved treatment text from Lab 4 ChromaDB.
        rag_relevance:       Cosine similarity of RAG retrieval (0.0–1.0).
        simulation_results:  Full output dict from Lab 7 run_simulation().
        ranked_scores:       Scored and sorted interventions from scoring.py.

    Returns:
        Structured report dict with all sections needed by the dashboard.
    """
    best_strategy = ranked_scores[0] if ranked_scores else None

    # Find peak infected across all scenarios for urgency calculation
    peak_infected_across_scenarios = max(
        scenario["peak_infected"]
        for scenario in simulation_results["scenarios"]
    )
    total_cells = simulation_results["total_cells"]

    urgency = _determine_urgency(
        model_confidence=model_confidence,
        peak_infected=peak_infected_across_scenarios,
        total_cells=total_cells,
    )

    # Baseline crop loss (no intervention scenario)
    baseline_scenario = next(
        (s for s in simulation_results["scenarios"] if s["strategy_key"] == "no_intervention"),
        None,
    )
    baseline_crop_loss = baseline_scenario["crop_loss_percent"] if baseline_scenario else None

    # Best strategy crop loss for savings calculation
    best_crop_loss = best_strategy.crop_loss_percent if best_strategy else None
    crop_loss_saving = (
        round(baseline_crop_loss - best_crop_loss, 2)
        if baseline_crop_loss is not None and best_crop_loss is not None
        else None
    )

    # Build scenario data for the dashboard spread chart
    scenarios_for_dashboard: list[dict[str, Any]] = [
        {
            "intervention_name": scenario["intervention_name"],
            "daily_counts":      scenario["daily_counts"],
            "crop_loss_percent": scenario["crop_loss_percent"],
        }
        for scenario in simulation_results["scenarios"]
    ]

    report: dict[str, Any] = {
        # --- Metadata ---
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "disease_name": disease_name,
        "is_healthy": False,
        "urgency_level": urgency,

        # --- Vision Model (Lab 5) ---
        "vision_detection": {
            "predicted_class":  disease_name,
            "confidence":       round(model_confidence * 100, 2),
            "confidence_label": _confidence_label(model_confidence),
        },

        # --- RAG Knowledge (Lab 4) ---
        "rag_treatment_protocol": {
            "retrieved_context": expert_advice,
            "relevance_score":   round(rag_relevance, 4),
        },

        # --- Simulation (Lab 7) ---
        "simulation_summary": {
            "grid_size":           simulation_results["grid_size"],
            "days_simulated":      simulation_results["num_days"],
            "baseline_crop_loss":  baseline_crop_loss,
            "best_crop_loss":      best_crop_loss,
            "crop_loss_saving":    crop_loss_saving,
        },

        # --- DSS Ranking (Lab 9) ---
        "ranked_interventions": _format_ranked_strategies(ranked_scores),

        # --- Simulation scenarios for dashboard charts ---
        "scenarios": scenarios_for_dashboard,

        # --- Top-line Recommendation ---
        "recommendation": {
            "best_strategy":    best_strategy.intervention_name if best_strategy else "Unknown",
            "strategy_key":     best_strategy.strategy_key if best_strategy else "unknown",
            "projected_loss":   best_crop_loss,
            "saving_vs_baseline": crop_loss_saving,
            "urgency":          urgency,
            "action_summary": _build_action_summary(
                disease_name=disease_name,
                best_strategy_name=best_strategy.intervention_name if best_strategy else "Unknown",
                urgency=urgency,
                crop_loss_saving=crop_loss_saving,
            ),
        },
    }

    return report


def _confidence_label(confidence: float) -> str:
    """Map a confidence float to a human-readable label.

    Args:
        confidence: Softmax probability (0.0–1.0).

    Returns:
        String label: 'Very High', 'High', 'Moderate', or 'Low'.
    """
    if confidence >= 0.90:
        return "Very High"
    if confidence >= 0.75:
        return "High"
    if confidence >= 0.50:
        return "Moderate"
    return "Low"


def _build_action_summary(
    disease_name: str,
    best_strategy_name: str,
    urgency: str,
    crop_loss_saving: float | None,
) -> str:
    """Generate a one-paragraph plain-language action summary.

    Args:
        disease_name:       Detected disease class name.
        best_strategy_name: Name of the top-ranked intervention.
        urgency:            Urgency level string.
        crop_loss_saving:   Projected crop loss saving vs no intervention.

    Returns:
        Plain-language string suitable for display in the dashboard.
    """
    saving_text = (
        f"This approach is projected to save approximately {crop_loss_saving:.1f}% "
        f"of your crop compared to taking no action."
        if crop_loss_saving is not None
        else ""
    )

    return (
        f"The system has detected {disease_name.replace('___', ' — ')} "
        f"with {urgency.lower()} urgency. "
        f"Based on simulation and verified agricultural protocols, "
        f"the recommended action is: {best_strategy_name}. "
        f"{saving_text}"
    ).strip()


if __name__ == "__main__":
    mock_ranked = [
        InterventionScore("Quarantine & Removal", "quarantine",      20.0, 0.94, 0.87),
        InterventionScore("Pesticide Application", "pesticide",      35.0, 0.94, 0.87),
        InterventionScore("No Intervention",       "no_intervention", 72.0, 0.94, 0.87),
    ]
    for position, item in enumerate(mock_ranked, start=1):
        item.rank = position

    mock_simulation = {
        "grid_size": [20, 20], "num_days": 30, "total_cells": 400,
        "scenarios": [
            {"strategy_key": "no_intervention", "peak_infected": 180, "crop_loss_percent": 72.0,
             "intervention_name": "No Intervention", "daily_counts": []},
            {"strategy_key": "pesticide",       "peak_infected": 90,  "crop_loss_percent": 35.0,
             "intervention_name": "Pesticide Application", "daily_counts": []},
            {"strategy_key": "quarantine",      "peak_infected": 50,  "crop_loss_percent": 20.0,
             "intervention_name": "Quarantine & Removal", "daily_counts": []},
        ],
    }

    report = build_report(
        disease_name="Tomato___Septoria_leaf_spot",
        model_confidence=0.94,
        expert_advice="Apply chlorothalonil fungicide at 7-day intervals.",
        rag_relevance=0.87,
        simulation_results=mock_simulation,
        ranked_scores=mock_ranked,
    )

    import json
    print(json.dumps({k: v for k, v in report.items() if k != "scenarios"}, indent=2))
    assert report["urgency_level"] in ("HIGH", "MEDIUM", "LOW")
    assert report["is_healthy"] is False
    assert report["recommendation"]["best_strategy"] == "Quarantine & Removal"
    assert "scenarios" in report
    print("Report builder smoke test passed.")