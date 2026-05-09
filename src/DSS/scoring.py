"""Intervention scoring and ranking for the Decision Support System.

Combines three signal sources into a single intervention score:
  1. Crop loss projection  (from Lab 7 simulation)
  2. Model confidence      (from Lab 5 ResNet-50 prediction)
  3. RAG context relevance (from Lab 4 ChromaDB retrieval)

Lower final score = better intervention (crop loss drives it down).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Scoring weights — must sum to 1.0
# ---------------------------------------------------------------------------
WEIGHT_CROP_LOSS: float = 0.60   # Dominant signal: how much crop is lost
WEIGHT_MODEL_CONFIDENCE: float = 0.25   # How certain the vision model is
WEIGHT_RAG_RELEVANCE: float = 0.15   # How well RAG context matches disease


@dataclass
class InterventionScore:
    """Score record for a single intervention strategy.

    Attributes:
        intervention_name:  Human-readable strategy name.
        strategy_key:       Internal key matching INTERVENTION_REGISTRY.
        crop_loss_percent:  Projected crop loss (0–100).
        model_confidence:   ResNet-50 softmax confidence (0.0–1.0).
        rag_relevance:      RAG retrieval relevance score (0.0–1.0).
        final_score:        Composite score — lower is better.
        rank:               1 = best strategy.
    """
    intervention_name: str
    strategy_key: str
    crop_loss_percent: float
    model_confidence: float
    rag_relevance: float
    final_score: float = field(init=False)
    rank: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.final_score = _compute_final_score(
            crop_loss_percent=self.crop_loss_percent,
            model_confidence=self.model_confidence,
            rag_relevance=self.rag_relevance,
        )


def _compute_final_score(
    crop_loss_percent: float,
    model_confidence: float,
    rag_relevance: float,
) -> float:
    """Compute a composite score for one intervention.

    Formula:
        score = (crop_loss_norm * W_crop)
              - (model_confidence * W_model)
              - (rag_relevance * W_rag)

    Crop loss is normalised to [0, 1] by dividing by 100.
    Confidence and relevance are already in [0, 1].
    Subtracting confidence and relevance rewards scenarios where the
    system is certain — a high-confidence detection with a well-matched
    RAG context should push the score down (i.e. better).

    Args:
        crop_loss_percent: Projected crop loss percentage (0–100).
        model_confidence:  ResNet-50 softmax confidence (0.0–1.0).
        rag_relevance:     RAG cosine similarity score (0.0–1.0).

    Returns:
        Float composite score. Lower = better intervention outcome.
    """
    crop_loss_normalised = crop_loss_percent / 100.0

    score = (
        (crop_loss_normalised * WEIGHT_CROP_LOSS)
        - (model_confidence * WEIGHT_MODEL_CONFIDENCE)
        - (rag_relevance * WEIGHT_RAG_RELEVANCE)
    )
    return round(score, 4)


def rank_interventions(
    simulation_results: dict[str, Any],
    model_confidence: float,
    rag_relevance: float,
) -> list[InterventionScore]:
    """Score and rank all intervention scenarios from the simulation.

    Args:
        simulation_results: Output dict from Lab 7 run_simulation().
        model_confidence:   Softmax confidence from Lab 5 ResNet-50.
        rag_relevance:      Cosine similarity from Lab 4 RAG retrieval.

    Returns:
        List of InterventionScore objects sorted best (rank 1) to worst.
    """
    scores: list[InterventionScore] = []

    for scenario in simulation_results["scenarios"]:
        intervention_score = InterventionScore(
            intervention_name=scenario["intervention_name"],
            strategy_key=scenario["strategy_key"],
            crop_loss_percent=scenario["crop_loss_percent"],
            model_confidence=model_confidence,
            rag_relevance=rag_relevance,
        )
        scores.append(intervention_score)

    # Sort ascending — lower final_score is better
    scores.sort(key=lambda scored_intervention: scored_intervention.final_score)

    for position, scored_intervention in enumerate(scores, start=1):
        scored_intervention.rank = position

    return scores


if __name__ == "__main__":
    # Smoke test with mock simulation output
    mock_simulation = {
        "scenarios": [
            {"intervention_name": "No Intervention",     "strategy_key": "no_intervention", "crop_loss_percent": 72.0},
            {"intervention_name": "Pesticide Application","strategy_key": "pesticide",       "crop_loss_percent": 35.0},
            {"intervention_name": "Quarantine & Removal", "strategy_key": "quarantine",      "crop_loss_percent": 20.0},
        ]
    }
    ranked = rank_interventions(
        simulation_results=mock_simulation,
        model_confidence=0.94,
        rag_relevance=0.87,
    )
    for scored in ranked:
        print(f"  Rank {scored.rank}: {scored.intervention_name:30s} score={scored.final_score:.4f}")

    assert ranked[0].rank == 1
    print("Scoring smoke test passed.")