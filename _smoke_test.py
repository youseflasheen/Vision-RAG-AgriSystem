"""Quick smoke test for the entire restructured project."""
import json

# Test 1: Simulation pipeline
from src.lab7_simulation.simulator import run_simulation
sim = run_simulation("Tomato___Early_blight", num_days=10, grid_rows=10, grid_cols=10)
print(f"[PASS] Simulation: {len(sim['scenarios'])} scenarios, best={sim['best_strategy']}")

# Test 2: DSS scoring
from src.lab9_dss.scoring import rank_interventions
ranked = rank_interventions(sim, model_confidence=0.90, rag_relevance=0.85)
print(f"[PASS] Scoring: {len(ranked)} interventions ranked, #1={ranked[0].intervention_name}")

# Test 3: DSS report
from src.lab9_dss.report import build_report
report = build_report(
    disease_name="Tomato___Early_blight",
    model_confidence=0.90,
    expert_advice="Apply copper fungicide.",
    rag_relevance=0.85,
    simulation_results=sim,
    ranked_scores=ranked,
)
assert report["is_healthy"] is False, "is_healthy should be False for diseased plant"
assert "scenarios" in report, "report must include scenarios key"
assert report["urgency_level"] in ("HIGH", "MEDIUM", "LOW")
print(f"[PASS] Report: urgency={report['urgency_level']}, best={report['recommendation']['best_strategy']}")

# Test 4: DSS engine (healthy path)
from src.lab9_dss.dss_engine import run_dss
healthy = run_dss("Tomato___healthy", model_confidence=0.95)
assert healthy["is_healthy"] is True
print(f"[PASS] Healthy path: is_healthy={healthy['is_healthy']}, urgency={healthy['urgency_level']}")

# Test 5: DSS engine (diseased path — uses fallback RAG since no API key loaded)
diseased = run_dss("Tomato___Septoria_leaf_spot", model_confidence=0.88)
assert diseased["is_healthy"] is False
assert len(diseased["ranked_interventions"]) > 0
print(f"[PASS] Diseased path: urgency={diseased['urgency_level']}, interventions={len(diseased['ranked_interventions'])}")

# Test 6: Lab 3 data modules
from src.lab3_data.collector import DataCollector, CLASS_NAMES
from src.lab3_data.preprocessor import get_training_transforms, get_inference_transforms
assert len(CLASS_NAMES) == 21
t = get_training_transforms()
print(f"[PASS] Lab 3: {len(CLASS_NAMES)} classes, {len(t.transforms)} training transforms")

print("\n" + "=" * 50)
print("  ALL TESTS PASSED — Project is healthy!")
print("=" * 50)
