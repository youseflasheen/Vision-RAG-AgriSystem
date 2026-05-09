"""FastAPI backend — main API entry point.

Exposes two endpoints:
    GET  /              — health check
    POST /diagnose/     — vision + RAG only (Lab 5 + Lab 4, original endpoint)
    POST /full_analysis/ — full pipeline: vision + RAG + simulation + DSS
                          (Labs 5, 4, 1, 2, 7, 9 — returns complete DSS report)

Run with:
    uvicorn api.main:app --reload --port 8000
"""

import os
import sys
import shutil
from typing import Any

from fastapi import FastAPI, UploadFile, File, HTTPException

# ---------------------------------------------------------------------------
# Path setup — ensures all src/ lab modules are importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Simulation modules need their own directory on sys.path (they use
# relative imports like `from farm_grid import FarmGrid`)
_SIMULATION_PATH = os.path.join(_PROJECT_ROOT, "src", "simulation")
if _SIMULATION_PATH not in sys.path:
    sys.path.insert(0, _SIMULATION_PATH)

_DSS_PATH = os.path.join(_PROJECT_ROOT, "src", "DSS")
if _DSS_PATH not in sys.path:
    sys.path.insert(0, _DSS_PATH)

from src.vision.inference import AgriVisionRAG
from src.rag.generator import RAGGenerator
from simulator import run_simulation
from dss_engine import run_dss

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AgriVision AI — Pest Detection API",
    description="End-to-end pest detection, RAG advice, simulation and DSS.",
    version="2.0.0",
)

# Globals — initialised once at startup to avoid reloading models per request
_vision_rag_pipeline: AgriVisionRAG = None
_rag_generator: RAGGenerator = None


@app.on_event("startup")
async def startup_event() -> None:
    """Load all models and vector DB into memory at server start."""
    global _vision_rag_pipeline, _rag_generator
    print("[API] Loading vision model and RAG pipeline...")
    _vision_rag_pipeline = AgriVisionRAG()
    _rag_generator = RAGGenerator()
    print("[API] Ready.")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _save_temp_file(uploaded_file: UploadFile) -> str:
    """Save an uploaded file to a temp path and return the path.

    Args:
        uploaded_file: FastAPI UploadFile object.

    Returns:
        Absolute path string of the saved temp file.
    """
    temp_path = os.path.join(_PROJECT_ROOT, f"_temp_{uploaded_file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)
    return temp_path


def _cleanup_temp_file(temp_path: str) -> None:
    """Delete a temp file if it exists.

    Args:
        temp_path: Path to the file to remove.
    """
    if os.path.exists(temp_path):
        os.remove(temp_path)


def _validate_image_filename(filename: str) -> None:
    """Raise HTTPException if the filename is not a supported image type.

    Args:
        filename: Original filename from the upload.

    Raises:
        HTTPException: 400 if the extension is not jpg, jpeg, or png.
    """
    allowed_extensions = (".jpg", ".jpeg", ".png")
    if not filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_extensions}",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "message": "AgriVision AI API is running."}


@app.post("/diagnose/")
async def diagnose_crop(file: UploadFile = File(...)) -> dict[str, Any]:
    """Vision + RAG only — original endpoint kept for compatibility.

    Returns disease class, confidence, and expert RAG advice.
    Does not run simulation or DSS.

    Args:
        file: Uploaded leaf image (jpg/jpeg/png).

    Returns:
        Dict with disease_class, confidence, expert_advice.
    """
    _validate_image_filename(file.filename)
    temp_path = _save_temp_file(file)
    try:
        result = _vision_rag_pipeline.predict_and_advise(temp_path)
        return result
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
    finally:
        _cleanup_temp_file(temp_path)


@app.post("/full_analysis/")
async def full_analysis(file: UploadFile = File(...)) -> dict[str, Any]:
    """Full pipeline endpoint — used by the Lab 8 dashboard.

    Runs in order:
        1. Lab 5  — ResNet-50 disease detection
        2. Lab 4  — RAG treatment retrieval
        3. Lab 7  — SIR spread simulation (all intervention scenarios)
        4. Lab 9  — DSS scoring, ranking, report assembly

    Args:
        file: Uploaded leaf image (jpg/jpeg/png).

    Returns:
        Complete DSS report dict (see src/DSS/report.py for schema)
        augmented with per-scenario daily_counts for the dashboard chart.
    """
    _validate_image_filename(file.filename)
    temp_path = _save_temp_file(file)

    try:
        # --- Step 1 & 2: Vision model + RAG ---
        vision_result = _vision_rag_pipeline.predict_and_advise(temp_path)
        disease_name = vision_result["disease_class"]
        confidence = vision_result["confidence"]

        # --- Step 3 & 4: Simulation + DSS ---
        dss_report = run_dss(
            disease_name=disease_name,
            model_confidence=confidence,
        )

        # Attach per-scenario daily_counts so the dashboard
        # can render the spread chart without a second API call
        simulation_results = run_simulation(disease_name=disease_name)
        dss_report["scenarios"] = [
            {
                "intervention_name": scenario["intervention_name"],
                "daily_counts":      scenario["daily_counts"],
                "crop_loss_percent": scenario["crop_loss_percent"],
            }
            for scenario in simulation_results["scenarios"]
        ]

        return dss_report

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
    finally:
        _cleanup_temp_file(temp_path)