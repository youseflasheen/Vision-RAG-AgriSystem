"""FastAPI backend — main API entry point.

Exposes endpoints:
    GET  /               — health check
    POST /diagnose/      — vision + RAG only (original endpoint)
    POST /full_analysis/  — full pipeline: vision + RAG + simulation + DSS
    POST /chat/          — chatbot Q&A

Run with:
    uvicorn api.main:app --reload --port 8000
"""

import os
import sys
import shutil
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Path setup — ensures src/ package is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.lab5_model.inference import AgriVisionRAG
from src.lab4_rag.generator import RAGGenerator
from src.lab7_simulation.simulator import run_simulation
from src.lab9_dss.dss_engine import run_dss
from src.lab1_chatbot.chatbot import AgriculturalChatbot

# ---------------------------------------------------------------------------
# Globals — initialised once at startup to avoid reloading models per request
# ---------------------------------------------------------------------------
_vision_rag_pipeline: AgriVisionRAG = None
_rag_generator: RAGGenerator = None
_chatbot: AgriculturalChatbot = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load all models and vector DB into memory at server start."""
    global _vision_rag_pipeline, _rag_generator, _chatbot
    print("[API] Loading vision model and RAG pipeline...")
    _vision_rag_pipeline = AgriVisionRAG()
    _rag_generator = RAGGenerator()
    _chatbot = AgriculturalChatbot()
    print("[API] Ready.")
    yield
    print("[API] Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AgriVision AI — Pest Detection API",
    description="End-to-end pest detection, RAG advice, simulation and DSS.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow Streamlit frontend on any port during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    """Full pipeline endpoint — vision, RAG, simulation and DSS.

    Args:
        file: Uploaded leaf image (jpg/jpeg/png).

    Returns:
        Complete DSS report with all lab outputs.
    """
    _validate_image_filename(file.filename)
    temp_path = _save_temp_file(file)

    try:
        # --- Step 1 & 2: Vision model + RAG ---
        vision_result = _vision_rag_pipeline.predict_and_advise(temp_path)
        disease_name = vision_result["disease_class"]
        confidence = vision_result["confidence"]

        # --- Step 3 & 4: DSS (includes simulation, scoring, and report) ---
        dss_report = run_dss(
            disease_name=disease_name,
            model_confidence=confidence,
        )

        return dss_report

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
    finally:
        _cleanup_temp_file(temp_path)


@app.post("/chat/")
async def chat(payload: dict) -> dict:
    """Chatbot Q&A endpoint.

    Args:
        payload: Dict with 'question' key.

    Returns:
        Dict with 'answer' key containing the chatbot response.
    """
    question = payload.get("question", "")
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        answer = _chatbot.ask(question)
        return {"answer": answer}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/system_info/")
async def system_info() -> dict[str, Any]:
    """Return system component status for the About page.

    Returns:
        Dict with component names and their status.
    """
    return {
        "system": "AgriVision AI",
        "version": "2.0.0",
        "components": {
            "lab1_chatbot": {"status": "active", "description": "Domain Q&A chatbot with Groq LLM"},
            "lab2_prompts": {"status": "active", "description": "Centralized prompt template library"},
            "lab3_data": {"status": "active", "description": "PlantVillage hybrid dataset pipeline"},
            "lab4_rag": {"status": "active", "description": "ChromaDB + HuggingFace RAG system"},
            "lab5_model": {"status": "active", "description": "ResNet-50 disease classifier"},
            "lab6_gan": {"status": "available", "description": "DCGAN synthetic data generator"},
            "lab7_simulation": {"status": "active", "description": "SIR pest spread simulation"},
            "lab8_dashboard": {"status": "active", "description": "Streamlit visualization UI"},
            "lab9_dss": {"status": "active", "description": "Multi-criteria decision support"},
        },
    }