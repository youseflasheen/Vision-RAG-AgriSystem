"""AgriVision AI — Application entry point.

This module provides a single command to launch the complete system:
    - FastAPI backend on port 8000
    - Streamlit dashboard on port 8501

For development, run each service separately:
    uvicorn api.main:app --reload --port 8000
    streamlit run app/frontend.py
"""

import subprocess
import sys
import os


def main() -> None:
    """Print startup instructions for the AgriVision AI system."""
    print("=" * 60)
    print("  AgriVision AI — Pest Detection & Decision Support System")
    print("=" * 60)
    print()
    print("To start the system, run these in two separate terminals:")
    print()
    print("  Terminal 1 (API backend):")
    print("    uvicorn api.main:app --reload --port 8000")
    print()
    print("  Terminal 2 (Dashboard frontend):")
    print("    streamlit run app/frontend.py")
    print()
    print("Then open http://localhost:8501 in your browser.")
    print()
    print("System Components:")
    print("  Lab 1: Chatbot Q&A         (src/lab1_chatbot/)")
    print("  Lab 2: Prompt Engine       (src/lab2_prompts/)")
    print("  Lab 3: Data Pipeline       (src/lab3_data/)")
    print("  Lab 4: RAG Knowledge       (src/lab4_rag/)")
    print("  Lab 5: Vision Model        (src/lab5_model/)")
    print("  Lab 6: GAN Synthesis       (src/lab6_gan/)")
    print("  Lab 7: Simulation          (src/lab7_simulation/)")
    print("  Lab 8: Dashboard           (app/frontend.py)")
    print("  Lab 9: Decision Support    (src/lab9_dss/)")
    print()


if __name__ == "__main__":
    main()
