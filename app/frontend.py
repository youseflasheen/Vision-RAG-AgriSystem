"""Streamlit dashboard — Lab 8 visualization layer.

Provides the end-to-end user interface for the pest detection system.
Communicates with the FastAPI backend (api/main.py) over HTTP so that
the UI layer remains decoupled from the ML and simulation modules.

Run with:
    streamlit run app/frontend.py
(FastAPI backend must be running on port 8000 first)
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import pandas as pd
from typing import Optional

# ---------------------------------------------------------------------------
# Page configuration — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriVision AI — Pest Detection System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_BASE_URL: str = "http://127.0.0.1:8000"
DIAGNOSE_ENDPOINT: str = f"{API_BASE_URL}/diagnose/"
FULL_ANALYSIS_ENDPOINT: str = f"{API_BASE_URL}/full_analysis/"

URGENCY_COLOURS: dict[str, str] = {
    "HIGH":   "#e74c3c",
    "MEDIUM": "#f39c12",
    "LOW":    "#2ecc71",
}


# ---------------------------------------------------------------------------
# Helper: check if backend is reachable
# ---------------------------------------------------------------------------
def check_backend_status() -> bool:
    """Return True if the FastAPI backend is responding."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=2)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


# ---------------------------------------------------------------------------
# Helper: call the full analysis endpoint
# ---------------------------------------------------------------------------
def run_full_analysis(uploaded_file) -> Optional[dict]:
    """POST image to /full_analysis/ and return the response dict.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Response dict or None if the request failed.
    """
    files = {
        "file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")
    }
    try:
        response = requests.post(FULL_ANALYSIS_ENDPOINT, files=files, timeout=60)
        if response.status_code == 200:
            return response.json()
        st.error(f"Backend error {response.status_code}: {response.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the backend. Make sure FastAPI is running on port 8000.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. The model may still be loading — try again.")
        return None


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------

def render_detection_panel(vision_detection: dict) -> None:
    """Render the Lab 5 vision model output panel.

    Args:
        vision_detection: Dict with keys predicted_class, confidence,
                          confidence_label.
    """
    st.subheader("🦠 Disease Detection — Lab 5")

    disease_display = vision_detection["predicted_class"].replace("___", " — ").replace("_", " ")
    confidence_pct = vision_detection["confidence"]
    confidence_label = vision_detection["confidence_label"]

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown(f"### {disease_display}")
        st.progress(confidence_pct / 100)
        st.caption(f"Confidence: **{confidence_pct:.1f}%** ({confidence_label})")

    with col_right:
        colour = (
            "#2ecc71" if confidence_pct >= 75
            else "#f39c12" if confidence_pct >= 50
            else "#e74c3c"
        )
        st.markdown(
            f"<div style='background:{colour};padding:16px;border-radius:8px;"
            f"text-align:center;color:white;font-size:1.4em;font-weight:bold'>"
            f"{confidence_pct:.1f}%</div>",
            unsafe_allow_html=True,
        )


def render_expert_advice_panel(rag_treatment: dict) -> None:
    """Render the Lab 4 RAG + Lab 1 chatbot expert advice panel.

    Args:
        rag_treatment: Dict with keys retrieved_context, relevance_score.
    """
    st.subheader("📋 Expert Treatment Protocol — Labs 1, 2, 4")

    relevance = rag_treatment.get("relevance_score", 0)
    st.caption(f"Knowledge base relevance score: {relevance:.2f}")

    with st.expander("View full treatment protocol", expanded=True):
        st.markdown(rag_treatment.get("retrieved_context", "No context retrieved."))


def render_simulation_panel(simulation_summary: dict, full_result: dict) -> None:
    """Render the Lab 7 simulation spread chart and summary.

    Args:
        simulation_summary: High-level summary dict from DSS report.
        full_result:        Full API response (contains scenario daily_counts).
    """
    st.subheader("📈 Spread Simulation — Lab 7")

    col1, col2, col3 = st.columns(3)
    col1.metric("Grid Size", f"{simulation_summary['grid_size'][0]}×{simulation_summary['grid_size'][1]}")
    col2.metric("Days Simulated", simulation_summary["days_simulated"])

    baseline = simulation_summary.get("baseline_crop_loss")
    best = simulation_summary.get("best_crop_loss")
    saving = simulation_summary.get("crop_loss_saving")

    col3.metric(
        "Crop Loss Saving",
        f"{saving:.1f}%" if saving is not None else "N/A",
        delta=f"-{saving:.1f}% vs no action" if saving else None,
    )

    # Build spread chart from scenario data if available
    scenarios = full_result.get("scenarios", [])
    if scenarios:
        fig = go.Figure()
        colours = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

        for index, scenario in enumerate(scenarios):
            daily_counts = scenario.get("daily_counts", [])
            if not daily_counts:
                continue
            days = [entry["day"] for entry in daily_counts]
            infected = [entry["infected"] for entry in daily_counts]
            colour = colours[index % len(colours)]

            fig.add_trace(go.Scatter(
                x=days,
                y=infected,
                mode="lines",
                name=scenario["intervention_name"],
                line=dict(color=colour, width=2),
            ))

        fig.update_layout(
            title="Infected Cells Over Time by Intervention Strategy",
            xaxis_title="Day",
            yaxis_title="Infected Cells",
            legend_title="Strategy",
            height=350,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Simulation chart unavailable — daily counts not returned by API.")


def render_dss_panel(
    ranked_interventions: list,
    recommendation: dict,
    urgency_level: str,
) -> None:
    """Render the Lab 9 DSS ranked interventions and final recommendation.

    Args:
        ranked_interventions: List of scored intervention dicts.
        recommendation:       Top-line recommendation dict.
        urgency_level:        'HIGH', 'MEDIUM', or 'LOW'.
    """
    st.subheader("🧠 Decision Support System — Lab 9")

    # Urgency banner
    urgency_colour = URGENCY_COLOURS.get(urgency_level, "#95a5a6")
    st.markdown(
        f"<div style='background:{urgency_colour};padding:12px;border-radius:8px;"
        f"color:white;font-weight:bold;font-size:1.1em;margin-bottom:12px'>"
        f"⚠️ Urgency Level: {urgency_level}</div>",
        unsafe_allow_html=True,
    )

    # Ranked interventions table
    if ranked_interventions:
        table_data = pd.DataFrame([
            {
                "Rank":              item["rank"],
                "Strategy":          item["intervention_name"],
                "Crop Loss (%)":     item["crop_loss_percent"],
                "Score":             item["final_score"],
            }
            for item in ranked_interventions
        ])
        st.dataframe(table_data, use_container_width=True, hide_index=True)

    # Final recommendation box
    best_strategy = recommendation.get("best_strategy", "Unknown")
    action_summary = recommendation.get("action_summary", "")
    projected_loss = recommendation.get("projected_loss")

    st.markdown("### ✅ Recommended Action")
    st.markdown(
        f"<div style='background:#1a1a2e;padding:16px;border-radius:8px;"
        f"border-left:4px solid #2ecc71;color:white'>"
        f"<strong>{best_strategy}</strong><br><br>"
        f"{action_summary}<br><br>"
        f"<em>Projected crop loss: {projected_loss:.1f}%</em>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> Optional[object]:
    """Render the sidebar with upload widget and system status.

    Returns:
        Streamlit UploadedFile or None.
    """
    with st.sidebar:
        st.image(
            "https://img.icons8.com/color/96/000000/leaf.png",
            width=60,
        )
        st.title("AgriVision AI")
        st.caption("Pest Detection & Decision Support System")
        st.divider()

        backend_ok = check_backend_status()
        status_icon = "🟢" if backend_ok else "🔴"
        st.markdown(f"**Backend:** {status_icon} {'Online' if backend_ok else 'Offline'}")

        st.divider()
        st.markdown("**System Components**")
        labs = [
            ("Lab 1", "Chatbot Q&A"),
            ("Lab 2", "Prompt Engine"),
            ("Lab 3", "Data Pipeline"),
            ("Lab 4", "RAG Knowledge"),
            ("Lab 5", "Vision Model"),
            ("Lab 6", "GAN Synthesis"),
            ("Lab 7", "Simulation"),
            ("Lab 8", "Dashboard"),
            ("Lab 9", "DSS Engine"),
        ]
        for lab_id, lab_name in labs:
            st.markdown(f"✅ {lab_id} — {lab_name}")

        st.divider()
        uploaded_file = st.file_uploader(
            "Upload a leaf image",
            type=["jpg", "jpeg", "png"],
            help="Upload a photo of a diseased crop leaf for analysis.",
        )

        if uploaded_file:
            st.image(Image.open(uploaded_file), caption="Uploaded image", use_column_width=True)

    return uploaded_file


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the Streamlit dashboard."""
    st.title("🌿 AgriVision AI — Pest Detection & Decision Support")
    st.caption(
        "Upload a leaf image to run the full pipeline: "
        "Vision Detection → RAG Expert Advice → Spread Simulation → DSS Recommendation"
    )

    uploaded_file = render_sidebar()

    if uploaded_file is None:
        # Landing state — show pipeline overview
        st.info("👈 Upload a leaf image in the sidebar to begin analysis.")

        st.markdown("### How it works")
        cols = st.columns(4)
        steps = [
            ("🖼️", "1. Upload", "Photo of diseased crop leaf"),
            ("🔬", "2. Detect", "ResNet-50 identifies the disease (Lab 5)"),
            ("📚", "3. Advise", "RAG retrieves treatment protocol (Labs 1, 2, 4)"),
            ("📊", "4. Decide", "Simulation + DSS ranks interventions (Labs 7, 9)"),
        ]
        for col, (icon, title, desc) in zip(cols, steps):
            with col:
                st.markdown(
                    f"<div style='text-align:center;padding:16px;background:#1e1e2e;"
                    f"border-radius:8px;height:120px'>"
                    f"<div style='font-size:2em'>{icon}</div>"
                    f"<strong>{title}</strong><br>"
                    f"<small>{desc}</small></div>",
                    unsafe_allow_html=True,
                )
        return

    # Analyse button
    if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):
        with st.spinner("Running full pipeline — this may take 20-30 seconds..."):
            result = run_full_analysis(uploaded_file)

        if result is None:
            return

        st.success("Analysis complete!")
        st.divider()

        # Pull sections from the DSS report
        vision = result.get("vision_detection", {})
        rag_treatment = result.get("rag_treatment_protocol", {})
        simulation_summary = result.get("simulation_summary", {})
        ranked = result.get("ranked_interventions", [])
        recommendation = result.get("recommendation", {})
        urgency = result.get("urgency_level", "LOW")

        # Render all panels
        render_detection_panel(vision)
        st.divider()

        col_advice, col_image = st.columns([3, 1])
        with col_advice:
            render_expert_advice_panel(rag_treatment)
        with col_image:
            st.image(Image.open(uploaded_file), caption="Analysed image", use_column_width=True)

        st.divider()
        render_simulation_panel(simulation_summary, result)
        st.divider()
        render_dss_panel(ranked, recommendation, urgency)

        # Raw JSON expander for debugging / presentation
        with st.expander("🔍 Raw API Response (JSON)"):
            st.json(result)


if __name__ == "__main__":
    main()