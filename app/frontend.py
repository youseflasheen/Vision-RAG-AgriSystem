"""Streamlit dashboard — AgriVision AI visualization layer.

AgriVision AI — complete redesign with tab-based navigation,
professional dark agricultural theme, and healthy-plant awareness.

Run with:
    streamlit run app/frontend.py
(FastAPI backend must be running on port 8000 first)
"""

import streamlit as st
import requests
import plotly.graph_objects as go
from PIL import Image
import pandas as pd
from typing import Optional
import io

# ---------------------------------------------------------------------------
# Page configuration — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Global CSS — full redesign
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
* { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}
.main .block-container {
    padding: 0 2rem 3rem 2rem;
    max-width: 1200px;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"] { display: none; }

/* ── Top navigation bar ── */
.nav-bar {
    display: flex;
    align-items: center;
    gap: 0;
    padding: 0 0 0 0;
    margin: 0 -2rem 2rem -2rem;
    background: #161b22;
    border-bottom: 1px solid #21262d;
    position: sticky;
    top: 0;
    z-index: 100;
}
.nav-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 24px;
    border-right: 1px solid #21262d;
    text-decoration: none;
    white-space: nowrap;
}
.nav-logo-icon { font-size: 22px; }
.nav-logo-text {
    font-size: 15px;
    font-weight: 700;
    color: #58d68d;
    letter-spacing: -0.3px;
}
.nav-tabs {
    display: flex;
    gap: 0;
    flex: 1;
    overflow-x: auto;
    scrollbar-width: none;
}
.nav-tabs::-webkit-scrollbar { display: none; }
.nav-tab {
    padding: 14px 20px;
    font-size: 13px;
    font-weight: 500;
    color: #8b949e;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    white-space: nowrap;
    transition: all 0.15s ease;
    text-decoration: none;
    display: flex;
    align-items: center;
    gap: 7px;
}
.nav-tab:hover { color: #e6edf3; background: #1c2128; }
.nav-tab.active { color: #58d68d; border-bottom-color: #58d68d; }
.nav-status {
    padding: 14px 24px;
    font-size: 12px;
    color: #8b949e;
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
    border-left: 1px solid #21262d;
}
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #e74c3c;
}
.status-dot.online { background: #58d68d; animation: pulse 2s infinite; }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* ── Page header ── */
.page-header {
    margin-bottom: 2rem;
    padding-top: 1.5rem;
}
.page-title {
    font-size: 28px;
    font-weight: 700;
    color: #e6edf3;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.page-subtitle {
    font-size: 14px;
    color: #8b949e;
    margin: 0;
    font-weight: 400;
}

/* ── Cards ── */
.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.card-title {
    font-size: 13px;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.card-title-icon { font-size: 14px; }

/* ── Detection result ── */
.disease-name {
    font-size: 24px;
    font-weight: 700;
    color: #e6edf3;
    margin: 4px 0 16px 0;
    letter-spacing: -0.4px;
}
.confidence-bar-bg {
    height: 6px;
    background: #21262d;
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 6px;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s ease;
}
.confidence-meta {
    font-size: 12px;
    color: #8b949e;
    font-family: 'DM Mono', monospace;
}

/* ── Badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
.badge-high { background: #3d1a1a; color: #f87171; border: 1px solid #7f1d1d; }
.badge-medium { background: #2d2007; color: #fbbf24; border: 1px solid #78350f; }
.badge-low { background: #0d2818; color: #4ade80; border: 1px solid #14532d; }
.badge-none { background: #0d2818; color: #4ade80; border: 1px solid #14532d; }
.badge-healthy { background: #0d2818; color: #4ade80; border: 1px solid #14532d; }

/* ── Urgency banner ── */
.urgency-banner {
    padding: 12px 18px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
}
.urgency-high { background: #1c0a0a; border: 1px solid #7f1d1d; color: #fca5a5; }
.urgency-medium { background: #1c1408; border: 1px solid #78350f; color: #fcd34d; }
.urgency-low { background: #081c10; border: 1px solid #14532d; color: #86efac; }
.urgency-none { background: #081c10; border: 1px solid #14532d; color: #86efac; }

/* ── Recommendation box ── */
.rec-box {
    background: #0d2818;
    border: 1px solid #14532d;
    border-left: 3px solid #4ade80;
    border-radius: 8px;
    padding: 18px 20px;
    margin-top: 16px;
}
.rec-strategy {
    font-size: 18px;
    font-weight: 700;
    color: #4ade80;
    margin-bottom: 8px;
}
.rec-summary { font-size: 13px; color: #a7f3d0; line-height: 1.6; }
.rec-meta {
    font-size: 12px;
    color: #6ee7b7;
    font-family: 'DM Mono', monospace;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #14532d;
}

/* ── Pipeline steps (landing) ── */
.pipeline-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 24px 0;
}
.pipeline-step {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.pipeline-step:hover { border-color: #58d68d; }
.pipeline-icon { font-size: 28px; margin-bottom: 10px; }
.pipeline-label {
    font-size: 13px;
    font-weight: 600;
    color: #e6edf3;
    margin-bottom: 5px;
}
.pipeline-desc { font-size: 11px; color: #8b949e; line-height: 1.5; }

/* ── Module info grid ── */
.lab-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
}
.lab-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 14px 16px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    transition: border-color 0.2s;
}
.lab-card:hover { border-color: #30363d; }
.lab-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #58d68d;
    margin-top: 5px;
    flex-shrink: 0;
}
.lab-name { font-size: 13px; font-weight: 600; color: #e6edf3; margin-bottom: 3px; }
.lab-desc { font-size: 11px; color: #8b949e; line-height: 1.4; }

/* ── Metrics row ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}
.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px 18px;
}
.metric-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 6px; }
.metric-value { font-size: 22px; font-weight: 700; color: #e6edf3; }
.metric-delta { font-size: 11px; color: #4ade80; margin-top: 3px; font-family: 'DM Mono', monospace; }

/* ── Table styling ── */
[data-testid="stDataFrame"] {
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    overflow: hidden;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 10px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #238636 !important;
    color: white !important;
    border: 1px solid #2ea043 !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #2ea043 !important;
    transform: translateY(-1px);
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
}

/* ── Healthy state ── */
.healthy-banner {
    background: #081c10;
    border: 1px solid #14532d;
    border-radius: 10px;
    padding: 24px 28px;
    text-align: center;
    margin: 20px 0;
}
.healthy-icon { font-size: 48px; margin-bottom: 12px; }
.healthy-title { font-size: 22px; font-weight: 700; color: #4ade80; margin-bottom: 8px; }
.healthy-text { font-size: 14px; color: #86efac; line-height: 1.6; }

/* ── GAN section ── */
.gan-arch {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.gan-flow {
    display: flex;
    align-items: center;
    gap: 0;
    margin: 20px 0;
    flex-wrap: wrap;
}
.gan-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
    min-width: 120px;
}
.gan-box-title { font-size: 12px; font-weight: 600; color: #8b949e; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 4px; }
.gan-box-name { font-size: 14px; font-weight: 700; color: #e6edf3; }
.gan-arrow { font-size: 20px; color: #30363d; padding: 0 8px; }
.gan-highlight { border-color: #58d68d; }
.gan-highlight .gan-box-name { color: #58d68d; }

/* ── Spinner override ── */
[data-testid="stSpinner"] > div { border-top-color: #58d68d !important; }

/* ── Divider ── */
hr { border-color: #21262d !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_BASE_URL: str = "http://127.0.0.1:8000"
FULL_ANALYSIS_ENDPOINT: str = f"{API_BASE_URL}/full_analysis/"

TAB_NAMES = [
    ("🔬", "Analyse", "analyse"),
    ("🤖", "Chatbot", "chatbot"),
    ("📊", "Simulation", "simulation"),
    ("🧠", "Decision Support", "dss"),
    ("🎨", "GAN Synthesis", "gan"),
    ("📖", "About", "about"),
]

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "analyse"
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def check_backend() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def run_full_analysis(uploaded_file) -> Optional[dict]:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
    try:
        r = requests.post(FULL_ANALYSIS_ENDPOINT, files=files, timeout=90)
        if r.status_code == 200:
            return r.json()
        st.error(f"Backend returned error {r.status_code}: {r.text[:300]}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the backend. Start it with: `uvicorn api.main:app --port 8000`")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. The model may be loading — please try again.")
        return None


# ---------------------------------------------------------------------------
# Navigation bar
# ---------------------------------------------------------------------------
def render_navbar(backend_online: bool) -> None:
    tab_html = ""
    for icon, label, key in TAB_NAMES:
        active_cls = "active" if st.session_state.active_tab == key else ""
        tab_html += (
            f'<a class="nav-tab {active_cls}" '
            f'onclick="window.location.href=\'?tab={key}\'" '
            f'href="?tab={key}">{icon} {label}</a>'
        )

    dot_cls = "online" if backend_online else ""
    status_text = "API online" if backend_online else "API offline"

    st.markdown(f"""
    <div class="nav-bar">
        <div class="nav-logo">
            <span class="nav-logo-icon">🌿</span>
            <span class="nav-logo-text">AgriVision AI</span>
        </div>
        <div class="nav-tabs">{tab_html}</div>
        <div class="nav-status">
            <div class="status-dot {dot_cls}"></div>
            {status_text}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab: Analyse (main pipeline)
# ---------------------------------------------------------------------------
def render_analyse_tab(backend_online: bool) -> None:
    st.markdown("""
    <div class="page-header">
        <p class="page-title">Crop Disease Analysis</p>
        <p class="page-subtitle">Upload a leaf photo to run the full AI pipeline — detection, expert advice, simulation, and decision support.</p>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 2], gap="large")

    with col_upload:
        st.markdown('<div class="card-title"><span class="card-title-icon">🖼️</span>Leaf Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded:
            st.image(Image.open(uploaded), use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)

            if not backend_online:
                st.warning("Backend is offline. Start it with:\n```\nuvicorn api.main:app --port 8000\n```")
            else:
                if st.button("🚀 Run Full Analysis", use_container_width=True):
                    with st.spinner("Running AI pipeline..."):
                        result = run_full_analysis(uploaded)
                    if result:
                        st.session_state.analysis_result = result
                        st.success("Analysis complete!")
        else:
            st.markdown("""
            <div style="text-align:center;padding:40px 20px;color:#8b949e;">
                <div style="font-size:36px;margin-bottom:12px;">🍃</div>
                <div style="font-size:13px;">Drop a JPG or PNG of a crop leaf</div>
                <div style="font-size:11px;margin-top:6px;color:#484f58;">Supports tomato, corn, potato, apple</div>
            </div>
            """, unsafe_allow_html=True)

    with col_result:
        result = st.session_state.analysis_result
        if result is None:
            st.markdown("""
            <div class="pipeline-grid">
                <div class="pipeline-step">
                    <div class="pipeline-icon">🖼️</div>
                    <div class="pipeline-label">Upload</div>
                    <div class="pipeline-desc">Photo of a crop leaf from the field</div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-icon">🔬</div>
                    <div class="pipeline-label">Detect</div>
                    <div class="pipeline-desc">ResNet-50 identifies the disease</div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-icon">📚</div>
                    <div class="pipeline-label">Advise</div>
                    <div class="pipeline-desc">RAG retrieves treatment protocol</div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-icon">📊</div>
                    <div class="pipeline-label">Decide</div>
                    <div class="pipeline-desc">Simulation + DSS ranks interventions</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            return

        is_healthy = result.get("is_healthy", False)
        vision = result.get("vision_detection", {})
        rag = result.get("rag_treatment_protocol", {})
        recommendation = result.get("recommendation", {})
        urgency = result.get("urgency_level", "LOW")

        # ── Disease / healthy result ──
        disease_raw = vision.get("predicted_class", "Unknown")
        disease_display = disease_raw.replace("___", " — ").replace("_", " ").title()
        confidence = vision.get("confidence", 0)

        if is_healthy:
            st.markdown(f"""
            <div class="healthy-banner">
                <div class="healthy-icon">✅</div>
                <div class="healthy-title">{disease_display}</div>
                <div class="healthy-text">No disease detected. Your crop appears healthy.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            urgency_lower = urgency.lower()
            badge_cls = f"badge-{urgency_lower}"
            bar_colour = "#f87171" if urgency == "HIGH" else "#fbbf24" if urgency == "MEDIUM" else "#4ade80"

            st.markdown(f"""
            <div class="card">
                <div class="card-title"><span class="card-title-icon">🦠</span>Detection Result</div>
                <div class="disease-name">{disease_display}</div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width:{confidence}%;background:{bar_colour};"></div>
                </div>
                <div class="confidence-meta">Confidence: {confidence:.1f}% · {vision.get('confidence_label', '')} · <span class="badge {badge_cls}">{urgency} urgency</span></div>
            </div>
            """, unsafe_allow_html=True)

        # ── Treatment protocol ──
        with st.expander("📋 Expert Treatment Protocol", expanded=True):
            st.markdown(rag.get("retrieved_context", "No context available."))

        # ── Recommendation ──
        if not is_healthy:
            best = recommendation.get("best_strategy", "—")
            summary = recommendation.get("action_summary", "")
            loss = recommendation.get("projected_loss")
            loss_text = f"Projected crop loss: {loss:.1f}%" if loss is not None else ""

            st.markdown(f"""
            <div class="rec-box">
                <div class="rec-strategy">✅ {best}</div>
                <div class="rec-summary">{summary}</div>
                <div class="rec-meta">{loss_text}</div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("🔍 Raw JSON response"):
            st.json(result)


# ---------------------------------------------------------------------------
# Tab: Chatbot
# ---------------------------------------------------------------------------
def render_chatbot_tab(backend_online: bool) -> None:
    st.markdown("""
    <div class="page-header">
        <p class="page-title">AgriBot — Chatbot Q&A</p>
        <p class="page-subtitle">Ask questions about crop diseases, treatments, and prevention strategies. Powered by RAG + LLM.</p>
    </div>
    """, unsafe_allow_html=True)

    if not backend_online:
        st.warning("Backend must be running to use the chatbot.")
        return

    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-end;margin-bottom:12px;">
                <div style="background:#21262d;border-radius:10px 10px 2px 10px;padding:10px 16px;
                            max-width:70%;font-size:14px;color:#e6edf3;">{content}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-start;margin-bottom:12px;gap:10px;">
                <div style="font-size:22px;margin-top:2px;">🌿</div>
                <div style="background:#161b22;border:1px solid #21262d;border-radius:2px 10px 10px 10px;
                            padding:10px 16px;max-width:75%;font-size:14px;color:#e6edf3;line-height:1.6;">{content}</div>
            </div>
            """, unsafe_allow_html=True)

    user_input = st.chat_input("Ask about a crop disease, treatment, or prevention strategy...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        try:
            response = requests.post(
                f"{API_BASE_URL}/chat/",
                json={"question": user_input,
                      "history": st.session_state.chat_history[:-1]},
                timeout=30,
            )
            if response.status_code == 200:
                answer = response.json().get("answer", "No response.")
            else:
                answer = f"Sorry, I couldn't process that (error {response.status_code})."
        except Exception:
            answer = (
                "⚠️ Chat endpoint not available. "
                "Make sure `/chat/` is added to `api/main.py` and the backend is running."
            )
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()


# ---------------------------------------------------------------------------
# Tab: Simulation
# ---------------------------------------------------------------------------
def render_simulation_tab() -> None:
    st.markdown("""
    <div class="page-header">
        <p class="page-title">Spread Simulation</p>
        <p class="page-subtitle">SIR model simulates how the detected disease spreads across a 20×20 farm grid under different intervention strategies.</p>
    </div>
    """, unsafe_allow_html=True)

    result = st.session_state.analysis_result
    if result is None or result.get("is_healthy"):
        st.info("Run an analysis first from the **Analyse** tab to see simulation results.")
        return

    sim_summary = result.get("simulation_summary", {})
    scenarios = result.get("scenarios", [])

    if sim_summary:
        grid = sim_summary.get("grid_size", [20, 20])
        days = sim_summary.get("days_simulated", 30)
        saving = sim_summary.get("crop_loss_saving")
        baseline = sim_summary.get("baseline_crop_loss")
        best_loss = sim_summary.get("best_crop_loss")

        st.markdown(f"""
        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-label">Grid Size</div>
                <div class="metric-value">{grid[0]}×{grid[1]}</div>
                <div class="metric-delta">{grid[0] * grid[1]} total cells</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Days Simulated</div>
                <div class="metric-value">{days}</div>
                <div class="metric-delta">Daily spread tracking</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Crop Loss Saving</div>
                <div class="metric-value" style="color:#4ade80;">{saving:.1f}%</div>
                <div class="metric-delta">vs. no intervention ({baseline:.1f}% → {best_loss:.1f}%)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if scenarios:
        fig = go.Figure()
        palette = ["#f87171", "#60a5fa", "#4ade80", "#fbbf24"]

        for idx, scenario in enumerate(scenarios):
            daily = scenario.get("daily_counts", [])
            if not daily:
                continue
            days_x = [d["day"] for d in daily]
            infected_y = [d["infected"] for d in daily]
            colour = palette[idx % len(palette)]
            fig.add_trace(go.Scatter(
                x=days_x, y=infected_y,
                mode="lines",
                name=scenario["intervention_name"],
                line=dict(color=colour, width=2.5),
                fill="tozeroy",
                fillcolor=colour.replace(")", ",0.05)").replace("rgb", "rgba") if "rgb" in colour else colour + "0d",
            ))

        fig.update_layout(
            title=dict(text="Infected Cells Over Time by Strategy", font=dict(size=14, color="#8b949e")),
            xaxis=dict(title="Day", color="#8b949e", gridcolor="#21262d", zeroline=False),
            yaxis=dict(title="Infected Cells", color="#8b949e", gridcolor="#21262d", zeroline=False),
            legend=dict(font=dict(color="#8b949e"), bgcolor="rgba(0,0,0,0)"),
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            height=380,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: Decision Support
# ---------------------------------------------------------------------------
def render_dss_tab() -> None:
    st.markdown("""
    <div class="page-header">
        <p class="page-title">Decision Support System</p>
        <p class="page-subtitle">Ranks intervention strategies by combining simulation results, model confidence, and RAG relevance into a single actionable score.</p>
    </div>
    """, unsafe_allow_html=True)

    result = st.session_state.analysis_result
    if result is None or result.get("is_healthy"):
        st.info("Run an analysis first from the **Analyse** tab to see DSS results.")
        return

    urgency = result.get("urgency_level", "LOW")
    ranked = result.get("ranked_interventions", [])
    recommendation = result.get("recommendation", {})

    urgency_cls = f"urgency-{urgency.lower()}"
    urgency_icon = "🔴" if urgency == "HIGH" else "🟡" if urgency == "MEDIUM" else "🟢"
    st.markdown(f"""
    <div class="urgency-banner {urgency_cls}">
        {urgency_icon} Urgency Level: <strong>{urgency}</strong>
    </div>
    """, unsafe_allow_html=True)

    if ranked:
        df = pd.DataFrame([{
            "Rank": f"#{r['rank']}",
            "Strategy": r["intervention_name"],
            "Crop Loss": f"{r['crop_loss_percent']:.1f}%",
            "Score": f"{r['final_score']:.3f}",
        } for r in ranked])
        st.dataframe(df, use_container_width=True, hide_index=True)

    best = recommendation.get("best_strategy", "—")
    summary = recommendation.get("action_summary", "")
    loss = recommendation.get("projected_loss")
    saving = recommendation.get("saving_vs_baseline")

    loss_text = f"Projected crop loss: {loss:.1f}%" if loss is not None else ""
    saving_text = f" · Saving vs no action: {saving:.1f}%" if saving is not None else ""

    st.markdown(f"""
    <div class="rec-box">
        <div class="rec-strategy">✅ {best}</div>
        <div class="rec-summary">{summary}</div>
        <div class="rec-meta">{loss_text}{saving_text}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab: GAN Synthesis
# ---------------------------------------------------------------------------
def render_gan_tab() -> None:
    st.markdown("""
    <div class="page-header">
        <p class="page-title">GAN Synthesis</p>
        <p class="page-subtitle">A Generative Adversarial Network trained on PlantVillage data produces synthetic leaf images to augment the training dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="gan-arch">
        <div class="card-title"><span class="card-title-icon">🏗️</span>Architecture</div>
        <div class="gan-flow">
            <div class="gan-box">
                <div class="gan-box-title">Input</div>
                <div class="gan-box-name">Noise Vector z</div>
            </div>
            <div class="gan-arrow">→</div>
            <div class="gan-box gan-highlight">
                <div class="gan-box-title">Generator G</div>
                <div class="gan-box-name">ConvTranspose2D</div>
            </div>
            <div class="gan-arrow">→</div>
            <div class="gan-box">
                <div class="gan-box-title">Output</div>
                <div class="gan-box-name">Synthetic Image</div>
            </div>
            <div class="gan-arrow">→</div>
            <div class="gan-box gan-highlight">
                <div class="gan-box-title">Discriminator D</div>
                <div class="gan-box-name">Conv2D Classifier</div>
            </div>
            <div class="gan-arrow">→</div>
            <div class="gan-box">
                <div class="gan-box-title">Output</div>
                <div class="gan-box-name">Real / Fake</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title"><span>🎯</span>Purpose</div>
            <p style="font-size:13px;color:#8b949e;line-height:1.7;margin:0;">
            The training dataset has <strong style="color:#e6edf3;">class imbalance</strong> —
            some diseases have only 200–400 images while others have 2,000+.
            The GAN addresses this by generating realistic synthetic leaf images
            for underrepresented classes, improving the ResNet-50 model's
            accuracy on rare diseases.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="card-title"><span>📁</span>Files</div>
            <div style="font-family:'DM Mono',monospace;font-size:12px;color:#8b949e;line-height:2;">
                src/Gan/generator.py<br>
                src/Gan/discriminator.py<br>
                src/Gan/trainer.py<br>
                src/Gan/generate.py
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title"><span>⚙️</span>Training Details</div>
            <table style="width:100%;font-size:13px;border-collapse:collapse;">
                <tr style="border-bottom:1px solid #21262d;">
                    <td style="padding:8px 0;color:#8b949e;">Architecture</td>
                    <td style="padding:8px 0;color:#e6edf3;text-align:right;">DCGAN</td>
                </tr>
                <tr style="border-bottom:1px solid #21262d;">
                    <td style="padding:8px 0;color:#8b949e;">Input noise dim</td>
                    <td style="padding:8px 0;color:#e6edf3;text-align:right;">100</td>
                </tr>
                <tr style="border-bottom:1px solid #21262d;">
                    <td style="padding:8px 0;color:#8b949e;">Output resolution</td>
                    <td style="padding:8px 0;color:#e6edf3;text-align:right;">64 × 64 px</td>
                </tr>
                <tr style="border-bottom:1px solid #21262d;">
                    <td style="padding:8px 0;color:#8b949e;">Optimizer</td>
                    <td style="padding:8px 0;color:#e6edf3;text-align:right;">Adam (β₁=0.5)</td>
                </tr>
                <tr>
                    <td style="padding:8px 0;color:#8b949e;">Loss function</td>
                    <td style="padding:8px 0;color:#e6edf3;text-align:right;">Binary Cross-Entropy</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="card-title"><span>🔄</span>Role in Pipeline</div>
            <p style="font-size:13px;color:#8b949e;line-height:1.7;margin:0;">
            After training, <code style="background:#21262d;padding:2px 6px;border-radius:4px;">generate.py</code>
            produces a batch of synthetic images saved to <code style="background:#21262d;padding:2px 6px;border-radius:4px;">data/synthetic/</code>.
            These are merged with the real PlantVillage + PlantDoc hybrid dataset
            before the next ResNet-50 training cycle.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#1c1408;border:1px solid #78350f;border-radius:8px;padding:14px 18px;
                font-size:13px;color:#fcd34d;margin-top:8px;">
        ⚠️ GAN training requires GPU and the full dataset (~30k images). The trained checkpoint
        is not included in this repository due to file size. Run <code>python src/Gan/trainer.py</code>
        with the processed dataset to produce a checkpoint, then use <code>generate.py</code> to sample images.
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab: About
# ---------------------------------------------------------------------------
def render_about_tab() -> None:
    st.markdown("""
    <div class="page-header">
        <p class="page-title">About This System</p>
        <p class="page-subtitle">A full-stack AI pipeline for agricultural pest detection and decision support.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="lab-grid">
        <div class="lab-card">
            <div class="lab-dot"></div>
            <div>
                <div class="lab-name">Chatbot Q&A</div>
                <div class="lab-desc">Stateful farmer Q&A chatbot grounded in the RAG knowledge base using Groq LLM.</div>
            </div>
        </div>
        <div class="lab-card">
            <div class="lab-dot"></div>
            <div>
                <div class="lab-name">Prompt Engine</div>
                <div class="lab-desc">Centralised prompt template library. All LLM prompts defined once and reused across modules.</div>
            </div>
        </div>
        <div class="lab-card">
            <div class="lab-dot"></div>
            <div>
                <div class="lab-name">Data Pipeline</div>
                <div class="lab-desc">PlantVillage + PlantDoc hybrid dataset — 21 classes, 80/10/10 train/val/test split.</div>
            </div>
        </div>
        <div class="lab-card">
            <div class="lab-dot"></div>
            <div>
                <div class="lab-name">RAG Knowledge</div>
                <div class="lab-desc">ChromaDB vector store with HuggingFace embeddings. Retrieves treatment protocols by cosine similarity.</div>
            </div>
        </div>
        <div class="lab-card">
            <div class="lab-dot"></div>
            <div>
                <div class="lab-name">Vision Model</div>
                <div class="lab-desc">ResNet-50 fine-tuned on the hybrid dataset. Classifies 21 crop disease categories with softmax confidence.</div>
            </div>
        </div>
        <div class="lab-card">
            <div class="lab-dot"></div>
            <div>
                <div class="lab-name">GAN Synthesis</div>
                <div class="lab-desc">DCGAN generates synthetic leaf images to augment underrepresented disease classes in training.</div>
            </div>
        </div>
        <div class="lab-card">
            <div class="lab-dot"></div>
            <div>
                <div class="lab-name">Simulation</div>
                <div class="lab-desc">SIR spread model on a 20×20 farm grid. Compares disease progression under three intervention strategies.</div>
            </div>
        </div>
        <div class="lab-card">
            <div class="lab-dot"></div>
            <div>
                <div class="lab-name">Dashboard</div>
                <div class="lab-desc">This Streamlit interface. Integrates all module outputs into one end-to-end user workflow.</div>
            </div>
        </div>
        <div class="lab-card">
            <div class="lab-dot"></div>
            <div>
                <div class="lab-name">Decision Support</div>
                <div class="lab-desc">Scores and ranks interventions using simulation results, model confidence, and RAG relevance.</div>
            </div>
        </div>
    </div>

    <br>
    <div class="card" style="margin-top:16px;">
        <div class="card-title"><span>🛠️</span>Technology Stack</div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;font-size:13px;">
            <div>
                <div style="color:#8b949e;margin-bottom:8px;font-weight:600;">ML / Vision</div>
                <div style="color:#e6edf3;line-height:2;">PyTorch · ResNet-50 · torchvision · DCGAN</div>
            </div>
            <div>
                <div style="color:#8b949e;margin-bottom:8px;font-weight:600;">NLP / RAG</div>
                <div style="color:#e6edf3;line-height:2;">LangChain · ChromaDB · HuggingFace · Groq LLM</div>
            </div>
            <div>
                <div style="color:#8b949e;margin-bottom:8px;font-weight:600;">Backend / UI</div>
                <div style="color:#e6edf3;line-height:2;">FastAPI · Streamlit · Plotly · Pandas</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab routing — read from URL query params
# ---------------------------------------------------------------------------
def get_active_tab() -> str:
    params = st.query_params
    tab = params.get("tab", "analyse")
    valid_tabs = {k for _, _, k in TAB_NAMES}
    return tab if tab in valid_tabs else "analyse"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    backend_online = check_backend()
    active = get_active_tab()
    st.session_state.active_tab = active

    render_navbar(backend_online)

    if active == "analyse":
        render_analyse_tab(backend_online)
    elif active == "chatbot":
        render_chatbot_tab(backend_online)
    elif active == "simulation":
        render_simulation_tab()
    elif active == "dss":
        render_dss_tab()
    elif active == "gan":
        render_gan_tab()
    elif active == "about":
        render_about_tab()


if __name__ == "__main__":
    main()