"""Streamlit dashboard — AgriVision AI visualization layer.

AgriVision AI — The Ultimate Premium UI.
Pushing Streamlit to its absolute visual limits using advanced CSS injection.

Run with:
    streamlit run app/frontend.py
"""

import streamlit as st
import requests
import plotly.graph_objects as go
from PIL import Image
import pandas as pd
from typing import Optional

# ---------------------------------------------------------------------------
# Page configuration MUST be the first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# ULTIMATE PREMIUM CSS INJECTION
# ---------------------------------------------------------------------------
# This block completely hijacks Streamlit's native styling to create
# a breathtaking, ultra-modern SaaS UI (Apple/Vercel aesthetic).
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* 1. Global Typography & Background */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');

html, body {
    font-family: 'Inter', sans-serif !important;
}
/* Apply font to elements but explicitly exclude Streamlit icons to prevent text overlap */
div[class*="st-"]:not([class*="stIcon"]):not([class*="material"]) {
    font-family: 'Inter', sans-serif;
}

/* Deep space gradient background */
.stApp {
    background: radial-gradient(circle at top center, #18181b 0%, #09090b 100%) !important;
    background-attachment: fixed !important;
}

/* Hide standard Streamlit chrome */
header[data-testid="stHeader"] { background: transparent !important; }
footer { display: none !important; }

/* 2. Stunning Tabs (Segmented Control style) */
[data-testid="stTabs"] {
    margin-top: 1rem;
}
[data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border-radius: 100px !important;
    padding: 6px !important;
    gap: 8px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5) !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: 100px !important;
    color: #a1a1aa !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
[data-baseweb="tab"]:hover {
    color: #f4f4f5 !important;
    background: rgba(255, 255, 255, 0.05) !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(59, 130, 246, 0.15) 100%) !important;
    color: #10b981 !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.2) !important;
}
/* Hide the ugly default active indicator line */
[data-baseweb="tab-highlight"] { display: none !important; }

/* 3. Glassmorphism Metrics (Cards) */
[data-testid="stMetric"] {
    background: rgba(24, 24, 27, 0.4) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 20px !important;
    padding: 24px !important;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5) !important;
    transition: all 0.4s ease !important;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-5px) !important;
    border: 1px solid rgba(16, 185, 129, 0.5) !important;
    box-shadow: 0 20px 40px -10px rgba(16, 185, 129, 0.15) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #f8fafc 0%, #94a3b8 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.9rem !important;
    color: #a1a1aa !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 8px !important;
}

/* 4. Glow Buttons */
button[kind="primary"] {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.3), 0 4px 6px -1px rgba(0,0,0,0.2) !important;
    transition: all 0.3s ease !important;
}
button[kind="primary"]:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 0 30px rgba(16, 185, 129, 0.6), 0 8px 15px -3px rgba(0,0,0,0.3) !important;
}

/* 5. Typography Enhancements */
h1 {
    font-weight: 800 !important;
    letter-spacing: -0.04em !important;
    background: linear-gradient(to right, #ffffff, #a1a1aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem !important;
}
h2, h3 {
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    color: #f4f4f5 !important;
}
p, li {
    color: #e4e4e7 !important;
    line-height: 1.6 !important;
}

/* 6. File Uploader Styling */
[data-testid="stFileUploadDropzone"] {
    background: rgba(24, 24, 27, 0.3) !important;
    border: 2px dashed rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border: 2px dashed #10b981 !important;
    background: rgba(16, 185, 129, 0.05) !important;
}

/* 7. Chat Bubbles */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
}
[data-testid="stChatMessage"][data-author="user"] {
    background: rgba(59, 130, 246, 0.05) !important;
    border: 1px solid rgba(59, 130, 246, 0.2) !important;
}

/* 8. Expander & Dataframe */
[data-testid="stExpander"] {
    background: rgba(24, 24, 27, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
}
[data-testid="stDataFrame"] {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

/* Pulse Animation for status */
@keyframes pulse {
    0% { opacity: 0.5; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { opacity: 1; box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
    100% { opacity: 0.5; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}
.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    background: #10b981;
    border-radius: 50%;
    animation: pulse 2s infinite;
    margin-right: 8px;
    vertical-align: middle;
}
.status-offline {
    background: #ef4444;
    animation: none;
    box-shadow: 0 0 10px rgba(239, 68, 68, 0.5);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants & State
# ---------------------------------------------------------------------------
API_BASE_URL: str = "http://127.0.0.1:8000"

if "analysis_result" not in st.session_state: 
    st.session_state.analysis_result = None
if "chat_history" not in st.session_state: 
    st.session_state.chat_history = []

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
@st.cache_data(ttl=5)
def check_backend() -> bool:
    try: 
        return requests.get(f"{API_BASE_URL}/", timeout=1).status_code == 200
    except: 
        return False

def run_analysis(file) -> Optional[dict]:
    try:
        r = requests.post(
            f"{API_BASE_URL}/full_analysis/", 
            files={"file": (file.name, file.getvalue(), "image/jpeg")}, 
            timeout=120
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None

# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main():
    # -----------------------------------------------------------------------
    # Header Area
    # -----------------------------------------------------------------------
    col_title, col_status = st.columns([5, 1])
    with col_title:
        st.markdown("<h1 style='font-size: 3rem; margin-top: -1rem;'>🌿 AgriVision AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1.1rem; color: #a1a1aa;'>The world's most advanced crop intelligence platform.</p>", unsafe_allow_html=True)
    
    with col_status:
        is_online = check_backend()
        st.write("") # spacing
        st.write("")
        if is_online:
            st.markdown("<div style='text-align: right; color: #a1a1aa; font-weight: 500;'><span class='status-indicator'></span>System Online</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align: right; color: #a1a1aa; font-weight: 500;'><span class='status-indicator status-offline'></span>System Offline</div>", unsafe_allow_html=True)

    st.write("")
    
    # -----------------------------------------------------------------------
    # Native Tabs (Powered by Ultimate CSS)
    # -----------------------------------------------------------------------
    tab_analyse, tab_chatbot, tab_simulation, tab_dss, tab_gan, tab_about = st.tabs([
        "🔬 Analyse", 
        "🤖 Chatbot", 
        "📊 Simulation", 
        "🧠 DSS", 
        "🎨 GAN Data", 
        "📖 About"
    ])

    # -----------------------------------------------------------------------
    # 1. Analyse Tab
    # -----------------------------------------------------------------------
    with tab_analyse:
        st.write("")
        col1, col_gap, col2 = st.columns([1, 0.1, 1.2])
        
        with col1:
            st.markdown("<h3>Upload Sample</h3>", unsafe_allow_html=True)
            st.markdown("<p>Upload a high-resolution image of a leaf to begin the analysis pipeline.</p>", unsafe_allow_html=True)
            
            uploaded = st.file_uploader("", type=["jpg", "png", "jpeg"])
            
            if uploaded:
                st.image(Image.open(uploaded), use_container_width=True)
                if not is_online:
                    st.error("Cannot run analysis while backend is offline.")
                else:
                    st.write("")
                    if st.button("Initialize Deep Scan", type="primary", use_container_width=True):
                        with st.spinner("Executing neural inference and spread modelling..."):
                            res = run_analysis(uploaded)
                            if res:
                                st.session_state.analysis_result = res
                                st.success("Scan Complete.")
            
        with col2:
            res = st.session_state.analysis_result
            if not res:
                st.markdown("""
                <div style="background: rgba(24, 24, 27, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 24px; padding: 40px; text-align: center; margin-top: 2rem;">
                    <h2 style="color: #52525b; font-size: 3rem; margin-bottom: 1rem;">🔍</h2>
                    <h3 style="color: #a1a1aa;">Awaiting Telemetry</h3>
                    <p style="color: #71717a;">Upload an image on the left to initiate the diagnostic pipeline.</p>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                name = res["disease_name"].replace("___", " — ").replace("_", " ")
                conf = res["vision_detection"]["confidence"]
                
                if res.get("is_healthy"):
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0) 100%); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 20px; padding: 32px;">
                        <h2 style="color: #10b981; margin: 0; font-size: 2rem;">✅ Healthy Crop Detected</h2>
                        <p style="color: #a1a1aa; font-size: 1.1rem; margin-top: 12px;">{name}</p>
                        <p style="color: #71717a; margin-top: 8px;">No pathogens or stress markers identified. Crop is in optimal condition.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    urg = res["urgency_level"]
                    u_color = "#f43f5e" if urg == "HIGH" else "#f59e0b" if urg == "MEDIUM" else "#10b981"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(24, 24, 27, 0.8) 0%, rgba(24, 24, 27, 0.2) 100%); border: 1px solid rgba(255,255,255,0.08); border-radius: 20px; padding: 32px; position: relative; overflow: hidden;">
                        <div style="position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: {u_color}; box-shadow: 0 0 20px {u_color};"></div>
                        <h4 style="color: {u_color}; text-transform: uppercase; letter-spacing: 2px; font-size: 0.8rem; margin: 0 0 8px 0;">Urgency: {urg}</h4>
                        <h2 style="color: #f8fafc; margin: 0 0 16px 0; font-size: 2.2rem; line-height: 1.2;">{name}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("")
                    st.markdown("<p style='font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #a1a1aa;'>Neural Network Confidence</p>", unsafe_allow_html=True)
                    st.progress(int(conf), text=f"Match Probability: {conf}%")
                    
                    st.write("")
                    rec = res["recommendation"]
                    
                    st.markdown(f"""
                    <div style="margin-top: 1rem; background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 20px; padding: 32px;">
                        <h4 style="color: #10b981; margin: 0 0 16px 0; text-transform: uppercase; letter-spacing: 1px; font-size: 0.85rem;">💡 Optimized Treatment Plan</h4>
                        <h3 style="color: #f8fafc; margin: 0 0 12px 0;">{rec['best_strategy']}</h3>
                        <p style="color: #a1a1aa; line-height: 1.6;">{rec['action_summary']}</p>
                        <div style="margin-top: 16px; display: inline-block; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); color: #10b981; padding: 6px 16px; border-radius: 100px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">
                            ↓ Saves {rec['saving_vs_baseline']}% crop loss vs baseline
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.write("")
                    with st.expander("View Deep RAG Protocol Details"):
                        st.markdown(res["rag_treatment_protocol"]["retrieved_context"])

    # -----------------------------------------------------------------------
    # 2. Chatbot Tab
    # -----------------------------------------------------------------------
    with tab_chatbot:
        st.write("")
        st.markdown("<h2>AgriBot Neural Assistant</h2>", unsafe_allow_html=True)
        st.markdown("<p>Ask contextual follow-up questions powered by Llama 3 and our localized ChromaDB vector store.</p>", unsafe_allow_html=True)
        
        if not is_online:
            st.error("Cannot initialize LLM connection while backend is offline.")
        else:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Query the agricultural knowledge base..."):
                st.chat_message("user").markdown(prompt)
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                with st.spinner("Synthesizing response..."):
                    try:
                        r = requests.post(f"{API_BASE_URL}/chat/", json={"question": prompt}, timeout=30)
                        ans = r.json().get("answer", "Error") if r.status_code == 200 else f"Error: {r.status_code}"
                    except:
                        ans = "Connection error. Model timeout."
                
                with st.chat_message("assistant"):
                    st.markdown(ans)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})
                
            st.write("")
            if st.button("Purge Chat Memory"):
                st.session_state.chat_history = []
                st.rerun()

    # -----------------------------------------------------------------------
    # 3. Simulation Tab
    # -----------------------------------------------------------------------
    with tab_simulation:
        st.write("")
        st.markdown("<h2>SIR Infection Spread Dynamics</h2>", unsafe_allow_html=True)
        st.markdown("<p>Predictive modelling of crop loss trajectories across your farm grid over time.</p>", unsafe_allow_html=True)
        st.write("")
        
        res = st.session_state.analysis_result
        if not res:
            st.info("Awaiting telemetry. Upload an image in the Analyse tab to generate models.")
        elif res.get("is_healthy"):
            st.success("Target is healthy. Simulation matrices are dormant.")
        else:
            sim = res["simulation_summary"]
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Farm Grid Resolution", f"{sim['grid_size'][0]}x{sim['grid_size'][1]}", "Cells")
            m2.metric("Simulation Horizon", f"{sim['days_simulated']} Days", "+1.0x Speed")
            m3.metric("Projected Savings", f"{sim['crop_loss_saving']}%", "vs Control")
            
            st.write("")
            st.markdown("<div style='background: rgba(24, 24, 27, 0.4); border: 1px solid rgba(255,255,255,0.08); border-radius: 20px; padding: 24px;'>", unsafe_allow_html=True)
            fig = go.Figure()
            colors = ["#f43f5e", "#3b82f6", "#10b981", "#f59e0b"]
            
            for i, scen in enumerate(res["scenarios"]):
                dc = scen["daily_counts"]
                if not dc: continue
                fig.add_trace(go.Scatter(
                    x=[d["day"] for d in dc], 
                    y=[d["infected"] for d in dc],
                    name=scen["intervention_name"],
                    line=dict(color=colors[i%len(colors)], width=4),
                ))
                
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Time (Days)",
                yaxis_title="Active Infections",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 4. DSS Tab
    # -----------------------------------------------------------------------
    with tab_dss:
        st.write("")
        st.markdown("<h2>Multi-Criteria Decision Matrix</h2>", unsafe_allow_html=True)
        st.markdown("<p>Algorithmic ranking of intervention strategies maximizing yield while minimizing resource expenditure.</p>", unsafe_allow_html=True)
        st.write("")
        
        res = st.session_state.analysis_result
        if not res:
            st.info("Awaiting telemetry. Upload an image in the Analyse tab to generate models.")
        elif res.get("is_healthy"):
            st.success("Target is healthy. Decision matrix is dormant.")
        else:
            df = pd.DataFrame([{
                "Rank": s['rank'],
                "Strategy": s["intervention_name"],
                "Projected Loss": f"{s['crop_loss_percent']}%",
                "Heuristic Score": s['final_score']
            } for s in res["ranked_interventions"]])
            
            st.dataframe(
                df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn(format="#%d"),
                    "Heuristic Score": st.column_config.NumberColumn(format="%.4f")
                }
            )

    # -----------------------------------------------------------------------
    # 5. GAN Tab
    # -----------------------------------------------------------------------
    with tab_gan:
        st.write("")
        st.markdown("<h2>Synthetic Data Generation (DCGAN)</h2>", unsafe_allow_html=True)
        st.markdown("<p>How we achieve zero-shot resilience against rare pathogens.</p>", unsafe_allow_html=True)
        st.write("")
        
        st.markdown("""
        <div style="background: rgba(24, 24, 27, 0.4); border: 1px solid rgba(255,255,255,0.08); border-radius: 20px; padding: 32px;">
            <h3 style="color: #f8fafc; margin-top: 0;">Overcoming Dataset Imbalance</h3>
            <p style="color: #a1a1aa; font-size: 1.1rem; line-height: 1.7;">
                In the real world, massive agricultural datasets are heavily skewed towards common diseases. Rare pathogens lack sufficient photographic evidence to train robust neural networks.
            </p>
            <p style="color: #a1a1aa; font-size: 1.1rem; line-height: 1.7;">
                To solve this, AgriVision employs a <strong>Generative Adversarial Network (GAN)</strong>. By pitting a Generator against a Discriminator, the AI learns the underlying manifold of leaf topology and pathogen presentation. It then hallucinates photorealistic synthetic samples of rare diseases, allowing the ResNet-50 classifier to achieve high accuracy even on edge-cases never seen on your farm.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 6. About Tab
    # -----------------------------------------------------------------------
    with tab_about:
        st.write("")
        st.markdown("<h2>System Architecture</h2>", unsafe_allow_html=True)
        if not is_online:
            st.warning("API offline. Cannot fetch component status.")
        else:
            try:
                info = requests.get(f"{API_BASE_URL}/system_info/", timeout=2).json()
                st.json(info)
            except:
                st.error("Failed to fetch system info.")

if __name__ == "__main__":
    main()