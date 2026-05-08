#  AgriVision-RAG: End-to-End Autonomous Crop Diagnosis System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

An industrial-grade, microservice-based AI system designed to solve the hallucination problem in agricultural AI. By integrating a **PyTorch Vision Model** with a **Retrieval-Augmented Generation (RAG)** pipeline, this system provides zero-hallucination, expert-verified treatment protocols for crop diseases.

---

##  System Demo & Live Results

Our system seamlessly processes leaf images to provide accurate disease detection and verified treatment plans. Below are real inferences from the system:

**Case Study 1: Tomato Septoria Leaf Spot**
<table>
  <tr>
    <td align="center"><b>1. Image Upload</b></td>
    <td align="center"><b>2. AI Detection</b></td>
    <td align="center"><b>3. RAG Treatment Plan</b></td>
  </tr>
  <tr>
    <td><img src="assets/tomato_upload.jpeg" width="300"></td>
    <td><img src="assets/tomato_detection.jpeg" width="300"></td>
    <td><img src="assets/tomato_treatment.jpeg" width="300"></td>
  </tr>
</table>

**Case Study 2: Corn Northern Leaf Blight**
<table>
  <tr>
    <td align="center"><b>1. Image Upload</b></td>
    <td align="center"><b>2. AI Detection</b></td>
    <td align="center"><b>3. RAG Treatment Plan</b></td>
  </tr>
  <tr>
    <td><img src="assets/corn_upload.jpeg" width="300"></td>
    <td><img src="assets/corn_detection.jpeg" width="300"></td>
    <td><img src="assets/corn_treatment.jpeg" width="300"></td>
  </tr>
</table>

---

##  The Problem & Our Solution
Standard LLMs often "hallucinate" incorrect chemical dosages when asked about plant diseases, which can destroy crops. Traditional CNNs only classify the disease but leave the farmer without an actionable plan.

**AgriVision-RAG** bridges this gap:
1. **Perception:** A fine-tuned ResNet-50 model analyzes the leaf image.
2. **Retrieval:** The predicted disease class triggers a semantic search in a custom ChromaDB vector database containing verified agricultural protocols.
3. **Generation:** Llama-3 (via Groq LPU) synthesizes a context-aware, structured treatment report based *strictly* on the retrieved context.

---

##  System Architecture

```mermaid
graph TD
    %% User Interaction Layer
    subgraph Frontend [UI Layer]
        A[ User / Farmer] -->|Uploads Leaf Image| B(Streamlit Dashboard)
    end

    %% API Orchestration Layer
    subgraph Backend [API Layer]
        B <-->|REST API JSON| C{FastAPI Server}
    end

    %% Deep Learning Perception
    subgraph Vision_Engine [Computer Vision Engine]
        C -->|Image Tensor| D[ResNet-50 Model]
        D -->|Feature Extraction| E((Predicted Class))
    end

    %% Retrieval-Augmented Generation Layer
    subgraph RAG_Pipeline [Knowledge & RAG Engine]
        E -->|Search Query| F[(ChromaDB Vector Store)]
        G[Expert Agricultural JSON] -->|Embeddings| F
        F -->|Top-K Context| H[Prompt Engineering Setup]
        H -->|Context + Query| I[Llama-3 LLM via Groq]
    end

    %% Final Output
    I -->|Verified Treatment Protocol| C
    C -->|Renders Report| B

    %% Styling for a professional look
    classDef ui fill:#0e1117,stroke:#ff4b4b,stroke-width:2px,color:#fff;
    classDef api fill:#005571,stroke:#059669,stroke-width:2px,color:#fff;
    classDef ml fill:#ee4c2c,stroke:#fff,stroke-width:2px,color:#fff;
    classDef rag fill:#2496ed,stroke:#fff,stroke-width:2px,color:#fff;
    classDef data fill:#f39c12,stroke:#fff,stroke-width:2px,color:#fff;

    class B ui;
    class C api;
    class D,E ml;
    class H,I rag;
    class F,G data;

```

