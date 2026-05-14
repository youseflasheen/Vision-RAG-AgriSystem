# 🌾 AgriVision-RAG: End-to-End Autonomous Crop Diagnosis System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An industrial-grade, microservice-based AI system designed to solve the hallucination problem in agricultural AI. By integrating a **PyTorch Vision Model** with a **Retrieval-Augmented Generation (RAG)** pipeline, this system provides zero-hallucination, expert-verified treatment protocols for crop diseases.

---

## ✨ Key Features

- **Zero-Hallucination AI:** Traditional LLMs guess plant treatments. Our RAG pipeline retrieves strictly verified agricultural protocols from a local vector database before generating a response.
- **Deep Learning Vision Engine:** Powered by a fine-tuned ResNet-50 PyTorch model for high-accuracy disease classification.
- **Blazing Fast LLM:** Utilizes Llama-3 8B via the Groq LPU, ensuring near-instantaneous protocol generation.
- **Microservices Architecture:** Fully containerized setup featuring a FastAPI backend and a responsive Streamlit frontend.
- **Docker Ready:** Deploy the entire stack anywhere with a single `docker-compose` command.

---

## 📸 System Demo & Live Results

Our system seamlessly processes leaf images to provide accurate disease detection and verified treatment plans.

### Case Study 1: Tomato Septoria Leaf Spot
| 1. Image Upload | 2. AI Detection | 3. RAG Treatment Plan |
|:---:|:---:|:---:|
| <img src="assets/tomato_upload.jpeg" width="300" alt="Upload"> | <img src="assets/tomato_detection.jpeg" width="300" alt="Detection"> | <img src="assets/tomato_treatment.jpeg" width="300" alt="Treatment"> |

### Case Study 2: Corn Northern Leaf Blight
| 1. Image Upload | 2. AI Detection | 3. RAG Treatment Plan |
|:---:|:---:|:---:|
| <img src="assets/corn_upload.jpeg" width="300" alt="Upload"> | <img src="assets/corn_detection.jpeg" width="300" alt="Detection"> | <img src="assets/corn_treatment.jpeg" width="300" alt="Treatment"> |

---

## 🧠 The Problem & Our Solution

Standard LLMs often "hallucinate" incorrect chemical dosages when asked about plant diseases, which can destroy crops. Traditional CNNs only classify the disease but leave the farmer without an actionable plan.

**AgriVision-RAG** bridges this gap:
1. **Perception:** A fine-tuned ResNet-50 model analyzes the leaf image.
2. **Retrieval:** The predicted disease class triggers a semantic search in a custom ChromaDB vector database containing verified agricultural protocols.
3. **Generation:** Llama-3 (via Groq LPU) synthesizes a context-aware, structured treatment report based *strictly* on the retrieved context.

---

## 🏗 System Architecture

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

---

## 🛠 Tech Stack

- **Deep Learning:** PyTorch, Torchvision (ResNet-50)
- **RAG & NLP:** Sentence-Transformers (all-MiniLM-L6-v2), ChromaDB, Groq API (Llama-3-8B)
- **Backend:** FastAPI, Uvicorn, Pydantic
- **Frontend:** Streamlit
- **DevOps:** Docker, Docker Compose

---

## 🚀 Getting Started

### Prerequisites
- Docker & Docker Desktop
- Groq API Key (Get one free at [Groq.com](https://groq.com/))

### Installation (Via Docker)
The easiest way to run the system in an isolated environment.

**1. Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/Vision-RAG-AgriSystem.git
cd Vision-RAG-AgriSystem
```

**2. Set up Environment Variables:**
Create a `.env` file in the root directory and add your API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

**3. Build and Run:**
```bash
docker build -t agri-vision-system .
docker run -p 8000:8000 -p 8501:8501 agri-vision-system
```

---

### Local Development (Without Docker)

**1. Create and activate virtual environment**
```bash
python -m venv agri_env
source agri_env/bin/activate  # On Windows: .\agri_env\Scripts\activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the FastAPI Backend**
```bash
uvicorn api.main:app --reload --port 8000
```

**4. Run the Streamlit Frontend (In a new terminal)**
```bash
streamlit run app/frontend.py
```

---

## 📂 Project Structure

```text
├── api/                  # FastAPI application & routes
├── app/                  # Streamlit frontend UI
├── assets/               # Demo images and icons
├── data/                 # Raw, processed, and synthetic datasets
├── docs/                 # System diagrams and presentations
├── models/               # PyTorch model weights (.pth)
├── notebooks/            # Jupyter notebooks for training (Colab)
├── src/                  # Core modules
│   ├── lab1_chatbot/     # Groq LLM Chatbot
│   ├── lab2_prompts/     # Centralized prompt templates
│   ├── lab3_data/        # Data collection & preprocessing
│   ├── lab4_rag/         # ChromaDB & Document Retrieval
│   ├── lab5_model/       # ResNet-50 Inference
│   ├── lab6_gan/         # DCGAN Synthetic Generation
│   ├── lab7_simulation/  # SIR Spread Simulation
│   ├── lab8_dashboard/   # Dashboard Utilities
│   ├── lab9_dss/         # Decision Support & Scoring
│   └── main.py           # Application entry point
├── vector_db/            # ChromaDB local persistence
├── Dockerfile            # Container configuration
├── requirements.txt      # Python dependencies
└── README.md
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/YOUR_USERNAME/Vision-RAG-AgriSystem/issues) if you want to contribute.
