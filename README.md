---
title: GlucoseCare AI
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# GlucoseCare AI 🤖🩺

**Agentic Early-Stage Diabetes Risk Consultant**  
*A stateful, hybrid AI assistant combining LangGraph orchestration, resilient multi-LLM fallback, and a Random Forest clinical classifier to deliver personalized diabetes risk assessments and guideline-aligned preventive care.*

[![Live Demo](https://img.shields.io/badge/Live_Demo-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://glucosecare-ai.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.12.3-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-1C3C3C?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![Chainlit](https://img.shields.io/badge/UI-Chainlit-F80061?style=for-the-badge)](https://docs.chainlit.io/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

> **Live Application:** [https://glucosecare-ai.onrender.com/](https://glucosecare-ai.onrender.com/)  
> *Note: Hosted on Render's free tier. If the service has been idle, the initial cold start may take **30–60 seconds** to wake up.*

---

## Demo & Walkthrough

<!-- TIP: Drag and drop your recorded .mp4/.mov demo video directly into the GitHub README editor to replace the link below with an auto-playing inline GitHub video player. -->
🎥 **[Watch the Walkthrough Video](#)** *(Replace `#` with your uploaded GitHub video asset URL or screen-recording link)*

<p align="center">
  <img width="1366" height="724" alt="GlucoseCare AI Agent Chainlit UI" src="https://github.com/user-attachments/assets/6eb81cba-1a3d-4f4a-8416-91fe6441585b" />
  <br>
  <em>Figure 1: GlucoseCare AI interactive Chainlit session demonstrating symptom triage, structured feature collection, and ML-backed risk assessment.</em>
</p>

---

## Overview

**GlucoseCare AI** bridges conversational AI with deterministic machine learning to address a core challenge in healthcare AI: **preventing LLM hallucination during clinical risk scoring while maintaining natural, empathetic user guidance.**

Instead of asking an LLM to guess a patient's risk, GlucoseCare AI uses a **LangGraph state machine** to classify user intent, guide users through a validated 12-feature clinical questionnaire, run inference against a trained **Random Forest Classifier**, and synthesize actionable lifestyle recommendations aligned with **ADA (American Diabetes Association)**, **IDF (International Diabetes Federation)**, and **WHO (World Health Organization)** guidelines.

### Key Engineering Highlights

- **Hybrid Neuro-Symbolic / ML Reasoning:** Decouples conversational triage (LLMs) from quantitative risk prediction (scikit-learn Random Forest) for repeatable, probability-backed assessments.
- **Stateful Multi-Node Graph (`LangGraph`):** Manages multi-turn conversation state, intent routing, and sequential Q&A checkpoints per session using `MemorySaver` and unique UUID thread IDs.
- **Resilient Dual-Provider LLM Fallback:** Routes primary inference through **Groq** (`openai/gpt-oss-20b`) for low-latency responses with automatic failover to **Google Gemini** (`gemini-3.8-flash`).
- **Strict Input Validation (`Pydantic`):** Enforces schema and binary/integer type validation across all 12 clinical indicators (`PatientFeatures`) before passing feature vectors to the ML pipeline.
- **Production-Ready Async Controls:** Implements `aiolimiter` request throttling (`AsyncLimiter`), session reset commands (`reset`, `/reset`, `restart`), and containerized deployment via **Docker**.

---

## System Architecture



### Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Orchestration** | `LangGraph` (`StateGraph`, `MemorySaver`) | Stateful routing, multi-turn slot filling, and thread persistence |
| **Primary LLM** | Groq (`openai/gpt-oss-20b`) | Low-latency intent detection, consultation, and prevention advice |
| **Fallback LLM** | Google Gemini (`gemini-3.8-flash`) | Automatic failover for high availability |
| **ML Inference** | `scikit-learn` (`RandomForestClassifier`), `joblib`, `pandas` | Early-stage diabetes risk classification and confidence scoring |
| **Model Registry** | Hugging Face Hub (`hf_hub_download`) | Fetches serialized pre-trained `.pkl` weights at startup |
| **Validation & Rate Limiting** | `Pydantic v2`, `aiolimiter` | Clinical feature schema enforcement and async request throttling |
| **Frontend & Runtime** | `Chainlit`, `Docker`, `Render` | Conversational web UI, containerization, and cloud hosting |

---

## Project Structure

```text
GlucoseCare-AI/
├── GlucoseCare.py      # Core LangGraph agent, ML inference pipeline, and Chainlit handlers
├── chainlit.md         # Chainlit welcome screen and UI configuration markdown
├── Dockerfile          # Production container specification for Render / cloud deployment
├── requirements.txt    # Pinned Python dependencies
└── README.md           # Project documentation
```

---

## Getting Started

### Prerequisites

- Python **3.12.3** (or **Docker**)
- API keys for [Google AI Studio (Gemini)](https://ai.google.dev/) and [Groq Cloud](https://console.groq.com/)

### Option 1: Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/abdulmumeen-abdullahi/GlucoseCare-AI.git
   cd GlucoseCare-AI
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**  
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

5. **Run the application**
   ```bash
   chainlit run GlucoseCare.py -w
   ```
   Navigate to `http://localhost:8000` in your browser. Type `reset` or `/reset` at any point in the chat to clear session state.

### Option 2: Run with Docker

The included `Dockerfile` is portable across Render, Google Cloud Run, and local container engines:

```bash
docker build -t glucosecare-ai .
docker run -p 8000:8000 --env-file .env glucosecare-ai
```

---

## Medical Disclaimer

GlucoseCare AI is **not a certified medical device** and is intended strictly for **educational, research, and portfolio demonstration purposes**. Predictions and lifestyle recommendations generated by this system do not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional regarding any medical condition or health concerns.