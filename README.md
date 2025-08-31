# GlucoseCare AI Agent 🤖

**Early Diabetes Risk Consultant Agent**
*AI-powered assistant for early-stage diabetes awareness, prevention, and personalized risk assessment.*

---

## Overview

GlucoseCare AI is an intelligent LangGraph-powered agent designed to consult, collect patient symptoms, and run an early-stage diabetes risk prediction using a machine learning model. It integrates conversational AI (LLMs) with structured stateful flows to provide personalized advice aligned with ADA, IDF, and WHO guidelines**.

This project demonstrates:

* ✅ **Hybrid reasoning:** LLM + predictive ML model.
* ✅ **Structured state management:** Built with `LangGraph` and `MemorySaver`.
* ✅ **Medical use-case compliance:** Always includes disclaimers.
* ✅ **Modern agentic stack:** Google Gemini, Groq (LLaMA 3), Hugging Face Hub, Chainlit UI.

GlucoseCare AI is a **portfolio project** showcasing applied AI engineering in healthcare.

---

## ⚙️ Features

* **Intent Classification**: Routes user queries (symptoms, lifestyle, greetings, prediction requests).
* **Feature Collection**: Guided questionnaire (age, gender, polyuria, polydipsia, etc.).
* **ML Prediction**: Runs a **pre-trained early-stage diabetes model** hosted on Hugging Face.
* **LLM Prevention Advice**: Provides personalized, concise prevention tips.
* **Memory & State Graph**: Conversation flow managed by LangGraph with checkpointing.
* **Fallback LLM Calls**: Automatically retries between Groq (LLaMA 3) and Gemini.
* **Interactive UI**: Chainlit chat interface with session reset.

---

## 🏗️ Architecture

```
User → Chainlit Chat UI → LangGraph State Machine
       ├─ Router Node (intent detection via LLM)
       ├─ Consultant Node (general Q&A, prevention)
       ├─ Feature Collection Node (structured Q&A)
       ├─ Doctor Node (ML prediction + LLM advice)
       └─ MemorySaver (threaded checkpointing)
```

* **LLMs**:

  * [Groq LLaMA3-8B](https://groq.com/) (fast inference, default)
  * [Google Gemini 1.5 Flash](https://ai.google.dev/) (fallback)
* **ML Model**: [VisionaryQuant/Early-Stage-Diabetes-Prediction-Model](https://huggingface.co/VisionaryQuant/Early-Stage-Diabetes-Prediction-Model)
* **Orchestration**: LangGraph + StateGraph + MemorySaver
* **Interface**: Chainlit

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/GlucoseCare-AI.git
cd GlucoseCare-AI
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

### 5. Run the Agent

```bash
chainlit run GlucoseCare.py -w
```

Open your browser at `http://localhost:8000`.

---

## 🧪 Example Usage

**User:** "I feel constant thirst and urinate a lot."
**Agent:**

```
I noticed you mentioned health-related issues.  
Would you like me to run a quick diabetes risk check? (Yes/No)
```

**User:** "Yes"
**Agent:**

```
Great! Let’s begin with some quick questions.  
What is your age?
```

... → After answering the questionnaire →

**Agent:**

```
Prediction: Diabetic (confidence 0.87).  
Note: Consult a medical practitioner for better diagnosis.  

Preventive Advice: Maintain hydration, monitor blood sugar, and seek medical consultation promptly.
```

---

## 📂 Project Structure

```
GlucoseCare-AI/
│── GlucoseCare.py         # Main Chainlit + LangGraph app
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
│── .env.example           # API key template
```

---

## ⚠️ Disclaimer

GlucoseCare AI is **not a medical device**. It is a **research and educational project** intended to demonstrate AI agent engineering. Always consult a **qualified healthcare provider** for medical advice, diagnosis, or treatment.

---

## 🌟 Portfolio Value

* Demonstrates **applied AI agent design** using **LangGraph**.
* Combines **ML prediction** with **LLM-based consultation**.
* Shows ability to **deploy AI-powered assistants** in healthcare.
* Highlights **structured state management** vs. free-form chatbots.
