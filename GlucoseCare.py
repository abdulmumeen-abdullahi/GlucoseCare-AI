# Import needed libraries
import os
import uuid
import sqlite3
from typing import Literal
import pandas as pd
import joblib
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field

# Access the Google API Key
os.environ["GOOGLE_API_KEY"] = st.secrets['GOOGLE_API_KEY']

# ==================================      SQLite Persistent memory system Setup      ==================================
class SQLiteMemory:
    def __init__(self, db_path=":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._reset_table()   # force a clean schema

    def _reset_table(self):
        self.conn.execute("DROP TABLE IF EXISTS memory")
        self.conn.execute("""
            CREATE TABLE memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT
            )
        """)
        self.conn.commit()

    def add_message(self, session_id, role, content):
        self.conn.execute(
            "INSERT INTO memory (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        self.conn.commit()

    def get_history(self, session_id):
        cur = self.conn.execute(
            "SELECT role, content FROM memory WHERE session_id = ? ORDER BY id ASC",
            (session_id,)
        )
        return cur.fetchall()
memory = SQLiteMemory("memory.db")

# Define Agent
class AgentState(dict):
    def __init__(self, session_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_id = session_id or str(uuid.uuid4())
        self["history"] = memory.get_history(self.session_id)

    def add_to_history(self, role, content):
        memory.add_message(self.session_id, role, content)
        if "history" not in self:
            self["history"] = []
        self["history"].append((role, content))

# Initialize Gemini LLM
chatGemini = ChatGoogleGenerativeAI(
    model = "gemini-1.5-flash",
    temperature = 0.3,
    google_api_key = google_api_key
)

# ==================================      Consultant Node      ==================================
# Define the Prompt Template
consultant_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a diabetes consultant and must follow the ADA (American Diabetes Association) guidelines. "
     "Your task is to answer ONLY questions that are directly or indirectly related to diabetes. "
     "This includes symptoms (even if diabetes is not explicitly mentioned), diagnosis, risk factors, lifestyle, diet, complications, monitoring, and treatment. "
     "If the user mentions a symptom or condition that *can be linked to diabetes* (e.g., visual blurring, weakness, frequent urination), you should treat it as diabetes-related and provide a concise, guideline-based response. "
     "Always provide a short safety disclaimer reminding the user to consult a qualified healthcare professional. "
     "If the user’s question is completely unrelated to health or diabetes, politely refuse. "
     "Keep answers concise, evidence-based, and user-friendly."),
    ("user", "{question}")
])

def consultant_node(state: AgentState) -> AgentState:
    query = state.get("input", "")
    state.add_to_history("user", query)

    # Symptom Detection Logic
    symptoms_list = [
        "polyuria", "polydipsia", "sudden weight loss", "weakness", "polyphagia",
        "genital thrush", "visual blurring", "itching", "irritability",
        "delayed healing", "partial paresis", "muscle stiffness", "alopecia", "obesity"
    ]

    # If any symptom keyword detected, flag prediction offer
    if any(symptom in query.lower() for symptom in symptoms_list):
        state["offer_prediction"] = True
    else:
        state["offer_prediction"] = False

    # Consultant Response
    answer = chatGemini.invoke(consultant_prompt.format(question=query))
    response = answer.content

    # If prediction offer, follow-up
    if state.get("offer_prediction", False):
        response += "\n\nI noticed you mentioned possible diabetes symptoms. " \
                    "Would you like me to run a diabetes risk prediction model for you? (Yes/No)"
        state["next_step"] = "prediction_offer"
    else:
        state["next_step"] = None

    state["output"] = response

    # Persist assistant response
    state.add_to_history("assistant", response)
    return state

# ==================================      Prediction Offer Node      ==================================
def prediction_offer_node(state: AgentState) -> AgentState:
    reply = state.get("input", "")

    # Gemini classification
    intent_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an intent classifier. "
         "Decide if the user wants to proceed with diabetes risk prediction. "
         "Answer only 'yes' or 'no'."),
        ("user", reply)
    ])

    decision = chatGemini.invoke(intent_prompt).content.strip().lower()

    if "yes" in decision:
        # Check if we already have patient features in memory
        features = state.get("features", {})

        # Ensure it's a dict
        if not isinstance(features, dict):
            features = {}
        
        # If enough features are already present, go straight to Doctor
        
        required_fields = PatientFeatures.model_fields.keys()
        if all(field in features for field in required_fields):
            state["next_step"] = "doctor"
            state["output"] = "I already have your details from earlier. Running your risk prediction now..."
        else:
            state["next_step"] = "feature_collection"
            state["output"] = "Great! Let’s start with some simple questions."
    else:
        state["next_step"] = "consultant"
        state["output"] = "No problem, we can continue chatting about diabetes."

    # Save response in memory
    state.add_to_history("assistant", state["output"])
    return state

===================================================================================
# Download Model
REPO_ID = "VisionaryQuant/Early-Stage-Diabetes-Prediction-Model"
MODEL_FILENAME = "early_stage_diabetes_best_model.pkl"

model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
print(f"Model downloaded to: {model_path}")

# Load Early Stage Diabetes model
model = joblib.load(model_path)

# Feature Schema for Random Forest model
class PatientFeatures(BaseModel):
    Age: int = Field(..., description="Age of the patient between 20 and 65")
    Gender: Literal[1, 2] = Field(..., description="1=Male, 2=Female")
    Polyuria: Literal[0, 1]
    Polydipsia: Literal[0, 1]
    sudden_weight_loss: Literal[0, 1]
    weakness: Literal[0, 1]
    Polyphagia: Literal[0, 1]
    Genital_thrush: Literal[0, 1]
    visual_blurring: Literal[0, 1]
    Itching: Literal[0, 1]
    Irritability: Literal[0, 1]
    delayed_healing: Literal[0, 1]
    partial_paresis: Literal[0, 1]
    muscle_stiffness: Literal[0, 1]
    Alopecia: Literal[0, 1]
    Obesity: Literal[0, 1]


# ==================================      Feature Collection Node      ==================================
def feature_collection_node(state: AgentState) -> AgentState:
    questions = [
        ("Age", "What is your age? (Enter a number between 20–65)"),
        ("Gender", "Are you male or female?"),
        ("Polyuria", "Do you often pass large amounts of urine (Polyuria)? (Yes/No)"),
        ("Polydipsia", "Do you feel very thirsty often (Polydipsia)? (Yes/No)"),
        ("sudden_weight_loss", "Have you experienced sudden weight loss? (Yes/No)"),
        ("weakness", "Do you often feel weak? (Yes/No)"),
        ("Polyphagia", "Do you feel excessive hunger (Polyphagia)? (Yes/No)"),
        ("Genital_thrush", "Have you had genital thrush? (Yes/No)"),
        ("visual_blurring", "Do you experience blurred vision (visual blurring)? (Yes/No)"),
        ("Itching", "Do you often feel itching? (Yes/No)"),
        ("Irritability", "Do you experience irritability? (Yes/No)"),
        ("delayed_healing", "Do your wounds take longer to heal (delayed healing)? (Yes/No)"),
        ("partial_paresis", "Do you have partial muscle weakness (paresis)? (Yes/No)"),
        ("muscle_stiffness", "Do you experience muscle stiffness? (Yes/No)"),
        ("Alopecia", "Do you have hair loss (Alopecia)? (Yes/No)"),
        ("Obesity", "Are you obese or overweight? (Yes/No)")
    ]

    idx = state.get("question_index", 0)
    user_input = state.get("input", "").strip()

    # Process the previous answer
    if idx > 0 and user_input:
        field, _ = questions[idx - 1]
        answer = user_input.lower()

        # Map answer → value
        if answer in ["yes", "1", "y"]:
            value = 1
        elif answer in ["no", "0", "n"]:
            value = 0
        elif answer in ["male", "m"]:
            value = 1
        elif answer in ["female", "f"]:
            value = 2
        else:
            try:
                value = int(answer)
            except:
                value = answer

        state["features"][field] = value
        state.add_to_history("user", f"{field}: {user_input}")

    # If done, send to doctor
    if idx >= len(questions):
        state["output"] = "Thanks, I’ve collected your details. Running your risk prediction now..."
        state.add_to_history("assistant", state["output"])
        state["next_step"] = "doctor"
        return state

    # Otherwise, ask next question
    field, q_text = questions[idx]
    state["output"] = q_text
    state.add_to_history("assistant", q_text)
    state["question_index"] = idx + 1
    state["next_step"] = "feature_collection"
    return state

# ==================================      Doctor Node      ==================================
def doctor_node(state: AgentState) -> AgentState:
    features = state.get("features", {})
    if not features:
        state["output"] = "No patient features found. Please restart the intake."
        return state

    X = pd.DataFrame([features])
    prediction = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max()
    result = "Positive" if prediction == 1 else "Negative"

    base_message = (
        f"Prediction: {result} (confidence: {prob:.2f}). "
        "This is not medical advice. Please consult a doctor."
    )

    patient_summary = ", ".join([f"{k}: {v}" for k, v in features.items()])
    preventive_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a diabetes prevention advisor. "
         "Based on patient risk prediction and their features, provide evidence-based lifestyle or dietary preventive measures. "
         "Be concise. Always include a disclaimer that this is not medical advice."),
        ("user", f"Patient features: {patient_summary}\n"
                 f"Model result: {result} (confidence {prob:.2f})\n"
                 f"Give preventive advice.")
    ])

    advice = chatGemini.invoke(preventive_prompt).content
    final_output = base_message + "\n\nPreventive Advice:\n" + advice

    state["output"] = final_output
    state.add_to_history("assistant", final_output)
    state["features"] = features
    state["next_step"] = "consultant"
    return state

# ==================================      Router Node      ==================================
def router_node(state: AgentState) -> AgentState:
    user_input = state.get("user_input", "")
    state.add_to_history("user", user_input)

    class RouteDecision(BaseModel):
        next_step: Literal["consultant", "feature_collection"]

    if any(symptom in user_input.lower() for symptom in ["Polyuria", "Polydipsia", "sudden_weight_loss", "weakness", "Polyphagia", 
                                                         "Genital thrush", "visual blurring", "Itching", "Irritability", "delayed_healing", 
                                                         "partial paresis", "muscle_stiffness", "Alopecia", "Obesity"]
):
        decision = RouteDecision(next_step="feature_collection")
    else:
        decision = RouteDecision(next_step="consultant")

    state["next_step"] = decision.next_step
    state["output"] = f"Routing to {decision.next_step}..."
    state.add_to_history("system", f"Routing to {decision.next_step}")
    return state

# ==================================      Build LangGraph      ==================================
graph = StateGraph(AgentState)

graph.add_node("router", router_node)
graph.add_node("consultant", consultant_node)
graph.add_node("feature_collection", feature_collection_node)
graph.add_node("doctor", doctor_node)

graph.add_conditional_edges(
    "router",
    lambda state, _: state["next_step"],
    {
        "feature_collection": "feature_collection",
        "consultant": "consultant"
    }
)

graph.add_edge("feature_collection", "doctor")

# Set entry point
graph.set_entry_point("router")

app = graph.compile()
