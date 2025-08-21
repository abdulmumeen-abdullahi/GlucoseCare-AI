# Import needed libraries
import os
import uuid
import sqlite3
from typing import Dict, Any, List, Tuple, Literal
import pandas as pd
import joblib
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field, ValidationError

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
    def __init__(self, session_id: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_id = session_id or str(uuid.uuid4())

        # Initialize safe defaults
        self.setdefault("history", memory.get_history(self.session_id) or [])
        self.setdefault("features", {})       # patient features
        self.setdefault("output", None)       # last system/assistant output
        self.setdefault("next_step", None)    # routing control
        self.setdefault("metadata", {})       # misc info (timestamps, retries, etc.)

    def add_to_history(self, role: str, content: str):
        # Adds a message to both persistent memory and in-session history.
        memory.add_message(self.session_id, role, content)
        if "history" not in self:
            self["history"] = []
        self["history"].append((role, content))

    def get_latest_user_input(self) -> str:
        # Retrieve the latest user input from history if available.
        for role, msg in reversed(self["history"]):
            if role == "user":
                return msg
        return ""

    def get(self, key: str, default: Any = None) -> Any:
        # Safe dictionary get with defaults for missing keys.
        return super().get(key, default)

    def update_features(self, new_features: Dict[str, Any]):
        # Update patient features incrementally.
        self["features"].update(new_features)

# ==========================================================================         Initialize Gemini LLM
chatGemini = ChatGoogleGenerativeAI(
    model = "gemini-1.5-flash",
    temperature = 0.3,
    google_api_key = google_api_key
)

# ==================================      Model & Schema      ==================================
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
    Gender: Literal[0, 1] = Field(..., description="1 = Male, 0 = Female")
    Polyuria: Literal[0, 1]
    Polydipsia: Literal[0, 1]
    sudden weight loss: Literal[0, 1]
    weakness: Literal[0, 1]
    Polyphagia: Literal[0, 1]
    Genital thrush: Literal[0, 1]
    visual blurring: Literal[0, 1]
    Itching: Literal[0, 1]
    Irritability: Literal[0, 1]
    delayed healing: Literal[0, 1]
    partial paresis: Literal[0, 1]
    muscle stiffness: Literal[0, 1]
    Alopecia: Literal[0, 1]
    Obesity: Literal[0, 1]

# Router decision schema
class RouteDecision(BaseModel):
    next_step: Literal["consultant", "prediction_offer", "end"]

# Router prompt Template
router_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a router that decides whether a user query is diabetes-related. "
     "Rules: "
     "1. If the query is unrelated to diabetes or health, respond politely with a short refusal message "
     "and set next_step='end'. "
     "2. If the query is about diabetes risks, lifestyle, diet, monitoring, or complications (but not heavy symptoms), "
     "set next_step='consultant'. "
     "3. If the query contains diabetes-related symptoms "
     "(e.g., frequent urination, excessive thirst, sudden weight loss, weakness, visual blurring, etc), answer the query, "
     "tell the user that you noticed symptoms of diabetes in hs query and ask if he would like to get a diabetes risk prediction. "
     "If yes, set next_step to 'prediction_offer', else set next_step to'consultant'. "
     "Note: The risk prediction should only be offered once per conversation."),
    ("user", "{question}")
])

# ==================================      Consultant Node      ==================================
# Prompt Template
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
    response = chatGemini.invoke(consultant_prompt.format(question=query))
    state.add_to_history("assistant", response)
    state["output"] = response
    
    return state

# ==================================      Prediction Offer Node      ==================================
def prediction_offer_node(state: AgentState) -> AgentState:
    reply = state.get("input", "")

    # Classify yes/no intent
    intent_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an intent classifier. "
         "Decide if the user wants to proceed with diabetes risk prediction. "
         "Answer only 'yes' or 'no'."),
        ("user", reply)
    ])
    decision = chatGemini.invoke(intent_prompt).content.strip().lower()

    # Get existing patient features
    features = state.get("features") or {}
    if not isinstance(features, dict):
        features = {}

    # If enough features are already present, go straight to Doctor
    required_fields = PatientFeatures.model_fields.keys()

    if "yes" in decision:
        if all(field in features for field in required_fields):
            state["next_step"] = "doctor"
            state["output"] = "I already have your details from earlier. Running your risk prediction now..."
        else:
            state["next_step"] = "feature_collection"
            state["output"] = "Great! Let’s start with some simple questions."
    else:
        state["next_step"] = "consultant"
        state["output"] = "No problem, we can continue chatting about diabetes."

    # Save assistant message in memory
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

# ==================================      Feature Collection Node      ==================================
def feature_collection_node(state: AgentState) -> AgentState:
    # Ensure features container exists
    state.setdefault("features", {})

    # Ordered questions
    questions = [
        ("Age", "What is your age? (Enter a number between 20–65)"),
        ("Gender", "Are you male or female? (Type 'male' or 'female')"),
        ("Polyuria", "Do you often pass large amounts of urine (Polyuria)? (Yes/No)"),
        ("Polydipsia", "Do you feel very thirsty often (Polydipsia)? (Yes/No)"),
        ("sudden weight loss", "Have you experienced sudden weight loss? (Yes/No)"),
        ("weakness", "Do you often feel weak? (Yes/No)"),
        ("Polyphagia", "Do you feel excessive hunger (Polyphagia)? (Yes/No)"),
        ("Genital thrush", "Have you had genital thrush? (Yes/No)"),
        ("visual blurring", "Do you experience blurred vision (visual blurring)? (Yes/No)"),
        ("Itching", "Do you often feel itching? (Yes/No)"),
        ("Irritability", "Do you experience irritability? (Yes/No)"),
        ("delayed healing", "Do your wounds take longer to heal (delayed healing)? (Yes/No)"),
        ("partial paresis", "Do you have partial muscle weakness (paresis)? (Yes/No)"),
        ("muscle stiffness", "Do you experience muscle stiffness? (Yes/No)"),
        ("Alopecia", "Do you have hair loss (Alopecia)? (Yes/No)"),
        ("Obesity", "Are you obese or overweight? (Yes/No)")
    ]

    idx = state.get("question_index", 0)
    user_input = state.get("input", "").strip()

    # Handle previous answer
    if idx > 0 and user_input:
        field, _ = questions[idx - 1]
        ans = user_input.lower().strip()

        def map_answer(field_name: str, text: str):
            if field_name == "Age":
                try:
                    age = int(text)
                    if 20 <= age <= 65:
                        return age, None
                    return None, "Please enter a valid age between 20 and 65."
                except:
                    return None, "Please enter a whole number for age (20–65)."

            if field_name == "Gender":
                if text in ["male", "m", "1"]:
                    return 1, None  # male = 1
                if text in ["female", "f", "0", "2"]:
                    return 0, None  # female = 0
                return None, "Please type 'male' or 'female'."

            # Binary symptom fields
            if text in ["yes", "y", "1"]:
                return 1, None
            if text in ["no", "n", "0"]:
                return 0, None

            return None, f"Please answer 'Yes' or 'No' for {field_name}."

        value, err = map_answer(field, ans)
        if err:
            state["output"] = err
            state.add_to_history("assistant", err)
            state["next_step"] = "feature_collection"
            return state

        # Save mapped value
        state["features"][field] = value
        state.add_to_history("user", f"{field}: {user_input}")

    # Completion Check
    if idx >= len(questions):
        try:
            # Validate against PatientFeatures schema
            PatientFeatures(**state["features"])
            msg = "Thanks, I’ve collected your details. Running your risk prediction now..."
            state["output"] = msg
            state.add_to_history("assistant", msg)
            state["next_step"] = "doctor"
            state["question_index"] = 0  # reset for future intake
            return state
        except Exception as e:
            # In case schema validation fails (e.g. missing field)
            err = f"Some information is missing or invalid: {str(e)}"
            state["output"] = err
            state.add_to_history("assistant", err)
            state["next_step"] = "feature_collection"
            return state

    # Ask Next Question
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

    try:
        # Validate features against schema before prediction
        PatientFeatures(**features)

        # Prepare input for model
        X = pd.DataFrame([features])
        prediction = model.predict(X)[0]
        prob = model.predict_proba(X)[0].max()
        result = "Positive" if prediction == 1 else "Negative"

        base_message = (
            f"Prediction: {result} (confidence: {prob:.2f}). "
            "This is not medical advice. Please consult a doctor."
        )

        # Build patient summary string
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
        state["next_step"] = "consultant"  # return to consultant after prediction

    except ValidationError as e:
        msg = f"Some details are missing or invalid: {e}. Please restart the intake."
        state["output"] = msg
        state.add_to_history("assistant", msg)
        state["next_step"] = "consultant"

    return state

# ==================================      Router Node      ==================================
def router_node(state: AgentState) -> AgentState:
    query = state.get("input", "")
    state.add_to_history("user", query)

    # Prevent asking twice for prediction
    if state.get("prediction_offered", False):
        decision = RouteDecision(next_step="consultant")
    else:
        decision = chatGemini.with_structured_output(RouteDecision).invoke(
            router_prompt.format(question=query)
        )

    # Store next step in state
    state["next_step"] = decision.next_step

    # Handle refusal inline if next_step == "end"
    if decision.next_step == "end":
        refusal = "I'm here to help with diabetes-related questions only. Please ask me about diabetes symptoms, risks, or lifestyle. Thanks."
        state.add_to_history("assistant", refusal)
        state["output"] = refusal

    # Mark that prediction has been offered once
    if decision.next_step == "prediction_offer":
        state["prediction_offered"] = True

    return state

# ==================================      Normalize next_step to a graph target or END      ==================================
def route_from_state(state) -> Literal["consultant", "prediction_offer", "feature_collection", "doctor", "end"]:
    step = state.get("next_step")
    # Defensive normalization
    if step in {"consultant", "prediction_offer", "feature_collection", "doctor", "end"}:
        return step
    # Treat None/unknown as end
    return "end"

# ==================================      Build LangGraph      ==================================
graph = StateGraph(AgentState)

# Add node
graph.add_node("router", router_node)
graph.add_node("feature_collection", feature_collection_node)
graph.add_node("consultant", consultant_node)

# Set Entry point
graph.set_entry_point("router")

# Add Edge
graph.add_edge("router", "feature_collection")
graph.add_edge("router", "consultant")
graph.add_edge("feature_collection", END)
graph.add_edge("consultant", END)

# Compile app
app = graph.compile()

# Run helper
def run_turn(state: AgentState, user_message: str) -> AgentState:
    # Put the new user input into state
    state["input"] = user_message

    # Execute the graph from the entry node until END
    final_state = app.invoke(state)

    # Read the assistant output
    return final_state
