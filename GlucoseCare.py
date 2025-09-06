# =============================            GlucoseCare AI - Early Diabetes Risk Consultant            =============================

# Import needed libraries
import os, traceback
from typing import Dict, Any, Literal
from typing_extensions import Annotated, TypedDict
import pandas as pd
import joblib
import chainlit as cl
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from sklearn.preprocessing import LabelEncoder
from langgraph.graph import StateGraph, END
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field, ValidationError
from aiolimiter import AsyncLimiter
from langgraph.checkpoint.memory import MemorySaver
import uuid

# =============================            Load API Keys            =============================
_ = load_dotenv(find_dotenv())
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# =============================            State Management            =============================

# Default state
DEFAULT_STATE = {
    "history": [],
    "features": {},
    "output": None,
    "next_step": None,
    "asked_prediction_offer": False,
    "awaiting_offer_reply": False,
    "question_index": 0,
    "input": "",
}

# Define State
class GlucoseState(TypedDict):
    history: Annotated[list, "history"]
    features: Annotated[dict, "features"]
    output: Annotated[str, "output"]
    next_step: Annotated[str, "next_step"]
    asked_prediction_offer: Annotated[bool, "asked_prediction_offer"]
    awaiting_offer_reply: Annotated[bool, "awaiting_offer_reply"]
    question_index: Annotated[int, "question_index"]
    input: Annotated[str, "input"]

# ---- Helper Functions
def ensure_state(state: Dict[str, Any]) -> Dict[str, Any]:
    s = dict(DEFAULT_STATE)
    if state:
        s.update(state)
    return s
def add_to_history(state: Dict[str, Any], role: str, content: str):
    state.setdefault("history", []).append({"role": role, "content": content or ""})
def safe_text(msg) -> str:
    return getattr(msg, "content", str(msg)) if msg else ""
def safe_query(state: Dict[str, Any]) -> str:
    return (state.get("input") or "").strip()

# =============================            LLMs            =============================
# Define LLMs
chatGemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY
)
chatGroq = ChatGroq(model="llama3-8b-8192", groq_api_key=GROQ_API_KEY)

# Set fallback
def call_llm_with_fallback(messages, prefer="groq") -> str:
    try:
        if prefer == "gemini":
            return safe_text(chatGemini.invoke(messages))
        return safe_text(chatGroq.invoke(messages))
    except Exception:
        try:
            if prefer == "gemini":
                return safe_text(chatGroq.invoke(messages))
            return safe_text(chatGemini.invoke(messages))
        except Exception:
            return "I couldn’t process that."

# =============================            Dabetes Model & Schema            =============================
# HuggingFace Details
REPO_ID = "VisionaryQuant/Early-Stage-Diabetes-Prediction-Model"
MODEL_FILENAME = "Early Stage Diabetes Model.pkl"
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
# Download model
model = joblib.load(model_path)

# Define model features
class PatientFeatures(BaseModel):
    Age: int
    Gender: Literal[0, 1]
    Polyuria: Literal[0, 1]
    Polydipsia: Literal[0, 1]
    sudden_weight_loss: Literal[0, 1] = Field(..., alias="sudden weight loss")
    Polyphagia: Literal[0, 1]
    visual_blurring: Literal[0, 1] = Field(..., alias="visual blurring")
    Itching: Literal[0, 1]
    Irritability: Literal[0, 1]
    delayed_healing: Literal[0, 1] = Field(..., alias="delayed healing")
    partial_paresis: Literal[0, 1] = Field(..., alias="partial paresis")
    Alopecia: Literal[0, 1]

    model_config = {"validate_by_name": True, "populate_by_name": True}

# =============================            Prompts            =============================
consultant_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are GlucoseCare AI, an ADA, IDF, and WHO aligned diabetes consultant. "
               "Be concise, helpful, and always add a short disclaimer for medication or seensitive information."),
    ("user", "{question}"),
])

prevention_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an ADA, IDF, and WHO aligned diabetes prevention advisor. Give concise, actionable tips. Always include a short disclaimer."),
    ("user", "Patient: {patient}\nResult: {result} ({probability})"),
])

# =============================            Nodes            =============================
# ---------------- Router Node ----------------
def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state = ensure_state(state)
    q = safe_query(state)
    add_to_history(state, "user", q)

    # Guard: if we’re in the middle of feature collection, don’t re-run intent detection
    if state.get("next_step") == "feature_collection" and state.get("question_index", 0) > 0:
        # keep the flow inside feature_collection until that node moves us to "doctor"
        state["next_step"] = "feature_collection"
        return state

    # intent detection logic
    messages = [
        {"role": "system", "content": (
            "You are an intent classifier for GlucoseCare AI. "
            "Classify the user's input into one of these intents:\n"
            "- greeting (hello, hi, etc.)\n"
            "- symptom_report (mentions health symptoms)\n"
            "- lifestyle (diet, exercise, sleep, stress, etc.)\n"
            "- prediction_request (the user explicitly asks to check THEIR OWN diabetes risk or wants a risk test)\n"
            "- other (general chat, medical knowledge, population statistics, or anything else)"
        )},
        {"role": "user", "content": q}
    ]
    intent = call_llm_with_fallback(messages, prefer="groq").lower()
    state["last_intent"] = intent

    if "symptom_report" in intent and not state.get("asked_prediction_offer", False):
        state["asked_prediction_offer"] = True
        state["awaiting_offer_reply"] = True
        state["output"] = (
            "I noticed you mentioned some health-related issues. "
            "Would you like me to run a quick diabetes risk check? (Yes/No)"
        )
        state["next_step"] = "consultant"

    elif "prediction_request" in intent or (
        state.get("awaiting_offer_reply", False) and q.lower() in ["yes", "yeah", "y", "sure"]
    ):
        state["awaiting_offer_reply"] = False
        state["next_step"] = "feature_collection"
        state["output"] = "Yes — I can help you predict your risk. Let’s begin with a few quick questions."
        return state

    elif "lifestyle" in intent:
        state["output"] = (
            "Got it. I’ll note your lifestyle details.\n"
            "Would you like me to run a diabetes risk check too?"
        )
        state["awaiting_offer_reply"] = True
        state["next_step"] = "consultant"

    elif "greeting" in intent:
        state["output"] = (
            "Hello! I'm **GlucoseCare AI**.\n"
            "I can answer your questions about diabetes, symptoms, or run a quick risk check."
        )
        state["next_step"] = "consultant"

    else:
        state["next_step"] = "consultant"
    return state

# ---------------- Consultant Node ----------------
def consultant_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state = ensure_state(state)
    user_text = safe_query(state)

    # Handle Yes/No reply to prediction offer
    if state.get("awaiting_offer_reply", False):
        if user_text.lower() in ["yes", "y"]:
            state["awaiting_offer_reply"] = False
            state["next_step"] = "feature_collection"
            state["output"] = "Great! Let’s begin with some quick questions."
            return state
        elif user_text.lower() in ["no", "n"]:
            state["awaiting_offer_reply"] = False
            state["next_step"] = END  # stop here, wait for next input
            state["output"] = "No problem. Let’s continue discussing diabetes."
            return state

    # Include memory
    messages = [{"role": h["role"], "content": h["content"]} for h in state.get("history", [])]
    messages.append({"role": "user", "content": user_text})
    response = call_llm_with_fallback(messages, prefer="groq")

    # Otherwise, normal LLM consultant
    """response = call_llm_with_fallback(
        consultant_prompt.format_prompt(question=user_text).to_messages(),
        prefer="groq",
    )"""
    state["output"] = response
    state["next_step"] = END
    return state


# ---------------- Feature Collection Node ----------------
def feature_collection_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state = ensure_state(state)
    questions = [
        ("Age", "What is your age?"),
        ("Gender", "Are you male or female?"),
        ("Polyuria", "Do you urinate excessively? (Yes/No)"),
        ("Polydipsia", "Do you feel excessive thirst? (Yes/No)"),
        ("sudden weight loss", "Have you had sudden weight loss? (Yes/No)"),
        ("Polyphagia", "Do you often feel excessive hunger? (Yes/No)"),
        ("visual blurring", "Do you have blurred vision? (Yes/No)"),
        ("Itching", "Do you experience itching? (Yes/No)"),
        ("Irritability", "Do you experience irritability? (Yes/No)"),
        ("delayed healing", "Do you have slow healing wounds? (Yes/No)"),
        ("partial paresis", "Do you have partial muscle weakness? (Yes/No)"),
        ("Alopecia", "Do you have hair loss (Alopecia)? (Yes/No)"),
    ]
    idx = state.get("question_index", 0)
    user_input = safe_query(state)

    # Save user response for the previous question
    if idx > 0 and user_input:
        field, _ = questions[idx - 1]
        ans = user_input.lower()
        val = None
        if field == "Age":
            try:
                val = int(user_input)
            except:
                val = None
        elif field == "Gender":
            if ans in ["male", "m", "1"]:
                val = 1
            elif ans in ["female", "f", "0"]:
                val = 0
        else:
            if ans in ["yes", "y", "1"]:
                val = 1
            elif ans in ["no", "n", "0"]:
                val = 0

        if val is None:
            state["output"] = f"Please answer {field} correctly."
            state["next_step"] = "feature_collection"
            return state

        # store encoded answer
        state["features"][field] = val

    # If all questions are answered, move to doctor
    if idx >= len(questions):
        try:
            # Validate with Pydantic
            PatientFeatures(**state["features"])

            state["output"] = "Thanks, I have all the details. Running prediction..."
            state["next_step"] = "doctor"
            state["question_index"] = 0  # reset
            return state

        except ValidationError as e:
            state["output"] = f"Some details are missing: {e}"
            state["next_step"] = "feature_collection"
            return state

    # If all questions are not answered, ask the next question
    field, qtext = questions[idx]
    preface = state["output"] if idx == 0 and "output" in state else ""
    state["output"] = (preface + ("\n\n" if preface else "") + qtext)
    state["question_index"] = idx + 1
    state["next_step"] = "feature_collection"
    return state

# ---------------- Doctor Node ----------------
def doctor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state = ensure_state(state)
    try:
        # Convert collected features into a DataFrame
        input_data = pd.DataFrame([state["features"]])

        # Map values back into the format the model expects
        # (Yes/No for symptoms, Male/Female for gender)
        reverse_map = {1: "Yes", 0: "No"}
        gender_map = {1: "Male", 0: "Female"}

        # Convert numerical encodings back to string labels
        for col in input_data.columns:
            if col == "Gender":
                input_data[col] = input_data[col].map(gender_map)
            elif col != "Age":
                input_data[col] = input_data[col].map(reverse_map)

        # Reapply the same preprocessing pipeline as training
        le_gender = LabelEncoder().fit(["Male", "Female"])
        input_data["Gender"] = le_gender.transform(input_data["Gender"])
        for col in input_data.columns[2:]:
            input_data[col] = input_data[col].map({"Yes": 1, "No": 0})

        # Predict
        pred = model.predict(input_data)[0]
        prob = float(model.predict_proba(input_data)[0].max())
        result = "Diabetic" if pred == 1 else "Not diabetic"

        # Build summary for advice
        summary = ", ".join([f"{k}: {v}" for k, v in state["features"].items()])
        advice = call_llm_with_fallback(
            prevention_prompt.format_prompt(
                patient=summary,
                result="Diabetic" if pred == 1 else "Not diabetic",
                probability=f"{prob:.2f}"
            ).to_messages()
        )

        state["output"] = (
            f"Prediction: **{result}** (confidence {prob:.2f}).\n\n"
            "Note: Consult a medical practitioner for better diagnosis.\n\n"
            f"**Preventive Advice:**\n{advice}"
        )
        state["next_step"] = END

    except Exception as e:
        state["output"] = f"Prediction failed: {e}"
        state["next_step"] = END
    return state
# =============================            Graph Assembly            =============================
def graph_tracer(state: Dict[str, Any], event: str):
    print(f"[GRAPH TRACE] {event} | next_step={state.get('next_step')}")

graph = StateGraph(GlucoseState)
# Add nodes
graph.add_node("router", router_node)
graph.add_node("consultant", consultant_node)
graph.add_node("feature_collection", feature_collection_node)
graph.add_node("doctor", doctor_node)
# Set Entry point
graph.set_entry_point("router")

# Add conditional edges
graph.add_conditional_edges("router", lambda s: s.get("next_step"), {
    "consultant": "consultant",
    "feature_collection": "feature_collection",
})

graph.add_conditional_edges(
    "consultant", lambda s: s.get("next_step"), {
        "feature_collection": "feature_collection",
        "consultant": END,  # if it says consultant again, just stop
        END: END
    },
)

# feature_collection -> {ask next question (end, wait for input) | doctor (ready)}
graph.add_conditional_edges(
    "feature_collection", lambda s: s.get("next_step"), {
        "doctor": "doctor",
        "feature_collection": END,  # stop and wait until next user reply
    },
)

# Add edge
graph.add_edge("doctor", END)
graph.add_edge("consultant", END)
# Add memory
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# =============================            Chainlit Integration            =============================

llm_limiter = AsyncLimiter(3, 1)

@cl.on_chat_start
async def start_chat():
    # Create a unique thread_id for this session
    thread_id = str(uuid.uuid4())
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("agent_state", ensure_state({}))
    await cl.Message(
        content="""Hello! I'm **GlucoseCare AI**
Ask me about diabetes symptoms, lifestyle, or risks."""
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    if message.content.lower().strip() in ["reset", "/reset", "restart"]:
        cl.user_session.set("agent_state", ensure_state({}))
        await cl.Message(content="Session reset. You can start again.").send()
        return

    state = cl.user_session.get("agent_state") or ensure_state({})
    state["input"] = message.content
    try:
        async with llm_limiter:
            # Use memory with thread_id
            thread_id = cl.user_session.get("thread_id")
            new_state = app.invoke(
                state,
                config={"configurable": {"thread_id": thread_id}},
                checkpointer=memory
                )
            cl.user_session.set("agent_state", new_state)
            await cl.Message(content=new_state.get("output", "No response.")).send()
    except Exception:
        print(traceback.format_exc())
        await cl.Message(content="Something went wrong...").send()
