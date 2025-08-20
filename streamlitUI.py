import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import your agent and memory from GlucoseCare.py
from GlucoseCare import app, AgentState, memory

# =============================   Streamlit UI   =============================
st.set_page_config(page_title="GlucoseCare AI", page_icon="🩺", layout="wide")

st.title("GlucoseCare AI - Early Diabetes Risk Consultant")

# --- Session state setup ---
if "agent_state" not in st.session_state:
    st.session_state.agent_state = AgentState(session_id="default_session")
if "messages" not in st.session_state:
    st.session_state.messages = memory.get_history("default_session")

# --- Sidebar ---
with st.sidebar:
    st.header("Controls")
    if st.button("🗑️ Clear All Chat", use_container_width=True):
        memory._reset_table()  # clear persistent SQLite memory
        st.session_state.messages = []
        st.session_state.agent_state = AgentState(session_id="default_session")
        st.experimental_rerun()

    st.markdown("---")
    st.write("**About**")
    st.caption("GlucoseCare AI uses ADA guidelines & a predictive model to "
               "assist with diabetes risk awareness. ⚠️ Not medical advice.")

# --- Chat history display ---
st.subheader("💬 Conversation")

chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.info("No conversation yet. Ask me about diabetes, symptoms, or start risk assessment.")
    else:
        for role, content in st.session_state.messages:
            if role == "user":
                st.chat_message("user").markdown(content)
            elif role == "assistant":
                st.chat_message("assistant").markdown(content)
            elif role == "system":
                st.chat_message("system").markdown(f"_{content}_")

# --- User input ---
st.markdown("---")
prompt = st.chat_input("Ask a question or describe your symptoms...")

if prompt:
    # Update state with user input
    st.session_state.agent_state["user_input"] = prompt

    # Run agent graph
    new_state = app.invoke(st.session_state.agent_state)

    # Append new messages to memory + session state
    st.session_state.messages = memory.get_history(st.session_state.agent_state.session_id)

    # Display assistant response immediately
    if "output" in new_state:
        st.chat_message("assistant").markdown(new_state["output"])

    # ========================== Clinical Dashboard ==========================
    if "features" in new_state and new_state["features"]:
        features = new_state["features"]

        with st.expander("Patient Summary", expanded=False):
            st.write(pd.DataFrame([features]).T.rename(columns={0: "Value"}))

        if "Prediction:" in new_state["output"]:
            try:
                # Extract confidence score from text
                text_out = new_state["output"]
                conf_str = text_out.split("confidence:")[1].split(")")[0].strip()
                confidence = float(conf_str)

                with st.expander("Prediction Visualization", expanded=True):
                    # Confidence progress bar
                    st.write("**Confidence Level**")
                    st.progress(confidence)

                    # Radar chart for symptoms
                    symptom_keys = [k for k in features.keys() if k not in ["Age", "Gender"]]
                    values = [int(features[k]) for k in symptom_keys]

                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=values + [values[0]],  # close the radar loop
                        theta=symptom_keys + [symptom_keys[0]],
                        fill='toself',
                        name="Symptoms"
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=False,
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"Could not parse confidence: {e}")