# streamlit_app.py
import streamlit as st
import uuid, traceback

# Import your agent logic
from agent import app, ensure_state, memory

# =============================   Streamlit Setup   =============================
st.set_page_config(page_title="GlucoseCare AI", page_icon="🩺", layout="wide")

st.sidebar.title("🩺 GlucoseCare AI")
st.sidebar.markdown("""
GlucoseCare AI — *Early Diabetes Risk Consultant*

Ask about:
- Diabetes symptoms
- Lifestyle factors
- Quick risk check
""")

# =============================   Session Management   =============================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "agent_state" not in st.session_state:
    st.session_state.agent_state = ensure_state({})
if "history" not in st.session_state:
    st.session_state.history = []

# =============================   Reset Button   =============================
if st.sidebar.button("🔄 Reset Session"):
    st.session_state.agent_state = ensure_state({})
    st.session_state.history = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()

# =============================   Chat Interface   =============================
st.title("👋 Welcome to GlucoseCare AI")

# Display history
for entry in st.session_state.history:
    role, content = entry["role"], entry["content"]
    if role == "user":
        st.chat_message("user").write(content)
    else:
        st.chat_message("assistant").write(content)

# Input box
if prompt := st.chat_input("Type your message..."):
    # Show user input
    st.chat_message("user").write(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})

    # Update state and run agent
    try:
        st.session_state.agent_state["input"] = prompt
        new_state = app.invoke(
            st.session_state.agent_state,
            config={"configurable": {"thread_id": st.session_state.thread_id}},
            checkpointer=memory
        )
        st.session_state.agent_state = new_state
        response = new_state.get("output", "⚠️ No response.")

    except Exception:
        response = "⚠️ Something went wrong..."
        print(traceback.format_exc())

    # Show assistant output
    st.chat_message("assistant").write(response)
    st.session_state.history.append({"role": "assistant", "content": response})
