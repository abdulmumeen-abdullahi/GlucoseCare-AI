import streamlit as st

# ================== Page Configuration ==================
st.set_page_config(
    page_title='GlucoseCare AI',
    page_icon='🩺',
    layout='wide'
)

# ================== Page Links ==================
# Page links
st.page_link("streamlitUI.py", label="HOME")
#st.page_link("GlucoseCare.py", label="Diabetes Consultant")

# ================== Styling ==================
background_color = "#FFF8F0"
text_color = "#2E2E2E"
highlight_color = "#E63946"

st.markdown(f"""
    <style>
        body {{
            background-color: {background_color};
            color: {text_color};
        }}
        .stApp {{
            background-color: {background_color};
        }}
        .block-container {{
            padding-top: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {highlight_color};
        }}
        p, div, label {{
            color: {text_color};
            font-size: 1.1rem;
        }}
        .stButton>button {{
            background-color: {highlight_color};
            color: white;
            font-weight: bold;
        }}
    </style>
""", unsafe_allow_html=True)

# ================== Home Page ==================
if page == "Home":
    st.title("GlucoseCare AI")
    st.header("Empowering People to Understand & Manage Diabetes")
    
    st.write("""
    Welcome to **GlucoseCare AI**, your intelligent assistant for diabetes awareness, risk prediction, and guidance.
    
    Our AI-powered system can:
    - **Predict your diabetes risk** based on health features like age, sex, lifestyle symptoms.
    - **Consult with an AI assistant** to answer diabetes-related questions in simple, actionable language.
    
    ⚠️ *Disclaimer: GlucoseCare AI is for informational purposes only. It does not replace medical advice from a licensed physician.*
    """)
    
    st.subheader("Why Use GlucoseCare AI?")
    st.markdown("""
    - **Early Risk Awareness**: Get insights into your potential diabetes risk.
    - **Accessible Advice**: Understand complex medical topics in plain language.
    - **Local Relevance**: Advice tailored for everyday Nigerians.
    """)
    
    st.success("Take control of your health today!")
