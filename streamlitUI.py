import streamlit as st

# Page configuration
st.set_page_config(page_title='GlucoseCare AI', page_icon='🩺', layout='wide')

# Page links
#st.page_link("streamlitUI.py", label="HOME")
#st.page_link("pages/GlucoseCare.py", label="Diabetes Consultant")

# ================== Styling ==================
background_color = "#F6FFF0"
text_color = "#2E382E"
highlight_color = "#A3C85A"

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
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {highlight_color};
        }}
        p, div, label {{
            color: {text_color};
            font-size: 1.1rem;
        }}
    </style>
""", unsafe_allow_html=True)

# Main content
st.title("GlucoseCare AI - Your intelligent companion for early diabetes risk prediction and guidance.")
st.header("Empowering People to Understand & Manage Diabetes Effectively")

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

st.markdown("---")
st.caption("Powered by LangGraph + Gemini-1.5-flash + Scikit-Learn | Built with ❤️ for Nigerian farmers")
