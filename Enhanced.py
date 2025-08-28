import os 
import requests
from PIL import Image 
from io import BytesIO
import base64
import matplotlib.pyplot as plt 
import matplotlib.image as mp
from IPython.display import display, Markdown
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS 

import streamlit as st

# OpenAI imports for Vision analysis & CLIP embeddings
import openai

# Set API keys -- replace with your own where needed
os.environ["GOOGLE_API_KEY"] = "AIzaSyDz0IZDs7gISYAOsx-o9DJtni7qx8W1uRA"
openai.api_key = "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"



# --- Model loader & base64 conversion remain as-is ---
def load_model(model_name="gemini-1.5-flash-latest"):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model_name)

def get_image_base64_from_file(file):
    img = Image.open(file)
    buffered = BytesIO()
    img_format = file.name.split(".")[-1].lower()
    img.save(buffered, format=img_format.upper())
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    data_url = f"data:image/{img_format};base64,{b64}"
    return data_url

def analyze_image_with_openai(image_file):
    img_bytes = image_file.read()
    image_file.seek(0)  # Reset for Streamlit reuse
    prompt = (
        "You are an expert in BINAR Handling Group industrial products. "
        "Analyze the uploaded image of a BINAR Handling Group product. "
        "Describe the product in clear, professional language as if you were writing for a product catalog: "
        "- Describe its appearance, visible specifications, and possible features. "
        "- Summarize the key functions, visible controls, and likely applications."
        "- Do NOT just extract the text: interpret visual cues, components, unique structures, etc."
        "- Identify the type of device (e.g., manipulator, gripper, balancer, etc.) and any unique BINAR characteristics if visible."
        " If text is visible on device, incorporate this info in the summary."
    )
    client = openai.OpenAI(api_key=openai.api_key)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for industrial product analysis."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode()}"}
                    ]
                }
            ],
            max_tokens=700
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"OpenAI analysis failed: {str(e)}"

# ------ MODERN, BEAUTIFUL, RESPONSIVE UI -----
st.set_page_config(
    page_title="BINAR Analyzer",
    page_icon="🤖",
    layout="centered",
    menu_items={
        'Get Help': 'https://binar.se/',
        'About': "Industrial product analysis using advanced AI vision models and a beautiful UI."
    }
)

# --- CUSTOM CSS (In-Python, editable) ---
st.markdown("""
<style>
body, html {
    font-family: 'Segoe UI', 'Roboto', sans-serif !important;
}
[data-testid="stAppViewContainer"] > .main {
    background: linear-gradient(135deg, #f6d365 0%, #fda085 100%) !important;
}
header[data-testid="stHeader"] {
    background: linear-gradient(135deg, #48c6ef 0%, #6f86d6 100%);
}
.analysis-card {
    background: #fff7e6;
    border-radius: 18px;
    box-shadow: 0 4px 32px 0 rgba(0,0,0,0.08);
    padding: 2.5em 2em 2em 2em;
    max-width: 540px;
    margin-left: auto; margin-right: auto; margin-top: 1.5em;
}
.upload-area {
    border: 2px dashed #ff9900;
    background: #fffafd;
    border-radius: 16px;
    padding: 2em;
    text-align: center;
    margin-bottom: 1.5em;
}
h1 {
    font-size: 2.8rem;
    font-weight: 800;
    text-shadow: 1px 1px 0 #fff7e6, 0 2px 6px #efefef;
}
.stTabs [data-baseweb="tab"]:first-child { color: #FF930F; }
.stTabs [data-baseweb="tab"]:last-child { color: #004e92; }
.stSpinner {
    color: #FF930F !important;
}
</style>
""", unsafe_allow_html=True)

# ---- SIDEBAR ----
with st.sidebar:
    st.image("https://img.batiweb.com/repo-images/supplier/5821825/binarhandling_logo.jpg", use_container_width=True)
    st.markdown("""
    <h3>How to Use</h3>
    <ul>
    <li>Upload a BINAR product or end-effector image.</li>
    <li>Select AI engine and click <strong>Analyze</strong>!</li>
    <li>Get a catalog-style smart summary.</li>
    </ul>
    <hr>
    <small>Made with 🤍 for industrial engineering.</small>
    """, unsafe_allow_html=True)

# ---- MAIN CONTENT ----
st.markdown(
    "<h1 align='center'>🤖 BINAR Product & End-Effector Analyzer</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='upload-area'>"
    "<p style='font-size:1.27rem;'>"
    "📸 <b>Upload an image of a BINAR Handling Group product or robotic end-effector:</b><br>"
    "<span style='color:#777;'>PNG, JPG or JPEG. (No images are saved.)</span>"
    "</p></div>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    vision_model = load_model()
    image_data_url = get_image_base64_from_file(uploaded_file)

    prompt_1 = """
    You are an expert in robotics and machine vision. Analyze the following image of a robotic end-effector.
    1. Provide a clear description of the end-effector’s appearance, shape, and components.
    2. Identify its possible function(s).
    3. Extract structured information in JSON format:
       - type
       - material
       - key_features
       - possible_applications
    """

    tab1, tab2 = st.tabs([
        "🦾 Gemini: End-Effector Analysis", 
        "🏭 OpenAI GPT: BINAR Product Summary"
    ])

    with tab1:
        if st.button("🔎 Analyze with Gemini", key="gemini"):
            with st.spinner("Gemini model analyzing image..."):
                from langchain_core.messages import HumanMessage
                message = HumanMessage(content=[
                    {"type": "text", "text": prompt_1},
                    {"type": "image_url", "image_url": image_data_url}
                ])
                response = vision_model.invoke([message])
                st.markdown(
                    f"<div class='analysis-card'><h4>Gemini Model Analysis:</h4><div>{response.content}</div></div>",
                    unsafe_allow_html=True,
                )

    with tab2:
        if st.button("🤖 Smart Analyze BINAR Product", key="openai"):
            with st.spinner("OpenAI is analyzing your BINAR product..."):
                desc = analyze_image_with_openai(uploaded_file)
                st.markdown(
                    f"<div class='analysis-card'><h4>OpenAI (GPT-4 Vision) Analysis:</h4>{desc}</div>",
                    unsafe_allow_html=True,
                )

# Optional footer
st.markdown("<br><hr><center><sub>©  25 Aout 2025 BINAR Analyzer. Made with 🧡 using Streamlit, Gemini, and GPT-4 Vision.</sub></center>", unsafe_allow_html=True)