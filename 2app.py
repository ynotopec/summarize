import os
from dotenv import load_dotenv
import streamlit as st
import validators
import requests
from PyPDF2 import PdfReader
from PIL import Image
from io import BytesIO

# LangChain & ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import WebBaseLoader
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# ─── Configuration ────────────────────────────────────────────────────────────

load_dotenv()
API_BASE   = os.getenv("OPENAI_API_BASE")  # exemple : https://api.openai.com/v1
API_KEY    = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")

# On passe la clé et l’endpoint à LangChain
os.environ["OPENAI_API_BASE"]  = API_BASE
os.environ["OPENAI_API_KEY"]   = API_KEY
os.environ["OPENAI_API_MODEL"] = MODEL_NAME

# ─── Fonctions utilitaires ────────────────────────────────────────────────────

def upload_pdf():
    uploaded_file = st.file_uploader("Choisissez un fichier PDF", type=["pdf"])
    if uploaded_file:
        reader = PdfReader(uploaded_file)
        return "".join(page.extract_text() or "" for page in reader.pages)
    return None

def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=512)
    return [Document(page_content=chunk) for chunk in splitter.split_text(text)]

def docs_summary(docs):
    llm = ChatOpenAI(temperature=0.2, model_name=MODEL_NAME)
    chain = load_summarize_chain(llm=llm, chain_type="map_reduce")
#refine")
    return chain.run(docs)

def handle_text(text: str):
    return docs_summary(split_text(text))

def handle_url(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load_and_split()
    return docs_summary(docs)

# Prompt de traduction
system_tmplt = "You are a helpful assistant that translates {input_language} to {output_language}."
human_tmplt  = "Translate this sentence from {input_language} to {output_language}. {text}"
chat_prompt  = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_tmplt),
    HumanMessagePromptTemplate.from_template(human_tmplt),
])

def translate_en_to_fr(text: str):
    chat = ChatOpenAI(temperature=0, model_name=MODEL_NAME)
    msgs = chat_prompt.format_prompt(
        input_language="English",
        output_language="French",
        text=text
    ).to_messages()
    return chat(msgs).content

def generate_image_via_api(prompt: str):
    """
    Appel HTTP au service tiers txt2image.
    """
    endpoint = "https://api-txt2image.c0.ai-dev.numerique-interieur.com/generate_image/"
    payload  = {"prompt": prompt}
    resp     = requests.post(endpoint, json=payload, timeout=30)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content))

# ─── Interface Streamlit ─────────────────────────────────────────────────────

st.title("Résumé et Illustration Automatique (sans dépendance OpenAI)")
st.write("Entrez une URL, du texte libre ou importez un PDF :")

user_input = st.text_area("URL ou texte :")
pdf_text   = upload_pdf()
if pdf_text:
    user_input = pdf_text

if user_input:
    with st.spinner("Traitement en cours..."):
        summary_en = handle_url(user_input) if validators.url(user_input) else handle_text(user_input)
        summary_fr = translate_en_to_fr(summary_en)
        # on ajoute éventuellement un style d'illustration
        img_prompt = f"Illustrate this summary: {summary_en}"
        img        = generate_image_via_api(img_prompt)

    st.subheader("Résumé")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(summary_fr)
    with col2:
        st.image(img, caption="Illustration générée")
