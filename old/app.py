# web ui
import streamlit as st
# summarize
from langchain.chains.summarize import load_summarize_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv #python-dotenv
# translate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

#import os
import requests
import subprocess
import re

import json

from PIL import Image
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

os.environ["OPENAI_API_MODEL"]="chat-leger"
os.environ["OPENAI_API_BASE"]="https://api-ai.ai-dev.numerique-interieur.com/v1"
os.environ["OPENAI_API_KEY"]="sk-X1UNQff2L9poj7RBx3HYig"

openai_api_model=os.environ["OPENAI_API_MODEL"]
openai_api_base=os.environ["OPENAI_API_BASE"]
openai_api_key=os.environ["OPENAI_API_KEY"]

from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

def handle_text(text):
    # Split the text into chunks
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(text)
    
    # Create document objects for each chunk
    docs = [Document(page_content=t) for t in texts]
    
    result = docs_summary(docs)
    return result

#web ui (streamlit)
# Function to scrape and summarize the web page
def handle_url(url):
    loader = WebBaseLoader(url)
    docs = loader.load_and_split()

    result = docs_summary(docs)
    return result

def docs_summary(docs):
    # Initialize the OpenAI model and load the summarize chain
    llm = ChatOpenAI(temperature=0,model_name=openai_api_model)

    chain = load_summarize_chain(llm, chain_type="map_reduce")
    
    # Generate the summary
    summary = chain.run(docs)
    return summary

# Function to process text file and generate prompt
def generate_prompt(text):
    words = re.sub(r'\W+', ' ', text).lower().split()
    unique_words = sorted(set(word for word in words if 2 <= len(word) <= 12))
    return "Give an image prompt from context :\n" + text

# Function to make an API request
def make_api_request(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
    }
    data = {
        "model": openai_api_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }
    response = requests.post(f"{os.environ['OPENAI_API_BASE']}/chat/completions", json=data, headers=headers)
    return response.json()

#chat = ChatOpenAI(temperature=0.2,model_name="vicuna")
chat = ChatOpenAI(temperature=0.2,model_name=openai_api_model)
template = (
    "You are a helpful assistant that translates {input_language} to {output_language}."
)
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "Translate this sentence from {input_language} to {output_language}. {text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Streamlit interface
st.title('Summary Generator')
st.write('https://github.com/ynotopec/summarize')

# Input for URL or text
#url = st.text_area("Enter the URL or text to summarize")
# Display the summary if URL is entered
#if url:
#    with st.spinner('Processing...'):
#        summary_english = get_summary(url)

import validators

user_input = st.text_area("Enter a URL or Text:")

if user_input:
    with st.spinner('Processing...'):
        if validators.url(user_input):
            summary_english = handle_url(user_input)
        else:
            summary_english = handle_text(user_input)

        #text_input="Write in markdown high style.\n" + summary_english
        text_input="Summarize this clearly and concisely:\n\n" + summary_english
        #text_input="Write in markdown high style a summary.\n" + summary_english
        #summary_english = make_api_request(text_input)["choices"][0]["message"]["content"]

        # translate
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        # get a chat completion from the formatted messages
        summary = chat(
            chat_prompt.format_prompt(
                input_language="English", output_language="French", text=summary_english
            ).to_messages()
        ).content

    st.subheader('Summary')

    # Example of processing a single iteration
    prompt = generate_prompt(summary_english)
    data = make_api_request(prompt)
    content = data["choices"][0]["message"]["content"]

    user_input = content
# + ',illustration style'

    if user_input:
      # Create two columns with the second column being smaller for the thumbnail
      col1, col2 = st.columns([3, 1])

      # Use the first column to display your text
      with col1:
        st.write(summary, unsafe_allow_html=True)

      # Use the second column to display the thumbnail image
      with col2:
            response = requests.post('https://api-txt2image.c0.ai-dev.numerique-interieur.com/generate_image/', json={'prompt': user_input})
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                st.image(image, caption=content) #, width=150)
            else:
                st.error(f"Failed to generate image.")
