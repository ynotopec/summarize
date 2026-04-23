# web ui
import streamlit as st
# summarize
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv #python-dotenv
# translate
from langchain.chat_models import ChatOpenAI
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

#web ui (streamlit)
# Function to scrape and summarize the web page
def get_summary(url):
    loader = WebBaseLoader(url)

    docs = loader.load_and_split()

    llm = ChatOpenAI(temperature=0.2)
#batch_size=1,temperature=0.1)
    chain = load_summarize_chain(llm, verbose = True, chain_type="refine")
#map_reduce")

    result=chain.run(docs)

    return result

# Function to process text file and generate prompt
def generate_prompt(text):
#    with open(file_path, 'r') as file:
#        text = file.read()
    words = re.sub(r'\W+', ' ', text).lower().split()
    unique_words = sorted(set(word for word in words if 2 <= len(word) <= 12))
    #return "Give 4 tags separated by commas about words below (DO NOT EXPLAIN) :\n" + ' '.join(unique_words[:256])
    return "Give an image prompt from context :\n" + text
#' '.join(unique_words[:256])
    #return "Give the meaning these words :\n" + ' '.join(unique_words[:256])

# Function to make an API request
def make_api_request(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
    }
    data = {
        "model": "gpt-3.5-turbo-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }
    response = requests.post(f"{os.environ['OPENAI_API_BASE']}/chat/completions", json=data, headers=headers)
    return response.json()

# translate def
chat = ChatOpenAI(temperature=0)
template = (
    "You are a helpful assistant that translates {input_language} to {output_language}."
)
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "Translate this sentence from {input_language} to {output_language}. {text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Streamlit interface
st.title('URL Summary Generator')
st.write('https://github.com/ynotopec/summarize')

# Input for URL
url = st.text_input("Enter the URL to summarize")

# Display the summary if URL is entered
if url:
    with st.spinner('Processing...'):
        summary_english = get_summary(url)
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
#        summary = chat("Affichez sous forme de liste à puces ceci. {summary}")
    st.subheader('Summary')
    #st.write(summary)

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
      #    st.image("path_to_your_thumbnail.jpg", width=100)  # Adjust the width as needed
            response = requests.post('https://api-txt2image.c0.ai-dev.numerique-interieur.com/generate_image/', json={'prompt': user_input})
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                st.image(image, caption=content) #, width=150)
            else:
                st.error(f"Failed to generate image.")
