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

# Load environment variables from .env file
load_dotenv()

#web ui (streamlit)
# Function to scrape and summarize the web page
def get_summary(url):
    loader = WebBaseLoader(url)

    docs = loader.load_and_split()

    llm = ChatOpenAI(temperature=0.2)
    chain = load_summarize_chain(llm, verbose = True, chain_type="refine")

    result=chain.run(docs)

    return result

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
    st.subheader('Summary')
    st.write(summary)
