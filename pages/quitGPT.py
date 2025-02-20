import json
import os

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser


st.title("QuizGPT-Challenage")
st.sidebar.title("QuizGPT Settings")
