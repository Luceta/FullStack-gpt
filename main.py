import json
import streamlit as st
import os
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_community.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser


st.set_page_config(page_title="QuizGPT-Challenge", page_icon="❓")

st.title("Quiz GPT-Turbo")


# Define function schema
function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# Define ChatPromptTemplate for generating questions
questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
            
    Based ONLY on the following context make 10 (TEN) questions minimum to test the user's knowledge about the text.

    Each question should have 4 answers, three of them must be incorrect and one should be correct.

    Context: {context}
    """,
        )
    ]
)

hard_mode_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
    You need to teach more deepl knowledge not just in the text.
    Based ONLY on the following context make 10 (TEN) questions minimum to test the user's knowledge about the text.
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
    You have to create a very tricky problem. 

    Context: {context}
    """,
        )
    ]
)


@st.cache
def get_openai_api_key():
    return st.sidebar.text_input("Enter your OpenAI API key", type="password")


# OpenAI API 키 입력 및 설정
with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="openai_api_key", type="password"
    )
    os.environ["OPENAI_API_KEY"] = openai_api_key
    st.markdown(
        "[GitHub Repository](https://github.com/Luceta/FullStack-gpt/tree/feat/quizGPT)"
    )

if openai_api_key:
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-1106",
        streaming=True,
        callbacks=[
            StreamingStdOutCallbackHandler(),
        ],
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[
            function,
        ],
    )

    easy_mode_chain = {"context": format_docs} | questions_prompt | llm
    hard_mode_chain = {"context": format_docs} | hard_mode_prompt | llm


else:
    st.header("Requirement: enter the OpenAPI key in sidebar")
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_easy_quiz_chain(_docs, topic):
    response = easy_mode_chain.invoke(_docs)
    response = response.additional_kwargs["function_call"]["arguments"]
    return json.loads(response)


@st.cache_data(show_spinner="Making quiz...")
def run_hard_quiz_chain(_docs, topic):
    response = hard_mode_chain.invoke(_docs)
    response = response.additional_kwargs["function_call"]["arguments"]
    return json.loads(response)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=1)
    docs = retriever.get_relevant_documents(term)
    return docs


difficulty = st.selectbox("Select quiz difficulty", ["Easy", "Hard"])

choice = st.selectbox(
    "Choose what you want to use.",
    (
        "File",
        "Wikipedia Article",
    ),
)

docs = None

if choice == "File":
    file = st.file_uploader(
        "Upload a .docx, .txt or .pdf file",
        type=["pdf", "txt", "docx"],
    )
    if file:
        split_file(file)

else:
    topic = st.text_input("Search Wikipedia")
    if topic:
        docs = wiki_search(topic)


if docs:

    if difficulty == "Hard":
        response = run_hard_quiz_chain(docs, topic if topic else file.name)
    else:
        response = run_easy_quiz_chain(docs, topic if topic else file.name)

    with st.form("questionns_form"):
        correct_answers = 0
        total_questions = len(response["questions"])

        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )

            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
                correct_answers += 1

            elif value is not None:
                st.error("Incorrect")
        button = st.form_submit_button()

    if button:
        if correct_answers == total_questions:
            st.balloons()
            st.success(f"All correct!")
        else:
            st.error(
                f"Score: {correct_answers}/{total_questions}. Some answers were incorrect. Try again!"
            )
